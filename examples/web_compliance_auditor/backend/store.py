"""PostgreSQL persistence and local Ollama policy retrieval."""

from __future__ import annotations

import json
import os
from pathlib import Path
from uuid import UUID, uuid4

import httpx
from pgvector import Vector
from pgvector.psycopg import register_vector
from psycopg import connect
from psycopg.rows import dict_row

from .models import AuditReport, Evidence

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


class OllamaEmbeddings:
    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
    ) -> None:
        self.base_url = (
            base_url or os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
        ).rstrip("/")
        self.model = model or os.getenv("OLLAMA_EMBED_MODEL", "all-minilm")

    def embed(self, texts: list[str]) -> list[list[float]]:
        response = httpx.post(
            f"{self.base_url}/api/embed",
            json={"model": self.model, "input": texts},
            timeout=120,
        )
        response.raise_for_status()
        embeddings = response.json().get("embeddings")
        if not isinstance(embeddings, list) or len(embeddings) != len(texts):
            raise RuntimeError("Ollama returned an invalid embedding response")
        return embeddings


class PolicyStore:
    def __init__(self, database_url: str | None = None) -> None:
        self.database_url = database_url or os.getenv("DATABASE_URL")
        self._embedder: OllamaEmbeddings | None = None

    @property
    def embedder(self) -> OllamaEmbeddings:
        if self._embedder is None:
            self._embedder = OllamaEmbeddings()
        return self._embedder

    def _vector(self, text: str) -> list[float]:
        return self.embedder.embed([text])[0]

    def initialize(self) -> None:
        if not self.database_url:
            return
        with connect(self.database_url) as db:
            db.execute("CREATE EXTENSION IF NOT EXISTS vector")
            db.execute("""CREATE TABLE IF NOT EXISTS policies (
                id TEXT PRIMARY KEY, title TEXT NOT NULL, body TEXT NOT NULL,
                embedding vector(384) NOT NULL)""")
            db.execute("""CREATE TABLE IF NOT EXISTS audits (
                id UUID PRIMARY KEY, url TEXT NOT NULL, report JSONB NOT NULL,
                evidence JSONB NOT NULL, provenance_id TEXT NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now())""")
            db.execute("""CREATE TABLE IF NOT EXISTS audit_feedback (
                id UUID PRIMARY KEY, audit_id UUID NOT NULL REFERENCES audits(id),
                rating TEXT NOT NULL, correction TEXT,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now())""")
            register_vector(db)
            count = db.execute("SELECT count(*) FROM policies").fetchone()[0]
            if count == 0:
                for policy in json.loads((DATA_DIR / "policies.json").read_text()):
                    db.execute(
                        "INSERT INTO policies VALUES (%s, %s, %s, %s)",
                        (
                            policy["id"],
                            policy["title"],
                            policy["body"],
                            Vector(self._vector(policy["body"])),
                        ),
                    )
            db.commit()

    def search(self, query: str, policy_ids: list[str]) -> list[Evidence]:
        if not self.database_url:
            raise RuntimeError("DATABASE_URL is required for policy retrieval")
        vector = Vector(self._vector(query))
        with connect(self.database_url, row_factory=dict_row) as db:
            register_vector(db)
            clause, values = "", [vector]
            if policy_ids:
                clause = "WHERE id = ANY(%s)"
                values.append(policy_ids)
            rows = db.execute(
                f"""SELECT id, title, body, 1 - (embedding <=> %s) AS similarity
                FROM policies {clause} ORDER BY embedding <=> %s LIMIT 8""",
                [*values, vector],
            ).fetchall()
        return [
            Evidence(
                policy_id=row["id"],
                title=row["title"],
                excerpt=row["body"][:700],
                similarity=max(0.0, min(1.0, float(row["similarity"]))),
            )
            for row in rows
        ]

    def save_audit(
        self,
        url: str,
        report: AuditReport,
        evidence: list[Evidence],
        provenance_id: str,
    ) -> UUID:
        if not self.database_url:
            raise RuntimeError("DATABASE_URL is required for audit persistence")
        audit_id = uuid4()
        with connect(self.database_url) as db:
            db.execute(
                """
                INSERT INTO audits (id, url, report, evidence, provenance_id)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (
                    audit_id,
                    url,
                    json.dumps(report.model_dump()),
                    json.dumps([item.model_dump() for item in evidence]),
                    provenance_id,
                ),
            )
            db.commit()
        return audit_id

    def save_feedback(
        self, audit_id: UUID, rating: str, correction: str | None
    ) -> UUID:
        if not self.database_url:
            raise RuntimeError("DATABASE_URL is required for feedback persistence")
        feedback_id = uuid4()
        with connect(self.database_url) as db:
            db.execute(
                """
                INSERT INTO audit_feedback (id, audit_id, rating, correction)
                VALUES (%s, %s, %s, %s)
                """,
                (feedback_id, audit_id, rating, correction),
            )
            db.commit()
        return feedback_id
