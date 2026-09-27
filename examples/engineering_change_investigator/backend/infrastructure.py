from __future__ import annotations

import json
import os
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any
from uuid import UUID

import httpx
import psycopg
from behaviorweave import BehaviorState
from feedback_manager import FeedbackEvent, FeedbackStatus
from feedback_manager.contracts.store import FeedbackQuery, FeedbackStore
from feedback_manager.core.lifecycle import validate_transition
from feedback_manager.errors import FeedbackNotFoundError, FeedbackStoreError
from langchain_core.documents import Document
from langchain_core.documents.compressor import BaseDocumentCompressor
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from mcp_capability_router import (
    Capability,
    CapabilityRegistryBase,
    CapabilityType,
    Prompt,
    Tool,
)
from mcp_capability_router import (
    Resource as MCPCapabilityResource,
)
from pgvector import Vector
from pgvector.psycopg import register_vector
from refresh_engine import ResourceState
from refresh_engine.api.abc import StateStoreABC, StateTransactionABC
from refresh_engine.core.models import ResourceFingerprint

EMBEDDING_DIMENSION = 384
CANONICAL_DOCUMENT_ID = re.compile(r"(?:ADR|RUNBOOK|RELEASE)-\d+")


class OllamaEmbeddings(Embeddings):
    def __init__(
        self,
        base_url: str | None = None,
        model_name: str | None = None,
    ) -> None:
        self.base_url = (
            base_url or os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
        ).rstrip("/")
        self.model_name = model_name or os.getenv("OLLAMA_EMBED_MODEL", "all-minilm")

    def _embed(self, texts: list[str]) -> list[list[float]]:
        response = httpx.post(
            f"{self.base_url}/api/embed",
            json={"model": self.model_name, "input": texts},
            timeout=120,
        )
        response.raise_for_status()
        embeddings = response.json().get("embeddings")
        if not isinstance(embeddings, list) or len(embeddings) != len(texts):
            raise RuntimeError("Ollama returned an invalid embedding response")
        return embeddings

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text])[0]


class Database:
    def __init__(self, dsn: str, embeddings: OllamaEmbeddings) -> None:
        self.dsn = dsn
        self.embeddings = embeddings

    def connect(self) -> psycopg.Connection[Any]:
        connection = psycopg.connect(self.dsn)
        register_vector(connection)
        return connection

    def initialize(self, workspace: Path) -> None:
        with psycopg.connect(self.dsn) as connection:
            connection.execute("CREATE EXTENSION IF NOT EXISTS vector")
        with self.connect() as connection:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS engineering_documents (
                    document_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    source_path TEXT NOT NULL,
                    content TEXT NOT NULL,
                    embedding VECTOR(384) NOT NULL,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
                );
                CREATE TABLE IF NOT EXISTS engineering_feedback (
                    feedback_id UUID PRIMARY KEY,
                    idempotency_key TEXT UNIQUE,
                    payload JSONB NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL
                );
                CREATE TABLE IF NOT EXISTS engineering_behavior_state (
                    state_key TEXT PRIMARY KEY,
                    payload JSONB NOT NULL,
                    version BIGINT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS engineering_mcp_capabilities (
                    capability_id TEXT PRIMARY KEY,
                    server_id TEXT NOT NULL,
                    capability_type TEXT NOT NULL,
                    payload JSONB NOT NULL
                );
                CREATE INDEX IF NOT EXISTS engineering_mcp_server_idx
                ON engineering_mcp_capabilities(server_id, capability_type);
                CREATE TABLE IF NOT EXISTS engineering_refresh_state (
                    resource_id TEXT PRIMARY KEY,
                    payload JSONB NOT NULL
                );
                """)
        self.index_workspace(workspace)

    def index_workspace(self, workspace: Path) -> int:
        documents = [
            path
            for path in sorted(workspace.glob("*.md"))
            if CANONICAL_DOCUMENT_ID.fullmatch(path.stem)
        ]
        texts = [path.read_text(encoding="utf-8") for path in documents]
        vectors = self.embeddings.embed_documents(texts)
        with self.connect() as connection:
            connection.execute(
                """
                DELETE FROM engineering_documents
                WHERE NOT (document_id = ANY(%s))
                """,
                ([path.stem for path in documents],),
            )
            for path, content, embedding in zip(documents, texts, vectors, strict=True):
                title = content.splitlines()[0].lstrip("# ").strip() or path.stem
                connection.execute(
                    """
                    INSERT INTO engineering_documents
                        (document_id, title, source_path, content, embedding)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (document_id) DO UPDATE SET
                        title = EXCLUDED.title,
                        source_path = EXCLUDED.source_path,
                        content = EXCLUDED.content,
                        embedding = EXCLUDED.embedding,
                        updated_at = now()
                    """,
                    (
                        path.stem,
                        title,
                        str(path),
                        content,
                        Vector(embedding),
                    ),
                )
        return len(documents)

    def search(self, query: str, limit: int = 5) -> list[Document]:
        embedding = Vector(self.embeddings.embed_query(query))
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT document_id, title, source_path, content,
                       1 - (embedding <=> %s) AS score
                FROM engineering_documents
                ORDER BY embedding <=> %s
                LIMIT %s
                """,
                (embedding, embedding, limit),
            ).fetchall()
        return [
            Document(
                page_content=row[3],
                metadata={
                    "document_id": row[0],
                    "title": row[1],
                    "source_path": row[2],
                    "score": float(row[4]),
                    "capability_name": "semantic_change_search",
                },
            )
            for row in rows
        ]

    def counts(self) -> dict[str, int]:
        tables = {
            "documents": "engineering_documents",
            "feedback": "engineering_feedback",
            "behavior_states": "engineering_behavior_state",
            "mcp_capabilities": "engineering_mcp_capabilities",
            "refresh_states": "engineering_refresh_state",
            "xai_records": "langgraph_xai_records",
        }
        result: dict[str, int] = {}
        with self.connect() as connection:
            for name, table in tables.items():
                exists = connection.execute(
                    "SELECT to_regclass(%s)", (table,)
                ).fetchone()[0]
                result[name] = (
                    connection.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
                    if exists
                    else 0
                )
        return result


class PgVectorRetriever(BaseRetriever):
    database: Any
    limit: int = 5

    def _get_relevant_documents(
        self, query: str, *, run_manager: Any
    ) -> list[Document]:
        del run_manager
        return self.database.search(query, self.limit)


class EvidenceReranker(BaseDocumentCompressor):
    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Any = None,
    ) -> Sequence[Document]:
        del callbacks
        terms = set(query.casefold().split())
        return tuple(
            sorted(
                documents,
                key=lambda document: (
                    -len(terms & set(document.page_content.casefold().split())),
                    -float(document.metadata.get("score", 0)),
                ),
            )
        )


class PostgresFeedbackStore(FeedbackStore):
    def __init__(self, dsn: str) -> None:
        self.dsn = dsn

    async def create(self, feedback: FeedbackEvent) -> FeedbackEvent:
        async with await psycopg.AsyncConnection.connect(self.dsn) as connection:
            if feedback.idempotency_key:
                existing = await (
                    await connection.execute(
                        """
                        SELECT payload FROM engineering_feedback
                        WHERE idempotency_key = %s
                        """,
                        (feedback.idempotency_key,),
                    )
                ).fetchone()
                if existing:
                    return FeedbackEvent.model_validate(existing[0])
            try:
                await connection.execute(
                    """
                    INSERT INTO engineering_feedback
                        (feedback_id, idempotency_key, payload, created_at)
                    VALUES (%s, %s, %s::jsonb, %s)
                    """,
                    (
                        feedback.feedback_id,
                        feedback.idempotency_key,
                        feedback.model_dump_json(),
                        feedback.created_at,
                    ),
                )
            except psycopg.errors.UniqueViolation as exc:
                raise FeedbackStoreError(
                    "feedback event already exists",
                    feedback_id=feedback.feedback_id,
                ) from exc
        return feedback

    async def get(self, feedback_id: UUID) -> FeedbackEvent | None:
        async with await psycopg.AsyncConnection.connect(self.dsn) as connection:
            row = await (
                await connection.execute(
                    "SELECT payload FROM engineering_feedback WHERE feedback_id = %s",
                    (feedback_id,),
                )
            ).fetchone()
        return FeedbackEvent.model_validate(row[0]) if row else None

    async def update(self, feedback: FeedbackEvent) -> FeedbackEvent:
        async with await psycopg.AsyncConnection.connect(self.dsn) as connection:
            cursor = await connection.execute(
                """
                UPDATE engineering_feedback SET payload = %s::jsonb
                WHERE feedback_id = %s
                """,
                (feedback.model_dump_json(), feedback.feedback_id),
            )
            if cursor.rowcount != 1:
                raise FeedbackNotFoundError(
                    "cannot update unknown feedback",
                    feedback_id=feedback.feedback_id,
                )
        return feedback

    async def transition(
        self, feedback_id: UUID, status: FeedbackStatus
    ) -> FeedbackEvent:
        current = await self.get(feedback_id)
        if current is None:
            raise FeedbackNotFoundError(
                "cannot transition unknown feedback", feedback_id=feedback_id
            )
        validate_transition(feedback_id, current.status, status)
        return await self.update(current.with_status(status))

    async def query(self, query: FeedbackQuery) -> Sequence[FeedbackEvent]:
        events = await self.list()
        matches = [event for event in events if query.matches(event)]
        return matches[: query.limit] if query.limit is not None else matches

    async def list(self) -> Sequence[FeedbackEvent]:
        async with await psycopg.AsyncConnection.connect(self.dsn) as connection:
            rows = await (await connection.execute("""
                    SELECT payload FROM engineering_feedback
                    ORDER BY created_at, feedback_id
                    """)).fetchall()
        return [FeedbackEvent.model_validate(row[0]) for row in rows]


def _behavior_state(payload: Mapping[str, Any]) -> BehaviorState:
    values = dict(payload)
    for name in (
        "first_seen",
        "last_seen",
        "window_start",
        "window_end",
        "cooldown_until",
    ):
        if values.get(name):
            values[name] = datetime.fromisoformat(values[name])
    values["intervention_history"] = tuple(values["intervention_history"])
    values["seen_event_ids"] = tuple(values["seen_event_ids"])
    return BehaviorState(**values)


class PostgresBehaviorStateStore:
    def __init__(self, dsn: str) -> None:
        self.dsn = dsn

    def get(self, key: str) -> BehaviorState | None:
        with psycopg.connect(self.dsn) as connection:
            row = connection.execute(
                "SELECT payload FROM engineering_behavior_state WHERE state_key = %s",
                (key,),
            ).fetchone()
        return _behavior_state(row[0]) if row else None

    def put(self, key: str, state: BehaviorState) -> None:
        with psycopg.connect(self.dsn) as connection:
            connection.execute(
                """
                INSERT INTO engineering_behavior_state(state_key, payload, version)
                VALUES (%s, %s::jsonb, %s)
                ON CONFLICT (state_key) DO UPDATE SET
                    payload = EXCLUDED.payload, version = EXCLUDED.version
                """,
                (key, state.json(), state.version),
            )

    def update(
        self,
        key: str,
        fn: Callable[[BehaviorState | None], BehaviorState],
    ) -> BehaviorState:
        with psycopg.connect(self.dsn) as connection:
            row = connection.execute(
                """
                SELECT payload FROM engineering_behavior_state
                WHERE state_key = %s FOR UPDATE
                """,
                (key,),
            ).fetchone()
            state = fn(_behavior_state(row[0]) if row else None)
            connection.execute(
                """
                INSERT INTO engineering_behavior_state(state_key, payload, version)
                VALUES (%s, %s::jsonb, %s)
                ON CONFLICT (state_key) DO UPDATE SET
                    payload = EXCLUDED.payload, version = EXCLUDED.version
                """,
                (key, state.json(), state.version),
            )
        return state

    def delete(self, key: str) -> None:
        with psycopg.connect(self.dsn) as connection:
            connection.execute(
                "DELETE FROM engineering_behavior_state WHERE state_key = %s",
                (key,),
            )


def _capability(payload: Mapping[str, Any]) -> Capability:
    values = dict(payload)
    values["tags"] = frozenset(values.get("tags", ()))
    values["depends_on"] = frozenset(values.get("depends_on", ()))
    capability_type = CapabilityType(values.pop("type"))
    implementation = {
        CapabilityType.TOOL: Tool,
        CapabilityType.RESOURCE: MCPCapabilityResource,
        CapabilityType.PROMPT: Prompt,
    }[capability_type]
    return implementation(**values)


class PostgresMCPRegistry(CapabilityRegistryBase):
    def __init__(self, dsn: str) -> None:
        self.dsn = dsn

    async def upsert_many(self, capabilities: Iterable[Capability]) -> None:
        async with await psycopg.AsyncConnection.connect(self.dsn) as connection:
            for capability in capabilities:
                payload = asdict(capability)
                payload["type"] = capability.type.value
                await connection.execute(
                    """
                    INSERT INTO engineering_mcp_capabilities
                        (capability_id, server_id, capability_type, payload)
                    VALUES (%s, %s, %s, %s::jsonb)
                    ON CONFLICT (capability_id) DO UPDATE SET
                        server_id = EXCLUDED.server_id,
                        capability_type = EXCLUDED.capability_type,
                        payload = EXCLUDED.payload
                    """,
                    (
                        capability.capability_id,
                        capability.server_id,
                        capability.type.value,
                        json.dumps(payload, default=list),
                    ),
                )

    async def remove_missing(self, server_id: str, current_ids: set[str]) -> None:
        async with await psycopg.AsyncConnection.connect(self.dsn) as connection:
            if current_ids:
                await connection.execute(
                    """
                    DELETE FROM engineering_mcp_capabilities
                    WHERE server_id = %s AND NOT (capability_id = ANY(%s))
                    """,
                    (server_id, list(current_ids)),
                )
            else:
                await connection.execute(
                    "DELETE FROM engineering_mcp_capabilities WHERE server_id = %s",
                    (server_id,),
                )

    async def remove(self, capability_id: str) -> None:
        async with await psycopg.AsyncConnection.connect(self.dsn) as connection:
            await connection.execute(
                """
                DELETE FROM engineering_mcp_capabilities
                WHERE capability_id = %s
                """,
                (capability_id,),
            )

    async def get(self, capability_id: str) -> Capability | None:
        async with await psycopg.AsyncConnection.connect(self.dsn) as connection:
            row = await (
                await connection.execute(
                    """
                    SELECT payload FROM engineering_mcp_capabilities
                    WHERE capability_id = %s
                    """,
                    (capability_id,),
                )
            ).fetchone()
        return _capability(row[0]) if row else None

    async def list(
        self,
        *,
        server_id: str | None = None,
        type: CapabilityType | None = None,
    ) -> list[Capability]:
        clauses: list[str] = []
        values: list[Any] = []
        if server_id is not None:
            clauses.append("server_id = %s")
            values.append(server_id)
        if type is not None:
            clauses.append("capability_type = %s")
            values.append(type.value)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        async with await psycopg.AsyncConnection.connect(self.dsn) as connection:
            rows = await (
                await connection.execute(
                    f"""
                    SELECT payload FROM engineering_mcp_capabilities
                    {where} ORDER BY capability_id
                    """,
                    values,
                )
            ).fetchall()
        return [_capability(row[0]) for row in rows]


def _resource_state(payload: Mapping[str, Any]) -> ResourceState:
    values = dict(payload)
    values["fingerprint"] = ResourceFingerprint(**values["fingerprint"])
    values["snapshot_metadata"] = dict(values["snapshot_metadata"])
    values["dependencies"] = frozenset(values["dependencies"])
    values["refreshed_at"] = datetime.fromisoformat(values["refreshed_at"])
    return ResourceState(**values)


class PostgresRefreshTransaction(StateTransactionABC):
    def __init__(self, dsn: str) -> None:
        self.dsn = dsn
        self.puts: dict[str, ResourceState] = {}
        self.deletes: set[str] = set()
        self.closed = False

    async def put(self, state: ResourceState) -> None:
        self.puts[state.resource_id] = state
        self.deletes.discard(state.resource_id)

    async def delete(self, resource_id: str) -> None:
        self.deletes.add(resource_id)
        self.puts.pop(resource_id, None)

    async def commit(self) -> None:
        async with await psycopg.AsyncConnection.connect(self.dsn) as connection:
            for resource_id in self.deletes:
                await connection.execute(
                    "DELETE FROM engineering_refresh_state WHERE resource_id = %s",
                    (resource_id,),
                )
            for state in self.puts.values():
                payload = {
                    "resource_id": state.resource_id,
                    "fingerprint": asdict(state.fingerprint),
                    "version": state.version,
                    "cheap_indicator": state.cheap_indicator,
                    "snapshot_metadata": dict(state.snapshot_metadata),
                    "dependencies": list(state.dependencies),
                    "refreshed_at": state.refreshed_at.isoformat(),
                }
                await connection.execute(
                    """
                    INSERT INTO engineering_refresh_state(resource_id, payload)
                    VALUES (%s, %s::jsonb)
                    ON CONFLICT (resource_id) DO UPDATE
                    SET payload = EXCLUDED.payload
                    """,
                    (state.resource_id, json.dumps(payload)),
                )
        self.closed = True

    async def rollback(self) -> None:
        self.puts.clear()
        self.deletes.clear()
        self.closed = True


class PostgresRefreshStore(StateStoreABC):
    def __init__(self, dsn: str) -> None:
        self.dsn = dsn

    async def load_all(self) -> Mapping[str, ResourceState]:
        async with await psycopg.AsyncConnection.connect(self.dsn) as connection:
            rows = await (
                await connection.execute(
                    "SELECT resource_id, payload FROM engineering_refresh_state"
                )
            ).fetchall()
        return MappingProxyType({row[0]: _resource_state(row[1]) for row in rows})

    def transaction(self) -> PostgresRefreshTransaction:
        return PostgresRefreshTransaction(self.dsn)
