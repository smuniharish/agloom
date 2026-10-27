"""Engineering evidence: a pgvector index kept in sync with the workspace.

`RefreshEngine` discovers the canonical workspace documents and (re)indexes
only what changed; the index serves the `semantic_change_search` tool and the
capability router's retriever.
"""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from typing import Any, Protocol

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.documents.compressor import BaseDocumentCompressor
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from langchain_core.tools import BaseTool, tool
from pgvector import Vector
from psycopg_pool import ConnectionPool
from refresh_engine import (
    DiscoveryResult,
    PlanAction,
    RefreshRequest,
    Resource,
    ResourceSnapshot,
)

from .governance import EVIDENCE_SEARCH

DOCUMENT_ID = re.compile(r"(?:ADR|RUNBOOK|RELEASE)-\d+")
EMBEDDING_DIMENSIONS = 384


class EvidenceSearch(Protocol):
    """Similarity search over indexed evidence documents."""

    def search(self, query: str, limit: int = 5) -> list[Document]:
        """Return the documents most similar to `query`, best first."""
        ...


class EvidenceStore(EvidenceSearch, Protocol):
    """An evidence index that the refresh operation maintains."""

    def upsert(self, document_id: str, title: str, source: str, content: str) -> None:
        """Index or re-index one document."""
        ...

    def delete(self, document_id: str) -> None:
        """Remove one document from the index."""
        ...


class EvidenceIndex:
    """Evidence documents and their embeddings in PostgreSQL with pgvector."""

    def __init__(self, pool: ConnectionPool, embeddings: Embeddings) -> None:
        self._pool = pool
        self._embeddings = embeddings

    def initialize(self) -> None:
        """Create the document table."""
        with self._pool.connection() as connection:
            connection.execute(f"""
                CREATE TABLE IF NOT EXISTS engineering_documents (
                    document_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    source TEXT NOT NULL,
                    content TEXT NOT NULL,
                    embedding VECTOR({EMBEDDING_DIMENSIONS}) NOT NULL,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """)

    def upsert(self, document_id: str, title: str, source: str, content: str) -> None:
        """Embed and store one document, replacing an earlier version."""
        embedding = Vector(self._embeddings.embed_documents([content])[0])
        with self._pool.connection() as connection:
            connection.execute(
                """
                INSERT INTO engineering_documents
                    (document_id, title, source, content, embedding)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (document_id) DO UPDATE SET
                    title = EXCLUDED.title,
                    source = EXCLUDED.source,
                    content = EXCLUDED.content,
                    embedding = EXCLUDED.embedding,
                    updated_at = now()
                """,
                (document_id, title, source, content, embedding),
            )

    def delete(self, document_id: str) -> None:
        """Remove one document."""
        with self._pool.connection() as connection:
            connection.execute(
                "DELETE FROM engineering_documents WHERE document_id = %s",
                (document_id,),
            )

    def search(self, query: str, limit: int = 5) -> list[Document]:
        """Return the documents closest to the query by cosine distance."""
        embedding = Vector(self._embeddings.embed_query(query))
        with self._pool.connection() as connection:
            rows = connection.execute(
                """
                SELECT document_id, title, source, content,
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
                    "source": row[2],
                    "score": float(row[4]),
                    "capability_name": EVIDENCE_SEARCH,
                },
            )
            for row in rows
        ]


class EvidenceRetriever(BaseRetriever):
    """Routes `semantic_change_search` to tasks with relevant indexed evidence."""

    index: Any
    limit: int = 5

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        del run_manager
        return self.index.search(query, self.limit)


class EvidenceReranker(BaseDocumentCompressor):
    """Orders candidate capabilities by the terms they share with the task."""

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Any = None,
    ) -> Sequence[Document]:
        """Return the documents with the most shared terms first."""
        del callbacks
        terms = set(query.casefold().split())
        return sorted(
            documents,
            key=lambda document: (
                -len(terms & set(document.page_content.casefold().split())),
                -float(document.metadata.get("score", 0)),
            ),
        )


def evidence_search_tool(index: EvidenceSearch) -> BaseTool:
    """Return the `semantic_change_search` tool backed by `index`."""

    @tool(EVIDENCE_SEARCH)
    def semantic_change_search(query: str) -> str:
        """Search indexed ADRs, release records, and runbooks for evidence."""
        return json.dumps(
            [
                {
                    "document_id": document.metadata["document_id"],
                    "score": round(float(document.metadata["score"]), 4),
                    "content": document.page_content,
                }
                for document in index.search(query)
            ]
        )

    return semantic_change_search


class WorkspaceSource:
    """Discovers the canonical evidence documents of the workspace."""

    def __init__(self, workspace: Path) -> None:
        self._workspace = workspace

    async def discover(self) -> DiscoveryResult:
        """List the `ADR-*`, `RELEASE-*`, and `RUNBOOK-*` documents."""

        async def resources() -> AsyncIterator[Resource]:
            for path in sorted(self._workspace.glob("*.md")):
                if DOCUMENT_ID.fullmatch(path.stem):
                    yield Resource(
                        path.stem,
                        version=str(path.stat().st_mtime_ns),
                        metadata={"source": path.name},
                    )

        return DiscoveryResult(resources(), complete=True)

    async def snapshot(self, resource: Resource, /) -> ResourceSnapshot:
        """Read one document."""
        path = self._workspace / f"{resource.resource_id}.md"
        content = await asyncio.to_thread(path.read_text, encoding="utf-8")
        return ResourceSnapshot(
            resource.resource_id,
            content=content,
            metadata={"source": path.name},
            version=resource.version,
        )


class IndexOperation:
    """Indexes refreshed documents and removes deleted ones."""

    def __init__(self, index: EvidenceStore) -> None:
        self._index = index

    async def __call__(
        self,
        resource: Resource,
        snapshot: ResourceSnapshot | None,
        action: PlanAction,
        request: RefreshRequest,
        /,
    ) -> None:
        """Apply one planned refresh action to the index."""
        del request
        if action is PlanAction.DELETE or snapshot is None:
            await asyncio.to_thread(self._index.delete, resource.resource_id)
            return
        content = str(snapshot.content)
        title = content.splitlines()[0].lstrip("# ").strip() if content else ""
        await asyncio.to_thread(
            self._index.upsert,
            resource.resource_id,
            title or resource.resource_id,
            str(snapshot.metadata.get("source", resource.resource_id)),
            content,
        )
