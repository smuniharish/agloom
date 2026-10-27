"""Approved web policies: a pgvector index kept in sync with `data/policies.json`.

`RefreshEngine` discovers the policies in the file and (re)indexes only the
ones that changed, so edits and removals reach the index without reseeding.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from typing import Any, Protocol

from langchain_core.embeddings import Embeddings
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

from .governance import POLICY_SEARCH
from .models import Evidence

EMBEDDING_DIMENSIONS = 384


class PolicySearch(Protocol):
    """Similarity search over the approved policies."""

    def search(
        self, query: str, policy_ids: Sequence[str] = (), limit: int = 8
    ) -> list[Evidence]:
        """Return the policies most similar to `query`, best first."""
        ...


class PolicyCatalog(PolicySearch, Protocol):
    """A policy index that the refresh operation maintains."""

    def upsert(self, policy_id: str, title: str, body: str) -> None:
        """Index or re-index one policy."""
        ...

    def delete(self, policy_id: str) -> None:
        """Remove one policy from the index."""
        ...


class PolicyIndex:
    """Approved policies and their embeddings in PostgreSQL with pgvector."""

    def __init__(self, pool: ConnectionPool, embeddings: Embeddings) -> None:
        self._pool = pool
        self._embeddings = embeddings

    def initialize(self) -> None:
        """Create the policy table."""
        with self._pool.connection() as connection:
            connection.execute(f"""
                CREATE TABLE IF NOT EXISTS compliance_policies (
                    policy_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    body TEXT NOT NULL,
                    embedding VECTOR({EMBEDDING_DIMENSIONS}) NOT NULL,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """)

    def upsert(self, policy_id: str, title: str, body: str) -> None:
        """Embed and store one policy, replacing an earlier version."""
        embedding = Vector(self._embeddings.embed_documents([body])[0])
        with self._pool.connection() as connection:
            connection.execute(
                """
                INSERT INTO compliance_policies (policy_id, title, body, embedding)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (policy_id) DO UPDATE SET
                    title = EXCLUDED.title,
                    body = EXCLUDED.body,
                    embedding = EXCLUDED.embedding,
                    updated_at = now()
                """,
                (policy_id, title, body, embedding),
            )

    def delete(self, policy_id: str) -> None:
        """Remove one policy."""
        with self._pool.connection() as connection:
            connection.execute(
                "DELETE FROM compliance_policies WHERE policy_id = %s", (policy_id,)
            )

    def search(
        self, query: str, policy_ids: Sequence[str] = (), limit: int = 8
    ) -> list[Evidence]:
        """Return the closest policies, optionally limited to `policy_ids`."""
        embedding = Vector(self._embeddings.embed_query(query))
        where = "WHERE policy_id = ANY(%s)" if policy_ids else ""
        filters = (list(policy_ids),) if policy_ids else ()
        with self._pool.connection() as connection:
            rows = connection.execute(
                f"""
                SELECT policy_id, title, body, 1 - (embedding <=> %s) AS similarity
                FROM compliance_policies {where}
                ORDER BY embedding <=> %s
                LIMIT %s
                """,
                (embedding, *filters, embedding, limit),
            ).fetchall()
        return [
            Evidence(
                policy_id=row[0],
                title=row[1],
                excerpt=row[2][:700],
                similarity=max(0.0, min(1.0, float(row[3]))),
            )
            for row in rows
        ]


def policy_search_tool(index: PolicySearch) -> BaseTool:
    """Return the `search_policies` tool backed by `index`."""

    @tool(POLICY_SEARCH)
    def search_policies(query: str) -> str:
        """Search the approved web policies for requirements relevant to a page."""
        return json.dumps([item.model_dump() for item in index.search(query)])

    return search_policies


class PolicyFileSource:
    """Discovers the policies listed in a JSON file."""

    def __init__(self, path: Path) -> None:
        self._path = path

    async def discover(self) -> DiscoveryResult:
        """List every policy with a version derived from its content."""
        policies = await asyncio.to_thread(self._load)

        async def resources() -> AsyncIterator[Resource]:
            for policy_id, policy in policies.items():
                yield Resource(
                    policy_id,
                    version=_digest(policy),
                    metadata={"title": policy["title"]},
                )

        return DiscoveryResult(resources(), complete=True)

    async def snapshot(self, resource: Resource, /) -> ResourceSnapshot:
        """Read one policy."""
        policy = (await asyncio.to_thread(self._load))[resource.resource_id]
        return ResourceSnapshot(
            resource.resource_id,
            content=policy["body"],
            metadata={"title": policy["title"]},
            version=_digest(policy),
        )

    def _load(self) -> dict[str, dict[str, str]]:
        entries = json.loads(self._path.read_text(encoding="utf-8"))
        policies: dict[str, dict[str, str]] = {}
        for entry in entries:
            policy = {key: str(entry[key]).strip() for key in ("id", "title", "body")}
            if not all(policy.values()):
                raise ValueError("every policy needs a non-empty id, title, and body")
            if policy["id"] in policies:
                raise ValueError(f"duplicate policy id {policy['id']!r}")
            policies[policy["id"]] = policy
        return policies


class IndexOperation:
    """Indexes refreshed policies and removes deleted ones."""

    def __init__(self, index: PolicyCatalog) -> None:
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
        await asyncio.to_thread(
            self._index.upsert,
            resource.resource_id,
            str(snapshot.metadata.get("title", resource.resource_id)),
            str(snapshot.content),
        )


def _digest(policy: dict[str, Any]) -> str:
    canonical = json.dumps(policy, sort_keys=True).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()[:16]
