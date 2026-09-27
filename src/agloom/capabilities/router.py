"""Capability routing with policy, retrieval, and reranking stages."""

from __future__ import annotations

import re
import threading
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from math import sqrt

from langchain_core.documents import Document
from langchain_core.documents.compressor import BaseDocumentCompressor
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever

from agloom.capabilities.registry import (
    CapabilityDescriptor,
    CapabilityKind,
    CapabilityRegistry,
)
from agloom.errors import CapabilityResolutionError


class CapabilityPolicy(ABC):
    @abstractmethod
    def allows(self, task: str, capability: CapabilityDescriptor) -> bool:
        """Return whether the capability may be considered for this task."""


class AllowAllCapabilities(CapabilityPolicy):
    def allows(self, task: str, capability: CapabilityDescriptor) -> bool:
        return True


@dataclass(frozen=True)
class CapabilitySelection:
    task: str
    capabilities: tuple[CapabilityDescriptor, ...]

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(capability.name for capability in self.capabilities)

    @property
    def tools(self) -> tuple[object, ...]:
        return tuple(
            capability.value
            for capability in self.capabilities
            if capability.kind is CapabilityKind.TOOL
        )


class CapabilityRouter(ABC):
    @abstractmethod
    def route(
        self,
        task: str,
        catalog: CapabilityRegistry,
    ) -> CapabilitySelection:
        """Select capabilities for one task."""

    @abstractmethod
    async def aroute(
        self,
        task: str,
        catalog: CapabilityRegistry,
    ) -> CapabilitySelection:
        """Asynchronously select capabilities for one task."""


class DefaultCapabilityRouter(CapabilityRouter):
    def __init__(
        self,
        *,
        policy: CapabilityPolicy | None = None,
        embeddings: Embeddings | None = None,
        retriever: BaseRetriever | None = None,
        reranker: BaseDocumentCompressor | None = None,
        max_results: int = 20,
    ) -> None:
        if max_results < 1:
            raise ValueError("max_results must be at least 1")
        self._policy = policy or AllowAllCapabilities()
        self._embeddings = embeddings
        self._retriever = retriever
        self._reranker = reranker
        self._max_results = max_results
        self._embedding_lock = threading.Lock()
        self._embedding_cache: dict[
            tuple[tuple[str, str], ...],
            list[list[float]],
        ] = {}

    def route(
        self,
        task: str,
        catalog: CapabilityRegistry,
    ) -> CapabilitySelection:
        allowed = self._allowed(task, catalog)
        candidates = self._initial_candidates(task, allowed)
        if self._embeddings is not None:
            candidates = _merge(
                candidates,
                self._embedding_candidates(task, allowed),
            )
        if self._retriever is not None:
            candidates = _merge(
                candidates,
                self._from_documents(
                    self._retriever.invoke(task),
                    allowed,
                ),
            )
        candidates = self._rerank(task, candidates)
        return CapabilitySelection(task, candidates[: self._max_results])

    async def aroute(
        self,
        task: str,
        catalog: CapabilityRegistry,
    ) -> CapabilitySelection:
        allowed = self._allowed(task, catalog)
        candidates = self._initial_candidates(task, allowed)
        if self._embeddings is not None:
            candidates = _merge(
                candidates,
                await self._aembedding_candidates(task, allowed),
            )
        if self._retriever is not None:
            candidates = _merge(
                candidates,
                self._from_documents(
                    await self._retriever.ainvoke(task),
                    allowed,
                ),
            )
        if self._reranker is not None and candidates:
            documents = await self._reranker.acompress_documents(
                [_as_document(capability) for capability in candidates],
                task,
            )
            candidates = self._from_documents(documents, allowed)
        return CapabilitySelection(task, candidates[: self._max_results])

    def _allowed(
        self,
        task: str,
        catalog: CapabilityRegistry,
    ) -> dict[str, CapabilityDescriptor]:
        return {
            capability.name: capability
            for capability in catalog.descriptors()
            if not capability.metadata.get("internal", False)
            and self._policy.allows(task, capability)
        }

    def _initial_candidates(
        self,
        task: str,
        allowed: dict[str, CapabilityDescriptor],
    ) -> tuple[CapabilityDescriptor, ...]:
        if self._retriever is None and self._embeddings is None:
            return tuple(allowed.values())
        normalized = task.casefold()
        exact = []
        for capability in allowed.values():
            raw_aliases = capability.metadata.get("aliases", ())
            if isinstance(raw_aliases, str):
                aliases = (raw_aliases,)
            elif isinstance(raw_aliases, Sequence):
                aliases = raw_aliases
            else:
                aliases = ()
            names = [capability.name, *aliases]
            if any(_contains_name(normalized, str(name)) for name in names):
                exact.append(capability)
        return tuple(exact)

    def _embedding_candidates(
        self,
        task: str,
        allowed: dict[str, CapabilityDescriptor],
    ) -> tuple[CapabilityDescriptor, ...]:
        if self._embeddings is None or not allowed:
            return ()
        capabilities = tuple(allowed.values())
        key = _embedding_key(capabilities)
        with self._embedding_lock:
            vectors = self._embedding_cache.get(key)
        if vectors is None:
            vectors = self._embeddings.embed_documents(
                [_embedding_text(capability) for capability in capabilities]
            )
            with self._embedding_lock:
                self._embedding_cache[key] = vectors
        query = self._embeddings.embed_query(task)
        return _rank_embeddings(capabilities, vectors, query)

    async def _aembedding_candidates(
        self,
        task: str,
        allowed: dict[str, CapabilityDescriptor],
    ) -> tuple[CapabilityDescriptor, ...]:
        if self._embeddings is None or not allowed:
            return ()
        capabilities = tuple(allowed.values())
        key = _embedding_key(capabilities)
        with self._embedding_lock:
            vectors = self._embedding_cache.get(key)
        if vectors is None:
            vectors = await self._embeddings.aembed_documents(
                [_embedding_text(capability) for capability in capabilities]
            )
            with self._embedding_lock:
                self._embedding_cache[key] = vectors
        query = await self._embeddings.aembed_query(task)
        return _rank_embeddings(capabilities, vectors, query)

    def _rerank(
        self,
        task: str,
        candidates: tuple[CapabilityDescriptor, ...],
    ) -> tuple[CapabilityDescriptor, ...]:
        if self._reranker is None or not candidates:
            return candidates
        documents = self._reranker.compress_documents(
            [_as_document(capability) for capability in candidates],
            task,
        )
        allowed = {capability.name: capability for capability in candidates}
        return self._from_documents(documents, allowed)

    @staticmethod
    def _from_documents(
        documents: Sequence[Document],
        allowed: dict[str, CapabilityDescriptor],
    ) -> tuple[CapabilityDescriptor, ...]:
        capabilities = []
        for document in documents:
            name = document.metadata.get("capability_name")
            if not isinstance(name, str):
                raise CapabilityResolutionError(
                    "capability retriever and reranker documents must include "
                    "string metadata 'capability_name'"
                )
            capability = allowed.get(name)
            if capability is not None and capability not in capabilities:
                capabilities.append(capability)
        return tuple(capabilities)


def _as_document(capability: CapabilityDescriptor) -> Document:
    return Document(
        page_content=f"{capability.name}\n{capability.description}",
        metadata={
            **capability.metadata,
            "capability_name": capability.name,
            "kind": capability.kind.value,
            "provider": capability.provider,
        },
    )


def _contains_name(task: str, name: str) -> bool:
    normalized = name.casefold().replace("_", " ").replace(":", " ")
    words = [re.escape(word) for word in normalized.split() if word]
    return (
        bool(words) and re.search(r"\b" + r"\s+".join(words) + r"\b", task) is not None
    )


def _merge(
    first: tuple[CapabilityDescriptor, ...],
    second: tuple[CapabilityDescriptor, ...],
) -> tuple[CapabilityDescriptor, ...]:
    merged = list(first)
    names = {capability.name for capability in merged}
    for capability in second:
        if capability.name not in names:
            merged.append(capability)
            names.add(capability.name)
    return tuple(merged)


def _embedding_key(
    capabilities: tuple[CapabilityDescriptor, ...],
) -> tuple[tuple[str, str], ...]:
    return tuple(
        (capability.name, capability.description) for capability in capabilities
    )


def _embedding_text(capability: CapabilityDescriptor) -> str:
    return f"{capability.name}\n{capability.description}"


def _rank_embeddings(
    capabilities: tuple[CapabilityDescriptor, ...],
    vectors: Sequence[Sequence[float]],
    query: Sequence[float],
) -> tuple[CapabilityDescriptor, ...]:
    if len(vectors) != len(capabilities):
        raise CapabilityResolutionError(
            "capability embedder returned an unexpected vector count"
        )
    scored = [
        (_cosine_similarity(vector, query), index, capability)
        for index, (capability, vector) in enumerate(
            zip(capabilities, vectors, strict=True)
        )
    ]
    scored.sort(key=lambda item: (-item[0], item[1]))
    return tuple(capability for _, _, capability in scored)


def _cosine_similarity(
    left: Sequence[float],
    right: Sequence[float],
) -> float:
    if len(left) != len(right):
        raise CapabilityResolutionError("capability embedding dimensions do not match")
    left_norm = sqrt(sum(value * value for value in left))
    right_norm = sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return sum(a * b for a, b in zip(left, right, strict=True)) / left_norm / right_norm
