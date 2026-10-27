"""Capability routing: filtering, lexical matching, semantic retrieval, reranking."""

from __future__ import annotations

import re
import threading
from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_core.documents.compressor import BaseDocumentCompressor
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import InMemoryVectorStore

from agloom.capabilities.registry import CapabilityDescriptor
from agloom.errors import CapabilityResolutionError, ConfigurationError

_INDEX_CACHE_SIZE = 8


class CapabilityFilter(ABC):
    """Decides which capabilities may be considered for one task.

    Filters run before retrieval and ranking, so a capability they reject is
    never shown to the analyzer or the model. Filters control visibility;
    execution is governed separately by a [`PolicyManager`][agloom.PolicyManager].
    """

    @abstractmethod
    def allows(self, task: str, capability: CapabilityDescriptor) -> bool:
        """Return whether the capability may be considered for the task."""


class AllowAllCapabilities(CapabilityFilter):
    """A filter that admits every capability."""

    def allows(self, task: str, capability: CapabilityDescriptor) -> bool:
        """Admit the capability."""
        return True


@dataclass(frozen=True)
class CapabilitySelection:
    """The capabilities selected for one task, best first."""

    task: str
    capabilities: tuple[CapabilityDescriptor, ...]

    @property
    def names(self) -> tuple[str, ...]:
        """The selected capability names."""
        return tuple(capability.name for capability in self.capabilities)

    @property
    def tool_names(self) -> frozenset[str]:
        """Names of the selected capabilities the model can call."""
        return frozenset(
            capability.name
            for capability in self.capabilities
            if capability.model_callable
        )


class CapabilityRouter(ABC):
    """Selects the capabilities relevant to one task."""

    @abstractmethod
    def route(
        self, task: str, catalog: Sequence[CapabilityDescriptor]
    ) -> CapabilitySelection:
        """Select capabilities for the task."""

    @abstractmethod
    async def aroute(
        self, task: str, catalog: Sequence[CapabilityDescriptor]
    ) -> CapabilitySelection:
        """Asynchronously select capabilities for the task."""


class DefaultCapabilityRouter(CapabilityRouter):
    """Filter, then optionally rank capabilities with LangChain retrieval components.

    1. The filter removes capabilities that may not be considered.
    2. Without `embeddings`, `retriever`, or `reranker`, every remaining
       capability is selected.
    3. Otherwise, capabilities named in the task (by name or `aliases`
       metadata) come first, followed by semantic matches from an
       `InMemoryVectorStore` built with `embeddings` and from `retriever`
       documents whose metadata contains `capability_name`.
    4. The optional `reranker` reorders the candidates; at most `max_results`
       are selected.

    Args:
        capability_filter: Visibility filter; admits everything by default.
        embeddings: Embeddings used to rank capability descriptions.
        retriever: Retriever returning documents with `capability_name` metadata.
        reranker: Document compressor that reorders candidates.
        max_results: Maximum number of capabilities selected by ranking.
    """

    def __init__(
        self,
        *,
        capability_filter: CapabilityFilter | None = None,
        embeddings: Embeddings | None = None,
        retriever: BaseRetriever | None = None,
        reranker: BaseDocumentCompressor | None = None,
        max_results: int = 20,
    ) -> None:
        if max_results < 1:
            raise ConfigurationError("max_results must be at least 1")
        self._filter = capability_filter or AllowAllCapabilities()
        self._embeddings = embeddings
        self._retriever = retriever
        self._reranker = reranker
        self._max_results = max_results
        self._lock = threading.Lock()
        self._indexes: OrderedDict[tuple[tuple[str, str], ...], InMemoryVectorStore] = (
            OrderedDict()
        )

    @property
    def ranks(self) -> bool:
        """Whether a retrieval or reranking stage is configured."""
        return any(
            component is not None
            for component in (self._embeddings, self._retriever, self._reranker)
        )

    def route(
        self, task: str, catalog: Sequence[CapabilityDescriptor]
    ) -> CapabilitySelection:
        """Select capabilities for the task."""
        allowed = self._allowed(task, catalog)
        if not self.ranks:
            return CapabilitySelection(task, tuple(allowed.values()))
        candidates = _exact_matches(task, allowed)
        if self._embeddings is not None and allowed:
            key = _index_key(allowed)
            store = self._cached_index(key)
            if store is None:
                store = InMemoryVectorStore(self._embeddings)
                store.add_documents(_documents(allowed), ids=list(allowed))
                store = self._cache_index(key, store)
            documents = store.similarity_search(task, k=len(allowed))
            candidates = _merge(candidates, _from_documents(documents, allowed))
        if self._retriever is not None:
            documents = self._retriever.invoke(task)
            candidates = _merge(candidates, _from_documents(documents, allowed))
        if self._reranker is not None and candidates:
            documents = self._reranker.compress_documents(
                [_document(capability) for capability in candidates], task
            )
            candidates = _from_documents(documents, allowed)
        return CapabilitySelection(task, candidates[: self._max_results])

    async def aroute(
        self, task: str, catalog: Sequence[CapabilityDescriptor]
    ) -> CapabilitySelection:
        """Asynchronously select capabilities for the task."""
        allowed = self._allowed(task, catalog)
        if not self.ranks:
            return CapabilitySelection(task, tuple(allowed.values()))
        candidates = _exact_matches(task, allowed)
        if self._embeddings is not None and allowed:
            key = _index_key(allowed)
            store = self._cached_index(key)
            if store is None:
                store = InMemoryVectorStore(self._embeddings)
                await store.aadd_documents(_documents(allowed), ids=list(allowed))
                store = self._cache_index(key, store)
            documents = await store.asimilarity_search(task, k=len(allowed))
            candidates = _merge(candidates, _from_documents(documents, allowed))
        if self._retriever is not None:
            documents = await self._retriever.ainvoke(task)
            candidates = _merge(candidates, _from_documents(documents, allowed))
        if self._reranker is not None and candidates:
            documents = await self._reranker.acompress_documents(
                [_document(capability) for capability in candidates], task
            )
            candidates = _from_documents(documents, allowed)
        return CapabilitySelection(task, candidates[: self._max_results])

    def _allowed(
        self, task: str, catalog: Sequence[CapabilityDescriptor]
    ) -> dict[str, CapabilityDescriptor]:
        return {
            capability.name: capability
            for capability in catalog
            if self._filter.allows(task, capability)
        }

    def _cached_index(
        self, key: tuple[tuple[str, str], ...]
    ) -> InMemoryVectorStore | None:
        with self._lock:
            store = self._indexes.get(key)
            if store is not None:
                self._indexes.move_to_end(key)
            return store

    def _cache_index(
        self, key: tuple[tuple[str, str], ...], store: InMemoryVectorStore
    ) -> InMemoryVectorStore:
        with self._lock:
            cached = self._indexes.setdefault(key, store)
            self._indexes.move_to_end(key)
            while len(self._indexes) > _INDEX_CACHE_SIZE:
                self._indexes.popitem(last=False)
            return cached


def _index_key(allowed: dict[str, CapabilityDescriptor]) -> tuple[tuple[str, str], ...]:
    return tuple((name, capability.description) for name, capability in allowed.items())


def _documents(allowed: dict[str, CapabilityDescriptor]) -> list[Document]:
    return [_document(capability) for capability in allowed.values()]


def _document(capability: CapabilityDescriptor) -> Document:
    return Document(
        page_content=f"{capability.name}\n{capability.description}",
        metadata={
            "capability_name": capability.name,
            "kind": capability.kind.value,
            "provider": capability.provider,
        },
    )


def _exact_matches(
    task: str, allowed: dict[str, CapabilityDescriptor]
) -> tuple[CapabilityDescriptor, ...]:
    normalized = task.casefold()
    matches = []
    for capability in allowed.values():
        aliases = capability.metadata.get("aliases", ())
        names: list[object] = [
            capability.name,
            *((aliases,) if isinstance(aliases, str) else aliases),
        ]
        if any(_mentions(normalized, str(name)) for name in names):
            matches.append(capability)
    return tuple(matches)


def _mentions(task: str, name: str) -> bool:
    words = [
        re.escape(word) for word in re.split(r"[\s_:\-]+", name.casefold()) if word
    ]
    return (
        bool(words) and re.search(r"\b" + r"\s+".join(words) + r"\b", task) is not None
    )


def _merge(
    first: tuple[CapabilityDescriptor, ...], second: tuple[CapabilityDescriptor, ...]
) -> tuple[CapabilityDescriptor, ...]:
    merged = dict.fromkeys(first)
    merged.update(dict.fromkeys(second))
    return tuple(merged)


def _from_documents(
    documents: Sequence[Document], allowed: dict[str, CapabilityDescriptor]
) -> tuple[CapabilityDescriptor, ...]:
    selected: dict[str, CapabilityDescriptor] = {}
    for document in documents:
        name = document.metadata.get("capability_name")
        if not isinstance(name, str):
            raise CapabilityResolutionError(
                "capability retriever and reranker documents must include "
                "string metadata 'capability_name'"
            )
        capability = allowed.get(name)
        if capability is not None:
            selected.setdefault(name, capability)
    return tuple(selected.values())
