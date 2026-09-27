from __future__ import annotations

from collections.abc import Sequence

import pytest
from langchain_core.documents import Document
from langchain_core.documents.compressor import BaseDocumentCompressor
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.retrievers import BaseRetriever
from langchain_core.tools import tool
from mcp_capability_router import Prompt, Resource, Tool

from agloom import (
    CapabilityDescriptor,
    CapabilityKind,
    CapabilityPolicy,
    CapabilityRegistry,
    ConfigurationError,
    DefaultCapabilityRouter,
    InMemoryCapabilityRegistry,
    create_agent,
)
from agloom.capabilities.providers import (
    ApplicationCapabilityProvider,
    LocalLangChainToolProvider,
    MCPCapabilityProvider,
)
from agloom.errors import CapabilityResolutionError


class StaticRetriever(BaseRetriever):
    documents: list[Document]

    def _get_relevant_documents(self, query: str, *, run_manager):
        return self.documents


class ReverseReranker(BaseDocumentCompressor):
    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks=None,
    ) -> Sequence[Document]:
        return tuple(reversed(documents))


class NoNetworkPolicy(CapabilityPolicy):
    def allows(self, task: str, capability: CapabilityDescriptor) -> bool:
        return not capability.metadata.get("network", False)


class KeywordEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    @staticmethod
    def _embed(text: str) -> list[float]:
        normalized = text.casefold()
        return [
            float("search" in normalized or "google" in normalized),
            float("document" in normalized),
            float("calculate" in normalized),
        ]


def descriptor(
    name: str,
    *,
    kind: CapabilityKind = CapabilityKind.APPLICATION,
    metadata: dict | None = None,
) -> CapabilityDescriptor:
    return CapabilityDescriptor(
        name=name,
        description=f"{name} capability",
        kind=kind,
        value=object(),
        metadata=metadata or {},
    )


def test_capability_registry_is_abstract() -> None:
    with pytest.raises(TypeError):
        CapabilityRegistry()


def test_in_memory_registry_resolves_descriptors_and_values() -> None:
    value = object()
    capability = CapabilityDescriptor(
        name="documents",
        description="Retrieve documents.",
        kind=CapabilityKind.APPLICATION,
        value=value,
    )
    registry = InMemoryCapabilityRegistry([capability])

    assert registry.descriptor("documents") is capability
    assert registry.resolve("documents") is value
    assert registry.names == ("documents",)
    with pytest.raises(CapabilityResolutionError, match="not registered"):
        registry.resolve("missing")


def test_local_and_application_providers_populate_catalog() -> None:
    @tool
    def google_search(query: str) -> str:
        """Search Google."""
        return query

    application = object()
    registry = InMemoryCapabilityRegistry(
        [
            *LocalLangChainToolProvider([google_search]).load(),
            *ApplicationCapabilityProvider({"document_store": application}).load(),
        ]
    )

    assert registry.descriptor("google_search").kind is CapabilityKind.TOOL
    assert registry.resolve("document_store") is application


def test_router_applies_policy_before_selection() -> None:
    registry = InMemoryCapabilityRegistry(
        [
            descriptor("local_documents"),
            descriptor("web_search", metadata={"network": True}),
        ]
    )
    router = DefaultCapabilityRouter(policy=NoNetworkPolicy())

    selection = router.route("Find documents", registry)

    assert selection.names == ("local_documents",)


def test_router_combines_exact_retrieval_and_reranking() -> None:
    registry = InMemoryCapabilityRegistry(
        [
            descriptor("calculator"),
            descriptor("google_search", kind=CapabilityKind.TOOL),
            descriptor("document_retriever"),
        ]
    )
    retriever = StaticRetriever(
        documents=[
            Document(
                page_content="documents",
                metadata={"capability_name": "document_retriever"},
            )
        ]
    )
    router = DefaultCapabilityRouter(
        retriever=retriever,
        reranker=ReverseReranker(),
        max_results=2,
    )

    selection = router.route(
        "Use google search and retrieve supporting documents",
        registry,
    )

    assert selection.names == ("document_retriever", "google_search")
    assert len(selection.tools) == 1


def test_router_accepts_custom_langchain_embeddings() -> None:
    registry = InMemoryCapabilityRegistry(
        [
            descriptor("calculator"),
            descriptor("google_search", kind=CapabilityKind.TOOL),
            descriptor("document_retriever"),
        ]
    )
    router = DefaultCapabilityRouter(
        embeddings=KeywordEmbeddings(),
        max_results=1,
    )

    selection = router.route("Search Google for current news", registry)

    assert selection.names == ("google_search",)


@pytest.mark.asyncio
async def test_router_supports_async_retrieval_and_reranking() -> None:
    registry = InMemoryCapabilityRegistry([descriptor("document_retriever")])
    router = DefaultCapabilityRouter(
        retriever=StaticRetriever(
            documents=[
                Document(
                    page_content="documents",
                    metadata={"capability_name": "document_retriever"},
                )
            ]
        ),
        reranker=ReverseReranker(),
    )

    selection = await router.aroute("Retrieve documents", registry)

    assert selection.names == ("document_retriever",)


def test_retriever_documents_require_capability_name_metadata() -> None:
    registry = InMemoryCapabilityRegistry([descriptor("documents")])
    router = DefaultCapabilityRouter(
        retriever=StaticRetriever(documents=[Document(page_content="documents")])
    )

    with pytest.raises(CapabilityResolutionError, match="capability_name"):
        router.route("Find documents", registry)


@pytest.mark.asyncio
async def test_mcp_provider_discovers_and_executes_remote_tools() -> None:
    capabilities = [
        Tool(
            capability_id="documents:tool:remote_search",
            server_id="documents",
            name="remote_search",
            description="Search a remote index.",
        ),
        Resource(
            capability_id="documents:resource:docs://guide",
            server_id="documents",
            name="guide",
            uri="docs://guide",
            description="Read the guide.",
        ),
        Prompt(
            capability_id="documents:prompt:summarize",
            server_id="documents",
            name="summarize",
            description="Render a summary prompt.",
        ),
    ]

    class Registry:
        async def list(self, *, server_id=None, type=None):
            return [
                capability
                for capability in capabilities
                if server_id is None or capability.server_id == server_id
            ]

    class Runtime:
        registry = Registry()

        async def execute(self, capability_id, arguments):
            return {"capability_id": capability_id, "arguments": arguments}

        async def read_resource(self, capability_id):
            return f"resource:{capability_id}"

        async def get_prompt(self, capability_id, arguments):
            return {"prompt": capability_id, "arguments": arguments}

    provider = MCPCapabilityProvider(
        {"mcp_runtime": Runtime()},
        server_name="documents",
    )

    discovered = await provider.adiscover()
    tool_result = await discovered[0].value.ainvoke({"query": "Agloom"})
    resource_result = await discovered[1].value.ainvoke()
    prompt_result = await discovered[2].value.ainvoke({"topic": "Agloom"})

    assert [capability.name for capability in discovered] == [
        "mcp:documents:tool:remote_search",
        "mcp:documents:resource:docs://guide",
        "mcp:documents:prompt:summarize",
    ]
    assert all(capability.kind is CapabilityKind.MCP for capability in discovered)
    assert tool_result == {
        "capability_id": "documents:tool:remote_search",
        "arguments": {"query": "Agloom"},
    }
    assert resource_result == "resource:documents:resource:docs://guide"
    assert prompt_result == {
        "prompt": "documents:prompt:summarize",
        "arguments": {"topic": "Agloom"},
    }


@pytest.mark.asyncio
async def test_agent_routes_and_executes_application_capabilities() -> None:
    registry = InMemoryCapabilityRegistry()
    agent = create_agent(
        model=FakeListChatModel(responses=["unused"]),
        tools=[],
        pattern="react",
        capabilities={"double": lambda value: value * 2},
        capability_registry=registry,
    )

    selection = agent.route_capabilities("Use double")
    sync_result = agent.execute_capability("double", 4)
    async_result = await agent.aexecute_capability("double", 5)

    assert agent.capabilities is registry
    assert selection.names == ("double",)
    assert "xai_runtime" in agent.capabilities.names
    assert sync_result == 8
    assert async_result == 10


def test_complete_custom_router_rejects_component_options() -> None:
    with pytest.raises(ConfigurationError, match="cannot be combined"):
        create_agent(
            model=FakeListChatModel(responses=["unused"]),
            tools=[],
            capability_router=DefaultCapabilityRouter(),
            capability_embeddings=KeywordEmbeddings(),
        )
