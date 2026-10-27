"""Capability catalog, providers, and routing."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.documents.compressor import BaseDocumentCompressor
from langchain_core.retrievers import BaseRetriever
from langchain_core.tools import BaseTool, tool

from agloom import (
    AllowAllCapabilities,
    CapabilityDescriptor,
    CapabilityFilter,
    CapabilityKind,
    CapabilityResolutionError,
    CapabilitySelection,
    ConfigurationError,
    DefaultCapabilityRouter,
    InMemoryCapabilityRegistry,
)
from agloom.capabilities.providers import (
    application_descriptors,
    mcp_tool_name,
    to_tool,
    tool_descriptors,
)
from helpers import KeywordEmbeddings


@tool
def lookup_order(order_id: str) -> str:
    """Look up an order by id."""
    return order_id


def describe(name: str, description: str = "", **metadata: Any) -> CapabilityDescriptor:
    return CapabilityDescriptor(
        name,
        description or f"{name} capability",
        CapabilityKind.APPLICATION,
        value=name,
        metadata=metadata,
    )


CATALOG = (
    describe("weather", "Current weather forecast and temperature."),
    describe("billing", "Invoices, refunds, and payments.", aliases=("invoice desk",)),
    describe("shipping", "Parcel tracking and delivery status.", aliases="parcels"),
)


class DenyBilling(CapabilityFilter):
    def allows(self, task: str, capability: CapabilityDescriptor) -> bool:
        return capability.name != "billing"


class NamedRetriever(BaseRetriever):
    names: list[str]

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        return [
            Document(page_content=name, metadata={"capability_name": name})
            for name in self.names
        ]


class ReverseReranker(BaseDocumentCompressor):
    def compress_documents(
        self, documents: Sequence[Document], query: str, callbacks: Any = None
    ) -> Sequence[Document]:
        return list(reversed(documents))


class BrokenRetriever(BaseRetriever):
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        return [Document(page_content="x", metadata={})]


class TestDescriptorsAndRegistry:
    def test_descriptor_normalizes_and_freezes(self) -> None:
        descriptor = CapabilityDescriptor(
            " name ", " text ", CapabilityKind.TOOL, None, " provider ", {"a": 1}
        )
        assert (descriptor.name, descriptor.description, descriptor.provider) == (
            "name",
            "text",
            "provider",
        )
        with pytest.raises(TypeError):
            descriptor.metadata["a"] = 2  # type: ignore[index]

    @pytest.mark.parametrize(
        ("name", "provider", "message"),
        [
            (" ", "p", "name must not be blank"),
            ("n", " ", "provider must not be blank"),
        ],
    )
    def test_descriptor_rejects_blank_fields(
        self, name: str, provider: str, message: str
    ) -> None:
        with pytest.raises(ConfigurationError, match=message):
            CapabilityDescriptor(name, "", CapabilityKind.TOOL, None, provider)

    def test_model_callable_kinds(self) -> None:
        assert CapabilityDescriptor("t", "", CapabilityKind.TOOL, None).model_callable
        assert not describe("app").model_callable
        mcp_tool = CapabilityDescriptor(
            "m", "", CapabilityKind.MCP, None, metadata={"mcp_type": "tool"}
        )
        mcp_resource = CapabilityDescriptor(
            "r", "", CapabilityKind.MCP, None, metadata={"mcp_type": "resource"}
        )
        assert mcp_tool.model_callable
        assert not mcp_resource.model_callable

    def test_registry_registration_and_lookup(self) -> None:
        registry = InMemoryCapabilityRegistry(CATALOG)
        assert registry.names == ("weather", "billing", "shipping")
        assert registry.resolve("billing") == "billing"
        assert registry.descriptor("weather").description.startswith("Current")
        with pytest.raises(ConfigurationError, match="already registered"):
            registry.register(describe("weather"))
        with pytest.raises(CapabilityResolutionError, match="not registered"):
            registry.descriptor("missing")


class TestProviders:
    def test_to_tool(self) -> None:
        assert to_tool(lookup_order) is lookup_order

        def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        converted = to_tool(add)
        assert isinstance(converted, BaseTool)
        assert converted.name == "add"
        with pytest.raises(ConfigurationError, match="LangChain tools or functions"):
            to_tool(42)  # type: ignore[arg-type]

    def test_tool_and_application_descriptors(self) -> None:
        (descriptor,) = tool_descriptors([lookup_order])
        assert descriptor.kind is CapabilityKind.TOOL
        assert descriptor.provider == "langchain"
        assert descriptor.value is lookup_order

        class Service:
            description = "Sends notifications."

        (app,) = application_descriptors({"notifier": Service()})
        assert app.description == "Sends notifications."
        assert app.metadata["type"] == "Service"
        (bare,) = application_descriptors({"plain": object()})
        assert bare.description == ""

    def test_mcp_tool_name_examples(self) -> None:
        assert mcp_tool_name("ops", "service_status") == "ops__service_status"
        assert mcp_tool_name("my server", "do/thing") == "my_server__do_thing"
        assert mcp_tool_name("", "") == "mcp_tool"
        long_name = mcp_tool_name("server", "x" * 100)
        assert len(long_name) == 64
        assert long_name == mcp_tool_name("server", "x" * 100)
        assert long_name != mcp_tool_name("server", "x" * 101)


@given(st.text(max_size=80), st.text(max_size=120))
def test_mcp_tool_names_are_function_calling_safe(server: str, name: str) -> None:
    result = mcp_tool_name(server, name)
    assert 1 <= len(result) <= 64
    assert re.fullmatch(r"[A-Za-z0-9_-]+", result)
    assert result == mcp_tool_name(server, name)


class TestRouting:
    def test_without_ranking_every_allowed_capability_is_selected(self) -> None:
        router = DefaultCapabilityRouter(capability_filter=DenyBilling())
        assert not router.ranks
        selection = router.route("anything", CATALOG)
        assert selection.names == ("weather", "shipping")
        assert selection.tool_names == frozenset()

    async def test_async_without_ranking(self) -> None:
        selection = await DefaultCapabilityRouter().aroute("x", CATALOG)
        assert selection.names == ("weather", "billing", "shipping")

    def test_max_results_must_be_positive(self) -> None:
        with pytest.raises(ConfigurationError, match="at least 1"):
            DefaultCapabilityRouter(max_results=0)

    def test_exact_names_and_aliases_rank_first(self) -> None:
        router = DefaultCapabilityRouter(
            embeddings=KeywordEmbeddings(["weather", "invoice", "parcel"]),
            max_results=2,
        )
        assert router.ranks
        assert router.route("Where are my parcels?", CATALOG).names[0] == "shipping"
        assert router.route("Ask the invoice desk", CATALOG).names[0] == "billing"
        assert router.route("Shipping status", CATALOG).names[0] == "shipping"

    def test_semantic_ranking_and_truncation(self) -> None:
        router = DefaultCapabilityRouter(
            embeddings=KeywordEmbeddings(["temperature", "refunds", "delivery"]),
            max_results=1,
        )
        assert router.route("What temperature is it?", CATALOG).names == ("weather",)
        assert router.route("I need refunds", CATALOG).names == ("billing",)

    async def test_async_semantic_ranking(self) -> None:
        router = DefaultCapabilityRouter(
            embeddings=KeywordEmbeddings(["temperature", "refunds", "delivery"]),
            max_results=1,
        )
        selection = await router.aroute("delivery please", CATALOG)
        assert selection.names == ("shipping",)
        cached = await router.aroute("temperature now", CATALOG)
        assert cached.names == ("weather",)

    def test_index_cache_is_reused_and_bounded(self) -> None:
        embeddings = KeywordEmbeddings(["weather"])
        calls: list[int] = []
        original = embeddings.embed_documents

        def counting(texts: list[str]) -> list[list[float]]:
            calls.append(len(texts))
            return original(texts)

        embeddings.embed_documents = counting  # type: ignore[method-assign]
        router = DefaultCapabilityRouter(embeddings=embeddings)
        router.route("weather", CATALOG)
        router.route("weather again", CATALOG)
        assert calls == [3]
        for index in range(9):
            router.route("x", (describe(f"cap{index}"),))
        router.route("weather", CATALOG)
        assert calls[-1] == 3
        assert len(calls) == 11

    def test_retriever_and_reranker(self) -> None:
        router = DefaultCapabilityRouter(
            retriever=NamedRetriever(names=["billing", "unknown", "weather"]),
            reranker=ReverseReranker(),
        )
        selection = router.route("help", CATALOG)
        assert selection.names == ("weather", "billing")

    async def test_async_retriever_and_reranker(self) -> None:
        router = DefaultCapabilityRouter(
            retriever=NamedRetriever(names=["shipping", "billing"]),
            reranker=ReverseReranker(),
        )
        selection = await router.aroute("help", CATALOG)
        assert selection.names == ("billing", "shipping")

    def test_reranker_without_candidates_is_skipped(self) -> None:
        router = DefaultCapabilityRouter(reranker=ReverseReranker())
        assert router.route("nothing matches", CATALOG).names == ()

    async def test_async_reranker_without_candidates_is_skipped(self) -> None:
        router = DefaultCapabilityRouter(reranker=ReverseReranker())
        assert (await router.aroute("nothing matches", CATALOG)).names == ()

    def test_embeddings_with_an_empty_catalog(self) -> None:
        router = DefaultCapabilityRouter(embeddings=KeywordEmbeddings(["a"]))
        assert router.route("a", ()).names == ()

    def test_documents_without_capability_name_are_rejected(self) -> None:
        router = DefaultCapabilityRouter(retriever=BrokenRetriever())
        with pytest.raises(CapabilityResolutionError, match="capability_name"):
            router.route("x", CATALOG)

    def test_filters_apply_before_ranking(self) -> None:
        router = DefaultCapabilityRouter(
            capability_filter=DenyBilling(),
            retriever=NamedRetriever(names=["billing", "weather"]),
        )
        assert router.route("billing", CATALOG).names == ("weather",)

    def test_allow_all_filter(self) -> None:
        assert AllowAllCapabilities().allows("task", CATALOG[0])

    def test_selection_tool_names_only_include_callable_capabilities(self) -> None:
        tool_capability = CapabilityDescriptor("t", "", CapabilityKind.TOOL, None)
        selection = CapabilitySelection("task", (tool_capability, CATALOG[0]))
        assert selection.tool_names == frozenset({"t"})


@given(
    st.lists(
        st.sampled_from(["alpha", "beta", "gamma", "delta", "epsilon", "zeta"]),
        unique=True,
        max_size=6,
    ),
    st.text(alphabet="abcdefghijklmnopqrstuvwxyz ", max_size=40),
    st.integers(1, 6),
)
def test_routing_selection_is_a_bounded_subset(
    names: list[str], task: str, max_results: int
) -> None:
    catalog = tuple(describe(name, f"{name} service") for name in names)
    router = DefaultCapabilityRouter(
        embeddings=KeywordEmbeddings(["alpha", "beta", "gamma"]),
        max_results=max_results,
    )
    selected = router.route(task, catalog).names
    assert set(selected) <= set(names)
    assert len(selected) == len(set(selected))
    assert len(selected) == min(len(names), max_results)
