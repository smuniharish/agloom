import asyncio

from _support import real_model
from langchain_core.tools import tool
from refresh_engine import (
    DiscoveryResult,
    PlanAction,
    Resource,
    ResourceSnapshot,
)

from agloom import create_agent


class DocumentSource:
    def __init__(self) -> None:
        self.documents = {
            "shipping": "Standard shipping takes five days.",
            "returns": "Returns are accepted within 30 days.",
        }

    async def discover(self) -> DiscoveryResult:
        async def resources():
            for document_id in self.documents:
                yield Resource(resource_id=document_id)

        return DiscoveryResult(resources=resources(), complete=True)

    async def snapshot(self, resource: Resource) -> ResourceSnapshot:
        return ResourceSnapshot(
            resource_id=resource.resource_id,
            content=self.documents[resource.resource_id],
        )


async def main() -> None:
    source = DocumentSource()
    search_index: dict[str, str] = {}
    tool_calls: list[str] = []

    @tool
    def lookup_policy(document_id: str) -> str:
        """Look up a refreshed policy document by identifier."""

        tool_calls.append(document_id)
        return search_index[document_id]

    async def update_index(
        resource: Resource | None,
        snapshot: ResourceSnapshot | None,
        action: PlanAction,
        request,
    ) -> None:
        del request
        if action is PlanAction.DELETE:
            assert resource is None
            assert snapshot is None
            return
        assert resource is not None
        assert snapshot is not None
        search_index[resource.resource_id] = str(snapshot.content)

    agent = create_agent(
        model=real_model(),
        tools=[lookup_policy],
        pattern="react",
        refresh_source=source,
        refresh_operation=update_index,
    )
    engine = agent.capabilities.resolve("refresh_engine")

    initial = await engine.refresh()
    source.documents["shipping"] = "Standard shipping takes three days."
    incremental = await engine.refresh()
    await engine.close()

    assert initial.added_count == 2
    assert initial.refreshed_count == 2
    assert incremental.modified_count == 1
    assert incremental.refreshed_count == 1
    assert search_index["shipping"].endswith("three days.")
    answer = agent.invoke(
        "Call lookup_policy with document_id='shipping', then state the returned "
        "policy exactly."
    )
    print(f"Agent lookup: {answer.content}")
    assert any(
        value in str(answer.content).casefold() for value in ("three days", "3 days")
    )
    assert tool_calls == ["shipping"]
    print(
        f"Initial refresh indexed {initial.refreshed_count} documents; "
        f"incremental refresh updated {incremental.refreshed_count}."
    )
    print(f"Current shipping policy: {search_index['shipping']}")


asyncio.run(main())
