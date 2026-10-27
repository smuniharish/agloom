"""Property-based tests: the policy judges exactly the arguments tools receive.

For every kind of argument schema Agloom governs:

* a valid call passes the tool exactly the arguments the policy judged, and the
  tool's own validators run once on them;
* an invalid call never runs the tool and never records an agentbarrier action;
* the creation-time idempotency-key check never refuses a key a string satisfies.

Known source bugs are kept as strict `xfail` properties at the end of the module.
"""

from __future__ import annotations

import asyncio
import copy
import functools
import inspect
import json
import math
import tempfile
import typing
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal, Self

import pytest
from agentbarrier.runtime import (
    ArgumentCondition,
    ConditionOperator,
    PolicyEffect,
    PolicyRule,
    RuntimePolicy,
    SQLiteRuntimeStore,
)
from hypothesis import example, given, settings
from hypothesis import strategies as st
from jsonschema.validators import validator_for
from langchain_core.messages import AIMessage
from langchain_core.tools import BaseTool, InjectedToolCallId, StructuredTool
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    ValidationError,
    create_model,
    field_validator,
    model_validator,
)

from agloom import (
    CapabilityResolutionError,
    EventName,
    PolicyConfigurationError,
    PolicyManager,
    create_agent,
)
from agloom.policy._arguments import (
    InvalidArguments,
    call_arguments,
    may_be_string,
    policy_schema,
)
from helpers import Call, EventLog, ScriptedChatModel, tool_call

# --- Values -----------------------------------------------------------------


def _finite(text: str) -> bool:
    """Whether the text does not parse as a non-finite number.

    Non-finite numbers are a known bug, covered at the end of the module.
    """
    try:
        return math.isfinite(float(text))
    except ValueError:
        return True


TEXT = st.text(max_size=8).filter(_finite)
INTEGERS = st.integers(min_value=-(10**6), max_value=10**6)
# Bounded, so validators multiplying by 100 stay finite.
FLOATS = st.floats(min_value=-1e6, max_value=1e6, allow_nan=False)
SCALARS = st.one_of(st.none(), st.booleans(), INTEGERS, FLOATS, TEXT)
JSON_VALUES = st.recursive(
    SCALARS,
    lambda children: (
        st.lists(children, max_size=2) | st.dictionaries(TEXT, children, max_size=2)
    ),
    max_leaves=4,
)
NOT_MAPPINGS = st.one_of(st.lists(SCALARS, max_size=2), INTEGERS, FLOATS, st.booleans())
NO_ARGUMENTS = st.none() | NOT_MAPPINGS
NAMES = ("alpha", "beta", "gamma", "delta")
UNDECLARED = "zeta"

# Values a field of each kind takes as they are, and values it may convert.
VALID: dict[str, st.SearchStrategy[Any]] = {
    "str": TEXT,
    "int": INTEGERS,
    "float": FLOATS,
    "bool": st.booleans(),
}
CONVERTIBLE: dict[str, st.SearchStrategy[Any]] = {
    "str": TEXT.map(lambda text: f"  {text} "),
    "int": INTEGERS.map(str) | INTEGERS.map(float) | FLOATS,
    "float": INTEGERS | INTEGERS.map(str) | FLOATS.map(repr),
    "bool": st.sampled_from([0, 1, "true", "False", "yes", "off", "1", "maybe"]),
}


def likely(kind: str) -> st.SearchStrategy[Any]:
    """Values a field of the kind takes or converts; a few it refuses."""
    return VALID[kind] | CONVERTIBLE[kind]


@st.composite
def model_arguments(
    draw: st.DrawFn,
    required: Mapping[str, st.SearchStrategy[Any]],
    optional: Mapping[str, st.SearchStrategy[Any]],
) -> dict[str, Any]:
    """Arguments a model might send: mostly well formed, otherwise anything.

    Well-formed arguments include every required one and some optional ones;
    the others may miss arguments, add undeclared ones, or hold wrong values.
    """
    if draw(st.integers(0, 2)):
        return draw(st.fixed_dictionaries(dict(required), optional=dict(optional)))
    anything = {
        name: values | JSON_VALUES
        for name, values in {**required, **optional, UNDECLARED: SCALARS}.items()
    }
    return draw(st.fixed_dictionaries({}, optional=anything))


@st.composite
def tool_inputs(
    draw: st.DrawFn,
    arguments: st.SearchStrategy[Any],
    others: st.SearchStrategy[Any] = NO_ARGUMENTS,
) -> Any:
    """Mostly arguments; sometimes no input, or an input that is not a mapping."""
    return draw(others if draw(st.integers(0, 9)) == 0 else arguments)


def as_json(value: Any) -> Any:
    """Return the strict JSON form of a value."""
    return json.loads(json.dumps(value, allow_nan=False))


def run_model_call(tool: BaseTool, args: Mapping[str, Any]) -> Any:
    """Run a tool as LangGraph's tool node runs a model's tool call."""
    return tool.invoke(
        {"type": "tool_call", "name": tool.name, "args": args, "id": "c1"}
    )


def recording_tool(
    args_schema: Any, name: str = "governed"
) -> tuple[StructuredTool, list[dict[str, Any]]]:
    """Return a tool and the keyword arguments each of its calls received."""
    received: list[dict[str, Any]] = []

    def record(**arguments: Any) -> str:
        received.append(arguments)
        return "recorded"

    tool = StructuredTool.from_function(
        record, name=name, description="A governed tool.", args_schema=args_schema
    )
    return tool, received


# --- Tools declaring a Pydantic argument model ------------------------------


class _Required:
    def __repr__(self) -> str:
        return "REQUIRED"


REQUIRED: Any = _Required()
INJECTED = "call_id"
TYPES: dict[str, type] = {"str": str, "int": int, "float": float, "bool": bool}


def _positive(value: float) -> float:
    if value <= 0:
        raise ValueError("must be positive")
    return value


TRANSFORMS: dict[str, Callable[[Any], Any]] = {
    "upper": str.upper,  # normalizes, idempotent
    "suffix": lambda value: f"{value}#",  # not idempotent
    "cents": lambda value: value * 100,  # not idempotent
    "positive": _positive,  # rejects
}
VALIDATORS_OF: dict[str, tuple[str | None, ...]] = {
    "str": (None, "upper", "suffix"),
    "int": (None, "cents", "positive"),
    "float": (None, "cents", "positive"),
    "bool": (None,),
}


def _validator(transform: str) -> Callable[[Any, Any], Any]:
    def check(cls: Any, value: Any) -> Any:
        return TRANSFORMS[transform](value)

    return check


@dataclass(frozen=True)
class FieldSpec:
    """One field of a generated argument model."""

    name: str
    kind: str
    default: Any = REQUIRED
    validator: str | None = None


@dataclass(frozen=True)
class ToolSpec:
    """A tool whose arguments are declared with a generated Pydantic model."""

    fields: tuple[FieldSpec, ...]
    strip: bool = False
    extra: str = "ignore"
    injected_at: int | None = None  # position of an InjectedToolCallId field

    @property
    def order(self) -> tuple[str, ...]:
        names = [spec.name for spec in self.fields]
        if self.injected_at is not None:
            names.insert(self.injected_at, INJECTED)
        return tuple(names)

    def model(self, *, twin: bool = False) -> type[BaseModel]:
        """Return the argument model.

        Its twin has the same fields and settings, but no validators and no
        injected argument.
        """
        specs = {spec.name: spec for spec in self.fields}
        definitions: dict[str, Any] = {}
        for name in self.order:
            if name == INJECTED:
                if not twin:
                    definitions[name] = (Annotated[str, InjectedToolCallId], ...)
                continue
            spec = specs[name]
            default = ... if spec.default is REQUIRED else spec.default
            definitions[name] = (TYPES[spec.kind], default)
        validators = {
            f"check_{spec.name}": field_validator(spec.name)(_validator(spec.validator))
            for spec in self.fields
            if spec.validator and not twin
        }
        return create_model(
            "Arguments",
            __config__=ConfigDict(str_strip_whitespace=self.strip, extra=self.extra),
            __validators__=validators,
            **definitions,
        )

    def build(self) -> tuple[StructuredTool, list[dict[str, Any]]]:
        """Return the tool and the arguments each call of its function received.

        The function takes the model's fields as parameters, in order and with
        the model's defaults, and any other keyword argument.
        """
        received: list[dict[str, Any]] = []
        defaults = {
            spec.name: spec.default
            for spec in self.fields
            if spec.default is not REQUIRED
        }

        def record(*args: Any, **kwargs: Any) -> str:
            positional = dict(zip(self.order, args, strict=False))
            received.append({**defaults, **positional, **kwargs})
            return "recorded"

        tool = StructuredTool.from_function(
            record,
            name="governed",
            description="A governed tool.",
            args_schema=self.model(),
        )
        return tool, received

    def expected(self, raw: Mapping[str, Any]) -> dict[str, Any]:
        """The arguments the tool must receive: each validator applied once."""
        provided = {name: value for name, value in raw.items() if name != INJECTED}
        values = self.model(twin=True).model_validate(provided).model_dump()
        return {
            spec.name: (
                TRANSFORMS[spec.validator](values[spec.name])
                if spec.validator and spec.name in raw
                else values[spec.name]
            )
            for spec in self.fields
        }


@st.composite
def tool_specs(
    draw: st.DrawFn,
    *,
    extras: tuple[str, ...] = ("ignore", "forbid"),
    injected: bool = True,
) -> ToolSpec:
    names = draw(st.lists(st.sampled_from(NAMES), min_size=1, max_size=4, unique=True))
    fields = []
    for name in names:
        kind = draw(st.sampled_from(sorted(TYPES)))
        fields.append(
            FieldSpec(
                name=name,
                kind=kind,
                default=draw(st.just(REQUIRED) | VALID[kind]),
                validator=draw(st.sampled_from(VALIDATORS_OF[kind])),
            )
        )
    position = st.none() | st.integers(0, len(fields)) if injected else st.none()
    return ToolSpec(
        fields=tuple(fields),
        strip=draw(st.booleans()),
        extra=draw(st.sampled_from(extras)),
        injected_at=draw(position),
    )


def raw_arguments(spec: ToolSpec) -> st.SearchStrategy[Any]:
    required = {
        field.name: likely(field.kind)
        for field in spec.fields
        if field.default is REQUIRED
    }
    optional = {
        field.name: likely(field.kind)
        for field in spec.fields
        if field.default is not REQUIRED
    }
    if spec.injected_at is not None:
        # A model may supply an injected argument too.
        optional[INJECTED] = JSON_VALUES
    return tool_inputs(model_arguments(required, optional))


@st.composite
def pydantic_calls(
    draw: st.DrawFn, *, extras: tuple[str, ...] = ("ignore", "forbid")
) -> tuple[ToolSpec, Any]:
    spec = draw(tool_specs(extras=extras))
    return spec, draw(raw_arguments(spec))


@given(case=pydantic_calls())
def test_pydantic_tools_receive_exactly_the_judged_arguments(
    case: tuple[ToolSpec, Any],
) -> None:
    spec, raw = case
    tool, received = spec.build()
    original = copy.deepcopy(raw)
    try:
        judged = call_arguments(tool, raw)
    except InvalidArguments:
        if raw is None or isinstance(raw, Mapping):
            # LangChain's own validation refuses the call as well.
            with pytest.raises(ValidationError):
                run_model_call(tool, raw or {})
        assert received == []
        return
    # The input passes on unchanged, because LangChain validates it once more.
    assert judged.values == original
    run_model_call(tool, judged.values or {})
    assert raw == original
    (arguments,) = received
    if spec.injected_at is not None:
        # LangChain injects the call id; the model cannot choose it.
        assert arguments.pop(INJECTED) == "c1"
    assert arguments == spec.expected(raw or {})
    assert as_json(arguments) == judged.policy


# --- Tools whose schema LangChain infers from `_run` ------------------------


class Search(BaseTool):
    """A tool without an argument schema; LangChain does not validate its input."""

    name: str = "search"
    description: str = "Search the catalog."
    calls: list[dict[str, Any]] = Field(default_factory=list)

    def _run(
        self,
        query: str,
        limit: int = 10,
        exact: bool = False,
        ratio: float = 0.5,
        label: str | None = None,
    ) -> str:
        self.calls.append(
            {
                "query": query,
                "limit": limit,
                "exact": exact,
                "ratio": ratio,
                "label": label,
            }
        )
        return "found"


SEARCH = Search()
SEARCH_INPUTS = tool_inputs(
    model_arguments(
        {"query": likely("str")},
        {
            "limit": likely("int"),
            "exact": likely("bool"),
            "ratio": likely("float"),
            "label": st.none() | likely("str"),
        },
    ),
    # A plain string is the first argument.
    others=NO_ARGUMENTS | TEXT,
)


@functools.cache
def _adapter(annotation: Any) -> TypeAdapter[Any]:
    return TypeAdapter(annotation)


def fits_signature(function: Callable[..., Any], arguments: Mapping[str, Any]) -> bool:
    """Whether the arguments bind to the function's parameters and types."""
    hints = typing.get_type_hints(function)
    for parameter in inspect.signature(function).parameters.values():
        if parameter.name == "self":
            continue
        if parameter.name not in arguments:
            if parameter.default is inspect.Parameter.empty:
                return False
            continue
        try:
            _adapter(hints[parameter.name]).validate_python(arguments[parameter.name])
        except ValidationError:
            return False
    return True


@given(raw=SEARCH_INPUTS)
def test_inferred_schemas_pass_on_exactly_the_judged_values(raw: Any) -> None:
    SEARCH.calls.clear()
    # A plain string is the first argument; no input means no arguments.
    arguments = {"query": raw} if isinstance(raw, str) else {} if raw is None else raw
    accepted = isinstance(arguments, Mapping) and fits_signature(Search._run, arguments)
    try:
        judged = call_arguments(SEARCH, raw)
    except InvalidArguments:
        assert not accepted
        return
    assert accepted
    run_model_call(SEARCH, judged.values)
    # LangChain passes the validated values, defaults included, on to `_run`.
    assert SEARCH.calls == [judged.values]
    assert as_json(judged.values) == judged.policy


# --- Tools declaring a JSON schema, such as MCP tools -----------------------

JSON_KINDS: dict[str, st.SearchStrategy[Any]] = {
    "string": TEXT,
    "integer": INTEGERS,
    "number": FLOATS | INTEGERS,
    "boolean": st.booleans(),
}


def filled(schema: Mapping[str, Any], arguments: Mapping[str, Any]) -> dict[str, Any]:
    """The arguments with the defaults of absent top-level properties."""
    defaults = {
        name: spec["default"]
        for name, spec in schema["properties"].items()
        if "default" in spec and name not in arguments
    }
    return {**arguments, **defaults}


def is_valid(schema: Mapping[str, Any], instance: Any) -> bool:
    return validator_for(schema)(schema).is_valid(instance)


@st.composite
def json_schemas(draw: st.DrawFn) -> dict[str, Any]:
    names = draw(st.lists(st.sampled_from(NAMES), min_size=1, max_size=4, unique=True))
    properties: dict[str, Any] = {}
    for name in names:
        kind = draw(st.sampled_from(sorted(JSON_KINDS)))
        spec: dict[str, Any] = {"type": kind}
        if draw(st.booleans()):
            # Mostly a valid default; sometimes one the schema rejects.
            spec["default"] = draw(JSON_KINDS[kind] | SCALARS)
        properties[name] = spec
    schema: dict[str, Any] = {"type": "object", "properties": properties}
    required = draw(st.lists(st.sampled_from(names), unique=True))
    if required:
        schema["required"] = required
    additional = draw(st.sampled_from([None, True, False]))
    if additional is not None:
        schema["additionalProperties"] = additional
    return schema


@st.composite
def json_schema_calls(draw: st.DrawFn) -> tuple[dict[str, Any], Any]:
    schema = draw(json_schemas())
    properties = schema["properties"]
    # Required arguments without a default; others may take their default.
    needed = {
        name for name in schema.get("required", ()) if "default" not in properties[name]
    }
    values = {name: JSON_KINDS[spec["type"]] for name, spec in properties.items()}
    arguments = model_arguments(
        {name: strategy for name, strategy in values.items() if name in needed},
        {name: strategy for name, strategy in values.items() if name not in needed},
    )
    return schema, draw(tool_inputs(arguments, others=NO_ARGUMENTS | TEXT))


@given(case=json_schema_calls())
def test_json_schema_tools_receive_exactly_the_judged_arguments(
    case: tuple[dict[str, Any], Any],
) -> None:
    schema, raw = case
    tool, received = recording_tool(schema)
    original = copy.deepcopy(raw)
    arguments = {} if raw is None else raw
    accepted = isinstance(arguments, Mapping) and is_valid(
        schema, filled(schema, arguments)
    )
    try:
        judged = call_arguments(tool, raw)
    except InvalidArguments:
        assert not accepted
        return
    assert accepted
    assert judged.values == filled(schema, arguments)
    assert judged.policy == as_json(judged.values)
    # Defaults are filled into a copy of the input.
    assert raw == original
    run_model_call(tool, judged.values)
    assert received == [judged.values]


# --- may_be_string ------------------------------------------------------------

JSON_TYPES = ("string", "integer", "number", "boolean", "null", "array", "object")
SAMPLE_STRINGS = ("", "a", "0", "1.5", "true", "null", "text")
CONSTANTS = st.sampled_from(["", "a", "0", 0, 1.5, True, None])
LEAF_SCHEMAS = st.one_of(
    st.booleans(),
    st.just({}),
    st.builds(lambda kind: {"type": kind}, st.sampled_from(JSON_TYPES)),
    st.builds(
        lambda kinds: {"type": kinds},
        st.lists(st.sampled_from(JSON_TYPES), max_size=3, unique=True),
    ),
    st.builds(
        lambda values: {"enum": values}, st.lists(CONSTANTS, min_size=1, max_size=3)
    ),
    st.builds(lambda value: {"const": value}, CONSTANTS),
    st.builds(
        lambda length: {"type": "string", "minLength": length}, st.integers(0, 3)
    ),
)


def _compound(children: st.SearchStrategy[Any]) -> st.SearchStrategy[Any]:
    options = st.lists(children, min_size=1, max_size=3)
    # Boolean schemas as equivalent objects, so their keywords can be merged.
    objects = children.map(
        lambda schema: (
            schema if isinstance(schema, dict) else {} if schema else {"not": {}}
        )
    )
    return st.one_of(
        st.builds(lambda items: {"anyOf": items}, options),
        st.builds(lambda items: {"oneOf": items}, options),
        st.builds(lambda items: {"allOf": items}, options),
        st.builds(lambda item: {"not": item}, children),
        # Keywords side by side, such as a type next to an anyOf.
        st.builds(lambda first, second: {**first, **second}, objects, objects),
    )


SUBSCHEMAS = st.recursive(LEAF_SCHEMAS, _compound, max_leaves=6)


@given(spec=SUBSCHEMAS)
@example(spec={"type": "string"})
@example(spec={"anyOf": [{"type": "integer"}, {"enum": ["a"]}]})
@example(spec={"type": ["integer", "null"], "anyOf": [{"type": "string"}]})
@example(spec=False)
def test_may_be_string_never_refuses_a_key_a_string_satisfies(spec: Any) -> None:
    schema = {"type": "object", "properties": {"k": spec}, "required": ["k"]}
    if any(is_valid(schema, {"k": text}) for text in SAMPLE_STRINGS):
        assert may_be_string(schema, "k")


@given(spec=SUBSCHEMAS, name=st.sampled_from(NAMES))
def test_may_be_string_refuses_undeclared_keys(spec: Any, name: str) -> None:
    assert not may_be_string({"type": "object", "properties": {name: spec}}, "k")
    assert not may_be_string({"type": "object"}, "k")


# --- Edge cases ---------------------------------------------------------------


class AliasedRefund(BaseModel):
    """Refund arguments whose idempotency key has an alias."""

    request_id: str = Field(alias="requestId")
    amount: int = 1


class AliasedRefundByName(AliasedRefund):
    model_config = ConfigDict(validate_by_name=True)


@given(
    model=st.sampled_from([AliasedRefund, AliasedRefundByName]),
    key=st.sampled_from(["request_id", "requestId"]),
    value=TEXT,
    amount=INTEGERS,
)
def test_aliased_idempotency_keys_are_judged_under_their_field_name(
    model: type[BaseModel], key: str, value: str, amount: int
) -> None:
    tool, received = recording_tool(model)
    schema = policy_schema(tool)
    # LangChain's tool-call schema drops aliases: the model sees, and the
    # creation-time check accepts, the field name.
    assert set(schema["properties"]) == {"request_id", "amount"}
    assert may_be_string(schema, "request_id")
    assert not may_be_string(schema, "requestId")
    raw = {key: value, "amount": amount}
    try:
        judged = call_arguments(tool, raw)
    except InvalidArguments:
        # Unless the model also validates by name, only the alias validates, so
        # calls that follow the schema fail closed, in LangChain as well.
        assert (model, key) == (AliasedRefund, "request_id")
        with pytest.raises(ValidationError):
            run_model_call(tool, raw)
        return
    run_model_call(tool, judged.values)
    (arguments,) = received
    assert set(judged.policy) == set(schema["properties"])
    assert judged.policy["request_id"] == arguments["request_id"] == value
    assert as_json(arguments) == judged.policy


class Order(BaseModel):
    """Arguments whose model validators read several fields."""

    order_id: str
    quantity: int = 1
    price: float = 1.0
    currency: Literal["USD", "cents"] = "cents"
    total: float = 0.0

    @model_validator(mode="before")
    @classmethod
    def dollars_to_cents(cls, data: Any) -> Any:
        # Not idempotent: running it twice would convert twice.
        if (
            isinstance(data, dict)
            and data.get("currency") == "USD"
            and isinstance(data.get("price"), int | float)
        ):
            return {**data, "price": data["price"] * 100}
        return data

    @model_validator(mode="after")
    def bulk_orders_need_a_bulk_id(self) -> Self:
        if self.quantity > 10 and not self.order_id.startswith("bulk-"):
            raise ValueError("orders of more than 10 items need a bulk- order id")
        self.total = self.quantity * self.price
        return self


ORDERS = model_arguments(
    {"order_id": TEXT | TEXT.map(lambda text: f"bulk-{text}")},
    {
        # Around the bulk threshold, which depends on the order id.
        "quantity": st.integers(0, 20) | st.integers(0, 20).map(str),
        "price": FLOATS | INTEGERS,
        "currency": st.sampled_from(["USD", "cents"]),
        "total": FLOATS,
    },
)


@given(raw=ORDERS)
def test_model_validators_reading_other_fields_run_once(raw: dict[str, Any]) -> None:
    tool, received = recording_tool(Order)
    try:
        judged = call_arguments(tool, raw)
    except InvalidArguments:
        with pytest.raises(ValidationError):
            run_model_call(tool, raw)
        assert received == []
        return
    run_model_call(tool, judged.values)
    (arguments,) = received
    assert as_json(arguments) == judged.policy
    assert arguments["total"] == arguments["quantity"] * arguments["price"]
    if raw.get("currency") == "USD" and isinstance(raw.get("price"), int | float):
        assert arguments["price"] == raw["price"] * 100


# --- End to end ---------------------------------------------------------------

ALLOW_ALL = RuntimePolicy(
    version="properties-v1",
    rules=(PolicyRule("allow everything", PolicyEffect.ALLOW, tool="*"),),
)


@dataclass(frozen=True)
class Harness:
    """A governed agent, its agentbarrier store, and the events it emitted."""

    agent: Any
    store: SQLiteRuntimeStore
    events: EventLog

    def actions(self) -> list[Any]:
        return list(self.store.list_actions())

    def denials(self) -> list[dict[str, Any]]:
        return [
            dict(event.payload)
            for event in self.events
            if event.name == EventName.TOOL_CALL_DENIED
        ]


@contextmanager
def governed_agent(
    tool: BaseTool,
    args: Mapping[str, Any],
    *,
    consequential: Mapping[str, str],
    policy: RuntimePolicy = ALLOW_ALL,
) -> Iterator[Harness]:
    """An agent whose model calls the tool once with the arguments, then answers.

    Each call gets a fresh agentbarrier store, closed afterwards.
    """

    def respond(call: Call) -> AIMessage | str:
        if call.last.type == "tool":
            return f"final: {call.last.text}"
        return tool_call(tool.name, args)

    events = EventLog()
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as folder:
        store = SQLiteRuntimeStore(str(Path(folder) / "barrier.db"))
        try:
            agent = create_agent(
                model=ScriptedChatModel(responder=respond),
                tools=[tool],
                pattern="react",
                policy=PolicyManager(
                    policy=policy,
                    store=store,
                    consequential=consequential,
                    on_denied="inform_model",
                ),
                observers=[events],
            )
            yield Harness(agent, store, events)
        finally:
            store.close()


class ChargeArguments(BaseModel):
    """A charge in dollars that the tool's own model normalizes into cents."""

    model_config = ConfigDict(str_strip_whitespace=True)

    order_id: str
    amount: int
    note: str = "none"

    @field_validator("order_id")
    @classmethod
    def canonical(cls, value: str) -> str:
        return value.upper()

    @field_validator("amount")
    @classmethod
    def to_cents(cls, value: int) -> int:
        return value * 100


def charge_tool() -> tuple[BaseTool, list[dict[str, Any]]]:
    received: list[dict[str, Any]] = []

    def charge(order_id: str, amount: int, note: str = "none") -> str:
        received.append({"order_id": order_id, "amount": amount, "note": note})
        return "charged"

    tool = StructuredTool.from_function(
        charge,
        name="charge",
        description="Charge an order in dollars.",
        args_schema=ChargeArguments,
    )
    return tool, received


def validated_charge(args: Mapping[str, Any]) -> Mapping[str, Any] | None:
    try:
        return ChargeArguments.model_validate(args).model_dump()
    except ValidationError:
        return None


TRANSFER_SCHEMA = {
    "type": "object",
    "properties": {
        "transfer_id": {"type": "string"},
        "amount": {"type": "number", "default": 5000},
    },
    "required": ["transfer_id"],
    "additionalProperties": False,
}


def transfer_tool() -> tuple[BaseTool, list[dict[str, Any]]]:
    return recording_tool(TRANSFER_SCHEMA, name="transfer")


def validated_transfer(args: Mapping[str, Any]) -> Mapping[str, Any] | None:
    arguments = filled(TRANSFER_SCHEMA, args)
    return arguments if is_valid(TRANSFER_SCHEMA, arguments) else None


def search_tool() -> tuple[BaseTool, list[dict[str, Any]]]:
    tool = Search()
    return tool, tool.calls


def validated_search(args: Mapping[str, Any]) -> Mapping[str, Any] | None:
    return args if fits_signature(Search._run, args) else None


@dataclass(frozen=True)
class Governed:
    """A consequential tool, the arguments a model sends it, and their validation."""

    key: str
    build: Callable[[], tuple[BaseTool, list[dict[str, Any]]]]
    arguments: st.SearchStrategy[dict[str, Any]]
    validate: Callable[[Mapping[str, Any]], Mapping[str, Any] | None]

    def accepts(self, args: Mapping[str, Any]) -> bool:
        """Whether the tool's own validation accepts the call and its key."""
        validated = self.validate(args)
        if validated is None:
            return False
        key = validated[self.key]
        return isinstance(key, str) and bool(key.strip())


KEYS = st.text(min_size=1, max_size=8) | st.sampled_from(["", " ", " r-1 ", "R-1"])


GOVERNED = {
    "charge": Governed(
        "order_id",
        charge_tool,
        model_arguments(
            {"order_id": KEYS, "amount": INTEGERS | INTEGERS.map(str)},
            {"note": TEXT},
        ),
        validated_charge,
    ),
    "transfer": Governed(
        "transfer_id",
        transfer_tool,
        model_arguments({"transfer_id": KEYS}, {"amount": FLOATS | INTEGERS}),
        validated_transfer,
    ),
    "search": Governed(
        "query",
        search_tool,
        model_arguments({"query": KEYS}, {"limit": INTEGERS | INTEGERS.map(str)}),
        validated_search,
    ),
}


@settings(max_examples=30)
@given(
    name=st.sampled_from(sorted(GOVERNED)), asynchronous=st.booleans(), data=st.data()
)
def test_actions_record_exactly_what_governed_tools_receive(
    name: str, asynchronous: bool, data: st.DataObject
) -> None:
    case = GOVERNED[name]
    args = data.draw(case.arguments, label="args")
    tool, received = case.build()
    with governed_agent(tool, args, consequential={name: case.key}) as governed:
        if asynchronous:
            answer = asyncio.run(governed.agent.ainvoke("Go"))
        else:
            answer = governed.agent.invoke("Go")
        actions = governed.actions()
    if case.accepts(args):
        (arguments,) = received
        (action,) = actions
        assert dict(action.arguments) == as_json(arguments)
        assert action.idempotency_key == arguments[case.key]
        assert governed.denials() == []
    else:
        assert received == []
        assert actions == []
        assert governed.denials() == [{"tool": name, "reason": "InvalidArguments"}]
        assert "was not executed" in answer.text


@settings(max_examples=30)
@given(
    name=st.sampled_from(sorted(GOVERNED)), asynchronous=st.booleans(), data=st.data()
)
def test_direct_execution_records_exactly_what_tools_receive(
    name: str, asynchronous: bool, data: st.DataObject
) -> None:
    case = GOVERNED[name]
    args = data.draw(case.arguments, label="args")
    tool, received = case.build()
    refusal: Exception | None = None
    with governed_agent(tool, args, consequential={name: case.key}) as governed:
        try:
            if asynchronous:
                asyncio.run(governed.agent.aexecute_capability(name, args))
            else:
                governed.agent.execute_capability(name, args)
        except (CapabilityResolutionError, PolicyConfigurationError) as error:
            refusal = error
        actions = governed.actions()
    if case.accepts(args):
        assert refusal is None
        (arguments,) = received
        (action,) = actions
        assert dict(action.arguments) == as_json(arguments)
    else:
        # Invalid arguments are a resolution error; a blank key that the
        # application supplies is a configuration error.
        invalid = case.validate(args) is None
        assert isinstance(
            refusal,
            CapabilityResolutionError if invalid else PolicyConfigurationError,
        )
        assert received == []
        assert actions == []


# --- Known bugs ---------------------------------------------------------------
# Each property fails today and passes once its bug is fixed. The explicit
# examples make the failures deterministic.

NORMALIZED = ToolSpec(
    fields=(
        FieldSpec("request_id", "str", validator="upper"),
        FieldSpec("amount", "int", default=5, validator="cents"),
    ),
    strip=True,
)


@st.composite
def string_calls(draw: st.DrawFn) -> tuple[ToolSpec, str]:
    # Tools with an injected call id run only as a model's tool call.
    return draw(tool_specs(injected=False)), draw(TEXT | CONVERTIBLE["str"])


@pytest.mark.xfail(
    strict=True,
    reason="a string input reaches a Pydantic-schema tool unvalidated: LangChain "
    "validates it but passes the raw string on",
)
@given(case=string_calls())
@example(case=(NORMALIZED, " r-1 "))
def test_string_inputs_reach_pydantic_tools_as_judged(
    case: tuple[ToolSpec, str],
) -> None:
    spec, text = case
    tool, received = spec.build()
    try:
        judged = call_arguments(tool, text)
    except InvalidArguments:
        return
    # As Agent.execute_capability runs a tool.
    tool.invoke(judged.values)
    (arguments,) = received
    assert as_json(arguments) == judged.policy


@pytest.mark.xfail(
    strict=True,
    reason="arguments a model with extra='allow' accepts reach the tool, but the "
    "policy judges only the declared fields",
)
@given(case=pydantic_calls(extras=("allow",)))
@example(
    case=(
        ToolSpec(fields=(FieldSpec("disk", "str"),), extra="allow"),
        {"disk": "d1", "force": True},
    )
)
def test_undeclared_arguments_never_reach_tools_unjudged(
    case: tuple[ToolSpec, Any],
) -> None:
    spec, raw = case
    tool, received = spec.build()
    try:
        judged = call_arguments(tool, raw)
    except InvalidArguments:
        return  # Refusing undeclared arguments is safe as well.
    run_model_call(tool, judged.values or {})
    (arguments,) = received
    if spec.injected_at is not None:
        arguments.pop(INJECTED)
    assert as_json(arguments) == judged.policy


class SerializedByAlias(AliasedRefund):
    model_config = ConfigDict(validate_by_name=True, serialize_by_alias=True)


@pytest.mark.xfail(
    strict=True,
    reason="with serialize_by_alias=True the policy names arguments by alias, while "
    "policy_schema and the idempotency-key check use field names",
)
@given(key=st.sampled_from(["request_id", "requestId"]), value=TEXT, amount=INTEGERS)
@example(key="request_id", value="r-1", amount=1)
def test_the_policy_names_arguments_as_policy_schema_does(
    key: str, value: str, amount: int
) -> None:
    tool, _ = recording_tool(SerializedByAlias)
    judged = call_arguments(tool, {key: value, "amount": amount})
    assert set(judged.policy) == set(policy_schema(tool)["properties"])
    assert judged.policy["request_id"] == value


def refund(request_id: str, amount: float) -> str:
    """Refund an amount."""
    return "refunded"


REFUNDS = {
    "pydantic": StructuredTool.from_function(refund),
    "json-schema": StructuredTool.from_function(
        refund,
        args_schema={
            "type": "object",
            "properties": {
                "request_id": {"type": "string"},
                "amount": {"type": "number"},
            },
            "required": ["request_id", "amount"],
        },
    ),
}
NON_FINITE = st.sampled_from(
    [math.nan, math.inf, -math.inf, "nan", "-inf", "Infinity", "1e999"]
)


def is_strict_json(value: Any) -> bool:
    try:
        return as_json(value) == value
    except ValueError:
        return False


@pytest.mark.xfail(
    strict=True,
    reason="non-finite numbers pass validation, though the policy cannot compare "
    "them and agentbarrier cannot record them",
)
@given(kind=st.sampled_from(sorted(REFUNDS)), amount=NON_FINITE | FLOATS)
@example(kind="pydantic", amount="nan")
def test_judged_arguments_are_strict_json(kind: str, amount: Any) -> None:
    try:
        judged = call_arguments(REFUNDS[kind], {"request_id": "r-1", "amount": amount})
    except InvalidArguments:
        return
    assert is_strict_json(judged.policy), judged.policy


LIMITS = RuntimePolicy(
    version="limits-v1",
    rules=(
        PolicyRule(
            "no large refunds",
            PolicyEffect.DENY,
            tool="refund",
            conditions=(ArgumentCondition("amount", ConditionOperator.GT, 100),),
        ),
        PolicyRule("allow the rest", PolicyEffect.ALLOW, tool="*"),
    ),
)


@pytest.mark.xfail(
    strict=True,
    reason="NaN slips past a deny rule on amount > 100, and agentbarrier's refusal "
    "to record it crashes consequential runs",
)
@settings(max_examples=10)
@given(amount=NON_FINITE | INTEGERS, consequential=st.booleans())
@example(amount="nan", consequential=False)
@example(amount="nan", consequential=True)
def test_deny_rules_hold_for_every_amount(amount: Any, consequential: bool) -> None:
    refunded: list[float] = []

    def pay_back(request_id: str, amount: float) -> str:
        """Refund an amount."""
        refunded.append(amount)
        return "refunded"

    with governed_agent(
        StructuredTool.from_function(pay_back, name="refund"),
        {"request_id": "r-1", "amount": amount},
        consequential={"refund": "request_id"} if consequential else {},
        policy=LIMITS,
    ) as governed:
        governed.agent.invoke("Refund r-1")
    assert all(paid <= 100 for paid in refunded), refunded
