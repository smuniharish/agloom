"""The arguments a governed tool call actually runs with."""

from __future__ import annotations

import copy
import inspect
import json
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, cast

from jsonschema.validators import validator_for
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from pydantic import (
    BaseModel,
    PydanticUserError,
    ValidationError,
    create_model,
)
from pydantic_core import PydanticSerializationError

_MAX_PROBLEMS = 5
_UNSUPPORTED = "the tool's argument schema is not a Pydantic v2 model or a JSON schema"
_INFERRED = (
    "the tool declares no argument schema, and its signature takes variadic or "
    "RunnableConfig parameters; declare an args_schema"
)
_VARIADIC = (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)


class InvalidArguments(ValueError):
    """The arguments of a tool call do not match the tool's schema."""


@dataclass(frozen=True)
class CallArguments:
    """Validated arguments of one tool call.

    Attributes:
        values: The input to pass on to the tool. It is unchanged when LangChain
            validates it with the tool's own argument model; otherwise it holds
            the validated values, including defaults.
        policy: The arguments the tool runs with, as JSON: what the policy
            evaluates and records.
    """

    values: Any
    policy: dict[str, Any]


def call_arguments(tool: BaseTool, raw: Any) -> CallArguments:
    """Validate arguments against the tool's schema, as invoking the tool would.

    A declared Pydantic argument model validates with all its validators,
    settings, and defaults, exactly as LangChain does before it calls the tool,
    so the policy judges the values the tool receives. The input itself is
    passed on unchanged, because LangChain validates it once more, and a
    validator that is not idempotent must not run twice. Arguments LangChain
    injects later, such as the tool call id or the graph state, are not part of
    the call.

    Tools whose schema is inferred from their signature, and JSON schemas, are
    not validated by LangChain, so their validated values are passed on: JSON
    schemas are checked strictly and their top-level defaults filled in, so a
    server cannot convert a value the policy never saw.

    Raises:
        InvalidArguments: If the arguments do not match the schema, or the tool
            declares no schema the policy can interpret.
    """
    schema = tool.args_schema
    if isinstance(schema, Mapping):
        return _json_schema_arguments(schema, raw)
    call_schema = _call_schema(tool)
    fields = tuple(call_schema.model_fields)
    if isinstance(schema, type) and issubclass(schema, BaseModel):
        injected = frozenset(schema.model_fields) - set(fields)
        _, policy = _validated(_validating(schema, injected), fields, raw)
        return CallArguments(raw, policy)
    model, policy = _validated(call_schema, fields, raw)
    return CallArguments({name: getattr(model, name) for name in fields}, policy)


def policy_schema(tool: BaseTool) -> Mapping[str, Any]:
    """Return the JSON schema of the arguments the policy judges for a tool.

    Raises:
        InvalidArguments: If the tool does not declare its arguments with a
            Pydantic v2 model or a JSON schema, or they have no JSON form.
    """
    schema = tool.args_schema
    if isinstance(schema, Mapping):
        return schema
    try:
        return _call_schema(tool).model_json_schema()
    except PydanticUserError as error:
        raise InvalidArguments("the tool's arguments have no JSON form") from error


def may_be_string(schema: Mapping[str, Any], name: str) -> bool:
    """Return whether the schema declares `name` and allows it to be a string.

    Unconstrained or referenced types count as possible strings; every call
    still checks the value itself.
    """
    properties = schema.get("properties")
    if not isinstance(properties, Mapping) or name not in properties:
        return False
    return _allows_string(properties[name])


def _allows_string(spec: Any) -> bool:
    if not isinstance(spec, Mapping):
        return spec is not False
    options = [*spec.get("anyOf", ()), *spec.get("oneOf", ())]
    if options:
        return any(_allows_string(option) for option in options)
    declared = spec.get("type")
    if declared is None:
        return True
    return "string" in (declared if isinstance(declared, list) else [declared])


def _call_schema(tool: BaseTool) -> type[BaseModel]:
    """Return the model of the arguments the model supplies, without injected ones.

    Raises:
        InvalidArguments: If the tool declares a schema other than a Pydantic v2
            model, or LangChain cannot infer a complete one from its signature.
    """
    schema = tool.args_schema
    if schema is None:
        return _inferred_schema(tool)
    if not (isinstance(schema, type) and issubclass(schema, BaseModel)):
        raise InvalidArguments(_UNSUPPORTED)
    return cast(type[BaseModel], tool.tool_call_schema)


def _inferred_schema(tool: BaseTool) -> type[BaseModel]:
    """Return the schema LangChain infers from `_run`, when it describes the call.

    Variadic parameters, such as those of a single-input `Tool`, and a
    `RunnableConfig` parameter appear in the inferred schema, but they are not
    arguments the model supplies, and `**kwargs` accepts arguments no schema
    describes, so such tools must declare their arguments.
    """
    schema = cast(type[BaseModel], tool.tool_call_schema)
    parameters = inspect.signature(tool._run).parameters.values()
    if any(parameter.kind in _VARIADIC for parameter in parameters) or any(
        field.annotation is RunnableConfig for field in schema.model_fields.values()
    ):
        raise InvalidArguments(_INFERRED)
    return schema


@lru_cache(maxsize=256)
def _validating(schema: type[BaseModel], injected: frozenset[str]) -> type[BaseModel]:
    """Return the argument model with injected arguments made optional.

    LangChain supplies injected arguments only after the policy judged the call;
    every validator and setting of the model still applies.
    """
    if not injected:
        return schema
    overrides: dict[str, Any] = dict.fromkeys(sorted(injected), (Any, None))
    return create_model(
        schema.__name__,
        __base__=schema,
        __module__=schema.__module__,
        **overrides,
    )


def _validated(
    model_type: type[BaseModel], fields: tuple[str, ...], raw: Any
) -> tuple[BaseModel, dict[str, Any]]:
    """Validate the call and return the model and the JSON arguments it supplies."""
    if isinstance(raw, str) and fields:
        # LangChain passes a plain string input to the tool's first argument.
        raw = {fields[0]: raw}
    try:
        model = model_type.model_validate(_mapping(raw))
        return model, model.model_dump(mode="json", include=set(fields))
    except ValidationError as error:
        raise InvalidArguments(
            "; ".join(
                f"{_location(problem['loc'])}: {problem['msg']}"
                for problem in error.errors()[:_MAX_PROBLEMS]
            )
        ) from error
    except PydanticSerializationError as error:
        raise InvalidArguments("the arguments are not JSON-compatible") from error


def _json_schema_arguments(schema: Mapping[str, Any], raw: Any) -> CallArguments:
    arguments = _mapping(raw)
    for name, spec in (schema.get("properties") or {}).items():
        if name not in arguments and isinstance(spec, Mapping) and "default" in spec:
            arguments[name] = copy.deepcopy(spec["default"])
    validator = validator_for(schema)(schema)
    problems = sorted(
        validator.iter_errors(arguments), key=lambda problem: list(problem.path)
    )
    if problems:
        raise InvalidArguments(
            "; ".join(
                f"{_location(problem.absolute_path)}: {problem.message}"
                for problem in problems[:_MAX_PROBLEMS]
            )
        )
    try:
        policy = json.loads(json.dumps(arguments))
    except (TypeError, ValueError) as error:
        raise InvalidArguments("the arguments are not JSON-compatible") from error
    return CallArguments(arguments, policy)


def _mapping(raw: Any) -> dict[str, Any]:
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise InvalidArguments("the arguments must be an object")
    return dict(raw)


def _location(path: Any) -> str:
    return ".".join(str(part) for part in path) or "arguments"
