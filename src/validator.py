#!/usr/bin/env python3

"""Validate parsed JSON function-call payloads."""

from typing import Any, Dict, List

from src.model import FunctionCall, FunctionDefinition


def validate_function_call(
        data: Dict[str, Any], function: List[FunctionDefinition]
        ) -> FunctionCall:
    """Validate a function-call payload against
     the available function definitions.

    Args:
        data: Parsed JSON dictionary containing
         the function name and arguments.
        function: List of available function definitions.

    Returns:
        A validated FunctionCall object.

    Raises:
        ValueError: If the payload is missing required fields,
         the function is unknown,
            or a parameter value does not match the expected type.
    """
    if "function" not in data:
        raise ValueError("Missing 'function' field")
    if "arguments" not in data:
        raise ValueError("Missing 'arguments' field")

    function_name = data["function"]
    function_def = next(
        (item for item in function if item.name == function_name), None
        )
    if not function_def:
        raise ValueError(f"\n\nERROR: Function '{function_name}' not found")

    for required_name in function_def.required:
        if required_name not in data["arguments"]:
            raise ValueError(
                f"\n\nERROR: Parameter '{required_name}' not found"
                )
    for name, value in data["arguments"].items():
        if name in function_def.parameters:
            parameter = function_def.parameters[name]
            if (
                parameter.type == "number"
                and not isinstance(value, (int, float))
            ):
                raise ValueError(f"{name} must be a number")
            if parameter.type == "string" and not isinstance(value, str):
                raise ValueError(f"{name} must be a string")

    return FunctionCall(
        prompt="", name=function_name,
        parameters=data["arguments"]
    )
