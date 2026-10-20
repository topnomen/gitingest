#!/usr/bin/env python3

"""Build prompts for function-call selection tasks."""

from typing import List

from src.model import FunctionDefinition


def build_prompt(prompt: str, functions: List[FunctionDefinition]) -> str:
    """Build a prompt that asks the model to
     choose one function and return JSON.

    Args:
        prompt: User input to answer.
        functions: Available function definitions.

    Returns:
        Formatted prompt string for the model.
    """
    func_lines = []
    for function in functions:
        parameters = ", ".join([
            f"{name}:{info.type}"
            for name, info in function.parameters.items()
        ])
        func_lines.append(
            f"- {function.name}({parameters}): {function.description}"
            )

    func_list = "\n".join(func_lines)
    return (
        f"Pick ONE function and return its call as JSON.\n\n"
        f"Functions:\n{func_list}\n\n"
        f'Format: {{"function": "fn_name", "arguments": {{...}}}}\n\n'
        f'IMPORTANT: Escape any quote or apostrophe in string values.\n\n'
        f'Example:\n'
        f'Q: What is 5 plus 3?\n'
        f'A: {{"function": "fn_add_numbers", '
        f'"arguments": {{"a": 5, "b": 3}}}}\n\n'
        f'Q: {prompt}\n'
        f'A:'
    )
