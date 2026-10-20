#!/usr/bin/env python3

"""Run the model pipeline that selects and validates function calls."""

import json
import os
import sys
from typing import List, Tuple, Any

from src.constrained_decoder import (
    JSONValidator,
    TokenCache,
    apply_constraints)
from src.decoder import extract_json
from src.model import FunctionDefinition
from src.prompt_builder import build_prompt
from src.validator import validate_function_call
from src.vocabulary import load_vocabulary

DEFAULT_FUNCTIONS = "data/input/functions_definition.json"
DEFAULT_INPUT = "data/input/function_calling_tests.json"
DEFAULT_OUTPUT = "data/output/function_calls.json"


def parse_arguments() -> Tuple[str, str, str]:
    """
    Parse the command-line arguments.

    Supports:
        --functions_definition <path>
        --input <path>
        --output <path>

    Returns:
        A tuple containing the functions file, input file,
        and output file paths.
    """
    functions_file = DEFAULT_FUNCTIONS
    input_file = DEFAULT_INPUT
    output_file = DEFAULT_OUTPUT

    args = sys.argv[1:]

    for i, arg in enumerate(args):
        if arg == "--functions_definition":
            if i + 1 < len(args):
                functions_file = args[i + 1]
        elif arg == "--input":
            if i + 1 < len(args):
                input_file = args[i + 1]
        elif arg == "--output":
            if i + 1 < len(args):
                output_file = args[i + 1]

    return functions_file, input_file, output_file


def load_json_file(filepath: str) -> Any:
    """Load a JSON file from disk.

    Args:
        filepath: Path to the JSON file to read.

    Returns:
        The parsed JSON data.

    Raises:
        SystemExit: If the file is missing or the JSON is invalid.
    """
    try:
        with open(filepath, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        print(f" File '{filepath}' not found")
        sys.exit(1)
    except json.JSONDecodeError as error:
        print(f" Invalid JSON: {error}")
        sys.exit(1)


def load_functions(filepath: str) -> List[FunctionDefinition]:
    """Load function definitions from a JSON file.

    Args:
        filepath: Path to the JSON file containing the function definitions.

    Returns:
        A list of function definition objects.
    """
    data = load_json_file(filepath)
    return [FunctionDefinition(**item) for item in data]


def load_prompts(filepath: str) -> List[str]:
    """Load the prompts from a JSON file.

    Args:
        filepath: Path to the JSON file containing prompts.

    Returns:
        A list of prompt strings.
    """
    data = load_json_file(filepath)
    return [item["prompt"] for item in data if "prompt" in item]


def main() -> None:
    """Run the full function-calling pipeline.

    Returns:
        None.
    """
    print("=== CALL ME MAYBE ===\n")
    functions_file, input_file, output_file = parse_arguments()
    print(f" Functions: {functions_file}")
    print(f" Questions: {input_file}")
    print(f" Output: {output_file}\n")

    functions = load_functions(functions_file)
    print(f" {len(functions)} functions loaded")

    prompts = load_prompts(input_file)
    print(f" {len(prompts)} questions loaded")

    print("\n Loading model...")
    try:
        from llm_sdk import Small_LLM_Model

        model = Small_LLM_Model()
        print("   Model loaded")
    except Exception as error:
        print(f"    Error: {error}")
        sys.exit(1)

    print(" Loading vocabulary...")
    try:
        vocab = load_vocabulary(model)
    except Exception as error:
        print(f"    Error: {error}")
        sys.exit(1)

    print("\n Generating function calls...\n")

    results = []

    for prompt in prompts:
        print(f" {prompt}")

        try:
            full_prompt = build_prompt(prompt, functions)
            input_ids = model.encode(full_prompt)
            input_ids_list = input_ids[0].tolist()

            generated_ids: List[int] = []
            max_tokens = 60
            validator = JSONValidator()
            token_cache = TokenCache(vocab)

            for _ in range(max_tokens):
                logits = model.get_logits_from_input_ids(
                    input_ids_list + generated_ids
                )

                if hasattr(logits, 'tolist'):
                    logits = logits.tolist()

                constrained = apply_constraints(logits, token_cache, validator)
                best_token = max(
                    range(len(constrained)),
                    key=lambda index: constrained[index]
                )

                if constrained[best_token] == float('-inf'):
                    break

                generated_ids.append(best_token)

                token_str = vocab.get(best_token, "")
                validator.add(token_str)

                visualization: str = validator.text.replace(
                    'Ġ', ' ').replace('Ċ', '\n')
                print(f"    {visualization} ...")

                if validator.is_complete() and validator.is_valid():
                    break

            output = model.decode(generated_ids)
            print(f"   {output}")

            data = extract_json(output)
            validated = validate_function_call(data, functions)

            results.append({
                "prompt": prompt,
                "name": validated.name,
                "parameters": validated.parameters
            })

        except Exception as error:
            print(f"    {error}\n")
            results.append({
                "prompt": prompt,
                "name": "error",
                "parameters": {"error": str(error)}
            })

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, 'w') as file:
        json.dump(results, file, indent=2, ensure_ascii=False)
