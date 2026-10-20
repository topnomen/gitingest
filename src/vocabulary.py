#!/usr/bin/env python3

"""Load the model vocabulary from a JSON file."""

import json
from typing import Dict
from llm_sdk import Small_LLM_Model


def load_vocabulary(model: Small_LLM_Model) -> Dict[int, str]:
    """Load the model vocabulary from a JSON file.

    Args:
        model: Model instance exposing the vocabulary file path.

    Returns:
        Dictionary mapping token ids to token text.
    """
    path = model.get_path_to_vocab_file()
    with open(path, 'r') as file:
        data = json.load(file)

    vocab: Dict[int, str] = {}
    if isinstance(data, dict):
        first_key = list(data.keys())[0] if data else ""
        try:
            int(first_key)
            for key, value in data.items():
                try:
                    token_id = int(key)
                    vocab[token_id] = value
                except (ValueError, TypeError):
                    continue
        except (ValueError, TypeError):
            for key, value in data.items():
                try:
                    token_id = int(value)
                    vocab[token_id] = key
                except (ValueError, TypeError):
                    continue
    elif isinstance(data, list):
        print(f"    Number of elements: {len(data)}")
        for item in data:
            if isinstance(item, dict) and "id" in item and "token" in item:
                vocab[item["id"]] = item["token"]

    print(f"   {len(vocab)} tokens loaded")
    return vocab
