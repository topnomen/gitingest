#!/usr/bin/env python3

"""Guide the model to generate valid JSON with token constraints."""

import json
from typing import Dict, List, Set


class JSONValidator:
    """Track the state of the JSON being generated.

    Args:
        None.

    Returns:
        None.
    """

    def __init__(self) -> None:
        """Initialize the validator state.

        Returns:
            None.
        """
        self.text: str = ""

    def add(self, token: str) -> None:
        """Append a token to the current JSON text.

        Args:
            token: Token string to append.

        Returns:
            None.
        """
        if token is None:
            return
        self.text += str(token)

    def is_complete(self) -> bool:
        """Check whether the current JSON is complete.

        Returns:
            True if the JSON has matching braces; otherwise False.
        """
        open_braces = self.text.count('{')
        close_braces = self.text.count('}')
        return open_braces > 0 and open_braces == close_braces

    def is_valid(self) -> bool:
        """Check whether the current text is valid JSON.

        Returns:
            True if the text parses as JSON; otherwise False.
        """
        try:
            json.loads(self.text)
            return True
        except (json.JSONDecodeError, ValueError):
            return False

    def get_state(self) -> str:
        """Return the current JSON generation state.

        Returns:
            A state name describing what type of token is expected next.
        """
        text = self.text
        if not text:
            return "start"
        if self.is_complete():
            return "complete"
        if text.endswith('{'):
            return "open_brace"
        if text.endswith('"'):
            return "quote"
        if text.endswith(':'):
            return "colon"
        if text.endswith(','):
            return "comma"
        if text.endswith('}'):
            return "close_brace"
        return "inside"


class TokenCache:
    """Cache valid tokens for each JSON-generation state.

    Args:
        vocab: Mapping of token ids to token strings.
    """

    def __init__(self, vocab: Dict[int, str]) -> None:
        """Initialize the token cache.

        Args:
            vocab: Mapping of token ids to token strings.

        Returns:
            None.
        """
        self.vocab = vocab
        self.cache: Dict[str, List[int]] = {}

    def get_valid_tokens(self, state: str) -> List[int]:
        """Return the valid tokens for a given JSON state.

        Args:
            state: The current JSON-generation state.

        Returns:
            List of valid token ids for the given state.
        """
        if state not in self.cache:
            valid_tokens = []
            for token_id, token_str in self.vocab.items():
                if self._is_valid_for_state(token_str, state):
                    valid_tokens.append(token_id)
            self.cache[state] = valid_tokens
        return self.cache[state]

    def _is_valid_for_state(self, token: str, state: str) -> bool:
        """Check whether a token is valid in a specific JSON state.

        Args:
            token: Token string to test.
            state: Current JSON-generation state.

        Returns:
            True if the token is valid for that state; otherwise False.
        """
        if state == "start":
            return token == '{'
        if state == "complete":
            return False
        if state == "open_brace":
            return token == '"'
        if state == "quote":
            return token not in ['{', '}', ':', ',']
        if state == "colon":
            return token not in ['{', '}', ':', ',']
        if state == "comma":
            return token == '"'
        if state == "close_brace":
            return token in [',', '}']
        return token not in ['{']

    def clear(self) -> None:
        """Clear the cached token lists.

        Returns:
            None.
        """
        self.cache.clear()


def apply_constraints(
    logits: List[float],
    token_cache: TokenCache,
    validator: JSONValidator
) -> List[float]:
    """Apply JSON validity constraints to the model logits.

    Args:
        logits: Raw logits for each token.
        token_cache: Cache of valid tokens for each state.
        validator: Validator tracking the current JSON text.

    Returns:
        A constrained logits list where Invalid tokens are set
         to negative infinity.
    """
    constrained = list(logits)
    state = validator.get_state()
    valid_tokens = token_cache.get_valid_tokens(state)
    valid_set: Set[int] = set(valid_tokens)
    for token_id in range(len(constrained)):
        if token_id not in valid_set:
            constrained[token_id] = float('-inf')
    return constrained
