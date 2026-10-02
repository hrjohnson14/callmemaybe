"""Utilities for maping between token IDs and their string respresentation"""

from llm_sdk import Small_LLM_Model


def build_id_to_string(model: Small_LLM_Model, vocab_size: int) -> dict[int, str]:
    """ Build a lookup table from token ID to its decoded string"""

    return {tid: model.decode([tid]) for tid in range(vocab_size)}
