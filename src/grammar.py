"""Functions for classifying vocabulary tokens for JSON-constrained decoding."""
import re

def classify_string_safe_tokens(
    id_to_str: dict[int, str]
) -> tuple[set[int], set[int]]:
    """Split token ids into (safe, needs_check) for use inside a JSON string."""
    safe = set()
    needs_check = set()

    for tid, s in id_to_str.items():
        if '"' in s or '\\' in s:
            needs_check.add(tid)
        else:
            safe.add(tid)

    return safe, needs_check


def find_string_start_tokens(id_to_str: dict[int, str]) -> set[int]:
    """Find every token id whose string is exactly a single double quote."""
    result = set()

    for tid, s in id_to_str.items():
        if s == '"':
            result.add(tid)
    return result


def find_number_start_tokens(id_to_str: dict[int, str]) -> set[int]:
    """Find every token id whose string is exactly a single digit 0-9."""
    result = set()
    for tid, s in id_to_str.items():
        if re.fullmatch(r'[0-9]', s):
            result.add(tid)
    return result
