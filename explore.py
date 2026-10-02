from llm_sdk import Small_LLM_Model
from src.tokenizer_utils import build_id_to_string
from src.grammar import (
    classify_string_safe_tokens,
    find_string_start_tokens,
    find_number_start_tokens,
    find_boolean_start_tokens
)

model = Small_LLM_Model()
id_to_str = build_id_to_string(model, 151643)

safe, needs_check = classify_string_safe_tokens(id_to_str)
print(f"safe: {len(safe)}")
print(f"needs_check: {len(needs_check)}")
print(f"total: {len(safe) + len(needs_check)}, vocab size: {len(id_to_str)}")
print(id_to_str[1], "-> needs_check" if 1 in needs_check else "-> safe")
print(id_to_str[279], "-> needs_check" if 279 in needs_check else "-> safe")

string_start = find_string_start_tokens(id_to_str)
print(f"string_start tokens: {len(string_start)}")
for tid in string_start:
    print(tid, repr(id_to_str[tid]))

number_start = find_number_start_tokens(id_to_str)
print(f"number_start tokens: {len(number_start)}")
for tid in sorted(number_start):
    print(tid, repr(id_to_str[tid]))

boolean_start = find_boolean_start_tokens(id_to_str)
print(f"boolean_start tokens: {len(boolean_start)}")
for tid in sorted(boolean_start):
    print(tid, repr(id_to_str[tid]))
