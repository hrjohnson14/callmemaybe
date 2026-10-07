from llm_sdk import Small_LLM_Model
import numpy as np
import json

model = Small_LLM_Model()
ids = model.encode("What is the sum of 2 and 3?")
print(ids)
print(type(ids))

logits = model.get_logits_from_input_ids(ids.tolist()[0])
print(len(logits))


def pick_token(logits, allowed_ids):
    scores = np.full(len(logits), -np.inf)
    for tid in allowed_ids:
        scores[tid] = logits[tid]
    return int(np.argmax(scores))


free = int(np.argmax(logits))
print("free pick:", free, repr(model.decode([free])))

digits = [tid for tid in range(151643)
          if model.decode([tid]) in "0123456789"]
forced = pick_token(logits, digits)
print("digit-only pick:", forced, repr(model.decode([forced])))

with open("data/input/functions_definition.json") as f:
    functions = json.load(f)

text = "Available functions:\n"
for fn in functions:
    text += f"- {fn['name']}: {fn['description']}\n"
text += "\nRequest: Greet john\n"
text += 'Answer: {"name": "'

print(text)


def build_id_to_string(model, vocab_size):
    return {tid: model.decode([tid]) for tid in range(vocab_size)}


id_to_str = build_id_to_string(model, 151643)
print(len(id_to_str))
print(repr(id_to_str[17]))
print(repr(id_to_str[3555]))

clean = {}
for tid, s in id_to_str.items():
    if s and "\ufffd" not in s:
        clean[tid] = s

print(len(id_to_str), len(clean))

names = [fn["name"] for fn in functions]
generated = ""

allowed = []
for tid, s in clean.items():
    for n in names:
        if n.startswith(generated + s):
            allowed.append(tid)
            break

print(len(allowed))
for tid in allowed[:10]:
    print(tid, repr(clean[tid]))

print("--- reached the end ---")
