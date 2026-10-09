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
request = "Reverse the string 'hello'"
text += f"\nRequest: {request}\n"
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

ids = model.encode(text).tolist()[0]
logits = model.get_logits_from_input_ids(ids)
tid = pick_token(logits, allowed)
print("picked:", tid, repr(clean[tid]))

while generated not in names:
    allowed = []
    for tid, s in clean.items():
        for n in names:
            if n.startswith(generated + s):
                allowed.append(tid)
                break

    logits = model.get_logits_from_input_ids(ids)
    tid = pick_token(logits, allowed)
    generated += clean[tid]
    ids.append(tid)
    print(repr(generated))

fn = next(f for f in functions if f["name"] == generated)
param = list(fn["parameters"])[0]

forced = '", "parameters": {"' + param + '": "'
ids += model.encode(forced).tolist()[0]

print(repr(model.decode(ids[-15:])))

safe = [t for t, s in clean.items() if '"' not in s and
        "\\" not in s and "\n" not in s]
print(len(safe))

logits = model.get_logits_from_input_ids(ids)
tid = pick_token(logits, safe)
print("first value token:", tid, repr(clean[tid]))

quote = [t for t, s in clean.items() if s == '"']
print(quote)

value = ""
allowed_value = safe + quote

while True:
    logits = model.get_logits_from_input_ids(ids)
    tid = pick_token(logits, allowed_value)
    if tid == quote[0]:
        break
    value += clean[tid]
    ids.append(tid)

print(repr(value))

result = {
    "prompt": request,
    "name": generated,
    "parameters": {param: value},
}
print(json.dumps(result))

