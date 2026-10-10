import json
import numpy as np
from llm_sdk import Small_LLM_Model


model = Small_LLM_Model()

with open("data/input/functions_definition.json") as f:
    functions = json.load(f)

id_to_str = {t: model.decode([t]) for t in range(151643)}
clean = {t: s for t, s in id_to_str.items() if s 
         and "\ufffd" not in s}


def pick_token(logits, allowed_ids):
    scores = np.full(len(logits), -np.inf)
    for t in allowed_ids:
        scores[t] = logits[t]
    return int(np.argmax(scores))


digit_ids = [t for t, s in clean.items() if len(s) == 1
             and s in "0123456789"]
comma = [t for t, s in clean.items() if s == ","]
brace = [t for t, s in clean.items() if s == "}"]

print(len(digit_ids), comma, brace)

request = "What is the sum of 265 and 345?"

text = "Available functions:\n"
for fn in functions:
    text += f"- {fn['name']}: {fn['description']}\n"
text += f"\nRequest: {request}\n"
text += 'Answer: {"name": "fn_add_numbers", "parameters": {"a": '

ids = model.encode(text).tolist()[0]

allowed_num = digit_ids + comma + brace


def gen_num(ids):
    value = ""
    while True:
        logits = model.get_logits_from_input_ids(ids)
        tid = pick_token(logits, allowed_num)
        if tid in comma + brace:
            break
        value += clean[tid]
        ids.append(tid)
    return value


a = gen_num(ids)
ids += model.encode(', "b": ').tolist()[0]
b = gen_num(ids)
print(repr(a), repr(b))


safe = [t for t, s in clean.items() if '"' not in s 
        and "\\" not in s and "\n" not in s]
quote = [t for t, s in clean.items() if s == '"']
stop = [t for t, s in clean.items() if s.startswith('"')]


def gen_string(ids):
    value = ""
    allowed = safe + stop
    for _ in range(50):
        logits = model.get_logits_from_input_ids(ids)
        tid = pick_token(logits, allowed)
        if tid in stop:
            break
        value += clean[tid]
        ids.append(tid)
    print("DEBUG string:", repr(value))
    return value


def gen_value(type_name, ids):
    if type_name in ("number", "integer"):
        return gen_num(ids)
    elif type_name == "string":
        return gen_string(ids)
    else:
        raise ValueError(f"unsupported type: {type_name}")


def forced_text(param, type_name, first):
    if first:
        start = '{"'
    else:
        start = ', "'
    text = start + param + '": '
    if type_name == "string":
        text += '"'
    return text


def gen_params(fn, ids):
    values = {}
    first = True
    for name, spec in fn["parameters"].items():
        if spec["type"] == "string":
            ids += model.encode('"').tolist()[0]
        forced = forced_text(name, spec["type"], first)
        ids += model.encode(forced).tolist()[0]
        value = gen_value(spec["type"], ids)
        if spec["type"] in ("number", "integer"):
            value = float(value)
        values[name] = value
        first = False
    return values


def choose_name(ids, names):
    generated = ""
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
    return generated


#tests
request = "Reverse the string 'hello'"
text = "Available functions:\n"
for fn in functions:
    text += f"- {fn['name']}: {fn['description']}\n"
text += f"\nRequest: {request}\n"
text += 'Answer: {"name": "fn_reverse_string", "parameters": {"s": "'
ids = model.encode(text).tolist()[0]
# print(repr(gen_string(ids)))  

print(repr(gen_value("string", ids)))
print(repr(forced_text("a", "number", True)))
print(repr(forced_text("s", "string", True)))
print(repr(forced_text("replacement", "string", False)))

request = "Replace all vowels in 'Programming is fun' with asterisks"
text = "Available functions:\n"
for f in functions:
    text += f"- {f['name']}: {f['description']}\n"
text += f"\nRequest: {request}\n"
text += 'Answer: {"name": "fn_substitute_string_with_regex", "parameters": '
ids = model.encode(text).tolist()[0]

fn = next(f for f in functions if f["name"] == "fn_substitute_string_with_regex")
print(gen_params(fn, ids))

names = [f["name"] for f in functions]
request = "Greet john"
text = "Available functions:\n"
for f in functions:
    text += f"- {f['name']}: {f['description']}\n"
text += f"\nRequest: {request}\n"
text += 'Answer: {"name": "'
ids = model.encode(text).tolist()[0]
print(choose_name(ids, names))
