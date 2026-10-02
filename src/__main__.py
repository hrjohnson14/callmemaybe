import argparse
from pathlib import Path
from src.utils import load_prompt_entries, load_functions_definition
from src.tokenizer_utils import build_id_to_string
from llm_sdk import Small_LLM_Model
REAL_VOCAB_SIZE = 151643


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="loads functions&prompts")
    parser.add_argument(
        "--functions_definition",
        type=Path,
        default=Path("data/input/functions_definition.json"),
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/input/function_calling_tests.json")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/output/function_calling_results.json")
    )
    return parser.parse_args()


# def main() -> None:
#     args = parse_args()
#     functions = load_functions_definition(args.functions_definition)
#     prompts = load_prompt_entries(args.input)
#     print(f"Loaded {len(functions.root)} functions"
#           f" and {len(prompts.root)} prompts")
#     model = Small_LLM_Model()

#     id_to_str = build_id_to_string(model, REAL_VOCAB_SIZE)
#     print(f"Built vocab lookup with {len(id_to_str)} entries")

def main() -> None:
    print("starting", flush=True)
    args = parse_args()
    print("args parsed", flush=True)

    functions = load_functions_definition(args.functions_definition)
    print("functions loaded", flush=True)

    prompts = load_prompt_entries(args.input)
    print("prompts loaded", flush=True)

    model = Small_LLM_Model()
    print("model loaded", flush=True)

    id_to_str = build_id_to_string(model, REAL_VOCAB_SIZE)
    print(f"Built vocab lookup with {len(id_to_str)} entries", flush=True)


if __name__ == "__main__":
    main()
