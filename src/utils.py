"""Load and validate the JSON input files."""

import json
import sys
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from src.models import FunctionsDefinition, PromptEntries

T = TypeVar("T", bound=BaseModel)


def load_functions_definition(path: Path) -> FunctionsDefinition:
    """Load and validate the functions definition file.

    Exits with a clear error message on failure instead of letting
    an unhandled exception propagate.
    """
    return _load_and_validate(
        path, FunctionsDefinition, "functions definition"
    )


def load_prompt_entries(path: Path) -> PromptEntries:
    """Load and validate the prompts file."""
    return _load_and_validate(path, PromptEntries, "prompt entries")


def _load_and_validate(path: Path, model: type[T], label: str) -> T:
    """Load a JSON file and validate it against a pydantic model."""
    if not path.exists():
        print(f"Error: {label} file not found at '{path}'.",
              file=sys.stderr)
        sys.exit(1)

    try:
        with path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
    except OSError as exc:
        print(f"Error: cannot read {label} file '{path}': {exc}",
              file=sys.stderr)
        sys.exit(1)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        print(f"Error: {label} file '{path}' is not valid JSON: {exc}",
              file=sys.stderr)
        sys.exit(1)

    try:
        return model.model_validate(raw)
    except ValidationError as exc:
        print(f"Error: {label} file '{path}' has the wrong format:\n{exc}",
              file=sys.stderr)
        sys.exit(1)
