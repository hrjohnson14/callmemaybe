from pydantic import BaseModel, RootModel


class Supported_types(str):
    """The supported types the program can handle in
    one place."""
    SUPPORTED_TYPES: frozenset[str] = {
                                "number",
                                "string",
                                "boolean",
                                "integer",
                                }


class TypeSpec(BaseModel):
    """Type of a function parameter or return value"""
    type: str


class FunctionDefintion(BaseModel):
    """A single function the model can choose to call"""

    name: str
    description: str
    parameters: dict[str, TypeSpec]
    returns: TypeSpec


class FunctionsDefinition(RootModel[list[FunctionDefintion]]):
    """The full list of available functions, as loaded from
    functions_definition"""


class PromptEntry(BaseModel):
    """A single natural-language prompt to process"""

    prompt: str


class PromptEntries(RootModel[list[PromptEntry]]):
    """The full list of prompts, as loaded from
    function_calling_tests.json"""


class FunctionCallResult(BaseModel):
    """A single result: which function was called with
    what arguments, for a given prompt """

    prompt: str
    name: str
    parameters: dict[str, object]
