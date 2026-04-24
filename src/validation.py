"""Input validation for grammar, function, and prompt files.

Scope: **structure only**. These checks confirm that each file is shaped the
way the rest of the code expects — correct JSON types at each position,
required keys present — nothing more. Whether a parameter type string names
a real type, whether a function name is a valid identifier, whether the
grammar covers every function's types, whether a state is reachable: all of
that is left to the consumer to handle.

The three ``check_*`` functions return ``None`` on success or a human-readable
error string on failure. ``check_data`` runs all three.
"""

from typing import Any, Optional


class DataError(Exception):
    """Raised when input data is missing, unreadable, or malformed."""

    def __init__(self, msg: str = "Not specified") -> None:
        """Initialize DataError with a message.

        Args:
            msg: Error description.
        """
        super().__init__(f"DataError: {msg}")


def _check_token_sets(
    type_name: str,
    token_sets: Any,
) -> Optional[str]:
    """Validate the shape of a token_sets block."""
    prefix = f"Grammar type '{type_name}': 'token_sets'"

    if not isinstance(token_sets, dict):
        return f"{prefix} must be an object."

    for set_name, members in token_sets.items():
        if not isinstance(set_name, str):
            return f"{prefix} has a non-string set name."
        loc = f"{prefix}['{set_name}']"
        if not isinstance(members, list):
            return f"{loc} must be a list, got {type(members).__name__}."
        for elem in members:
            if not isinstance(elem, str):
                return (
                    f"{loc} must contain only strings, found "
                    f"{type(elem).__name__}."
                )

    return None


def _check_state(
    type_name: str,
    state_name: str,
    state_def: Any,
) -> Optional[str]:
    """Validate the shape of a single state definition."""
    prefix = f"Grammar type '{type_name}': state '{state_name}'"

    if not isinstance(state_name, str):
        return (
            f"Grammar type '{type_name}': state name must be a string."
        )
    if not isinstance(state_def, dict):
        return f"{prefix} must be an object."

    for key in ("sets", "can_end", "transitions"):
        if key not in state_def:
            return f"{prefix} missing '{key}'."

    sets = state_def["sets"]
    can_end = state_def["can_end"]
    transitions = state_def["transitions"]

    if not isinstance(sets, list):
        return f"{prefix}: 'sets' must be a list."
    for set_name in sets:
        if not isinstance(set_name, str):
            return f"{prefix}: 'sets' entries must be strings."

    if not isinstance(can_end, bool):
        return f"{prefix}: 'can_end' must be a boolean."

    if not isinstance(transitions, dict):
        return f"{prefix}: 'transitions' must be an object."
    for trigger, target in transitions.items():
        if not isinstance(trigger, str):
            return f"{prefix}: transition key must be a string."
        if not isinstance(target, str):
            return f"{prefix}: transition target must be a string."

    return None


def _check_type(type_name: str, type_def: Any) -> Optional[str]:
    """Validate the shape of a single grammar type definition."""
    if not isinstance(type_name, str):
        return "Grammar type name must be a string."
    if not isinstance(type_def, dict):
        return f"Grammar type '{type_name}' must be an object."

    for key in ("token_sets", "start", "states"):
        if key not in type_def:
            return f"Grammar type '{type_name}' missing '{key}'."

    error = _check_token_sets(type_name, type_def["token_sets"])
    if error:
        return error

    if not isinstance(type_def["start"], str):
        return f"Grammar type '{type_name}': 'start' must be a string."

    states = type_def["states"]
    if not isinstance(states, dict):
        return f"Grammar type '{type_name}': 'states' must be an object."

    for state_name, state_def in states.items():
        error = _check_state(type_name, state_name, state_def)
        if error:
            return error

    return None


def check_grammar(grammar: Any) -> Optional[str]:
    """Validate grammar JSON structure.

    Args:
        grammar: Parsed grammar data.

    Returns:
        Error message if invalid, None if valid.
    """
    if not isinstance(grammar, dict):
        return "Grammar must be a JSON object."

    for type_name, type_def in grammar.items():
        error = _check_type(type_name, type_def)
        if error:
            return error

    return None


def _check_function(func: Any, index: int) -> Optional[str]:
    """Validate the shape of a single function definition."""
    if not isinstance(func, dict):
        return f"Function at index {index} must be an object."

    for key in ("name", "description", "parameters", "returns"):
        if key not in func:
            return f"Function at index {index} missing '{key}'."

    if not isinstance(func["name"], str):
        return f"Function at index {index}: 'name' must be a string."

    name = func["name"]

    if not isinstance(func["description"], str):
        return f"Function '{name}': 'description' must be a string."

    parameters = func["parameters"]
    if not isinstance(parameters, dict):
        return f"Function '{name}': 'parameters' must be an object."
    for param_name, param_def in parameters.items():
        if not isinstance(param_name, str):
            return f"Function '{name}': parameter name must be a string."
        if not isinstance(param_def, dict):
            return (
                f"Function '{name}': parameter '{param_name}' "
                "must be an object."
            )
        if "type" not in param_def:
            return (
                f"Function '{name}': parameter '{param_name}' "
                "missing 'type'."
            )

    returns = func["returns"]
    if not isinstance(returns, dict):
        return f"Function '{name}': 'returns' must be an object."
    if "type" not in returns:
        return f"Function '{name}': 'returns' missing 'type'."

    return None


def check_functions(functions: Any) -> Optional[str]:
    """Validate functions definition JSON structure.

    Args:
        functions: Parsed functions data.

    Returns:
        Error message if invalid, None if valid.
    """
    if not isinstance(functions, list):
        return "Functions definition must be a JSON array."

    for i, func in enumerate(functions):
        error = _check_function(func, i)
        if error:
            return error

    return None


def check_prompts(prompts: Any) -> Optional[str]:
    """Validate prompts JSON structure.

    Args:
        prompts: Parsed prompts data.

    Returns:
        Error message if invalid, None if valid.
    """
    if not isinstance(prompts, list):
        return "Prompts must be a JSON array."

    for i, entry in enumerate(prompts):
        if not isinstance(entry, dict):
            return f"Prompt at index {i} must be an object."
        if "prompt" not in entry:
            return f"Prompt at index {i} missing 'prompt' key."
        if not isinstance(entry["prompt"], str):
            return f"Prompt at index {i}: 'prompt' must be a string."

    return None


def check_data(
    grammar: Any,
    functions: Any,
    prompts: Any,
) -> Optional[str]:
    """Validate all input data structures.

    Args:
        grammar: Parsed grammar data.
        functions: Parsed functions data.
        prompts: Parsed prompts data.

    Returns:
        Error message if any input is invalid, None if all valid.
    """
    error = check_grammar(grammar)
    if error:
        return f"Invalid grammar: {error}"

    error = check_functions(functions)
    if error:
        return f"Invalid functions: {error}"

    error = check_prompts(prompts)
    if error:
        return f"Invalid prompts: {error}"

    return None
