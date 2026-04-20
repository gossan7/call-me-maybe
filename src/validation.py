"""Validation utilities for input data structures."""

from typing import Optional, Any


class DataError(Exception):
    def __init__(self, msg: str = "Not specified"):
        super().__init__(f"DataError: {msg}")


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
        if not isinstance(type_def, dict):
            return f"Grammar type '{type_name}' must be an object."

        for key in ("token_sets", "start", "states"):
            if key not in type_def:
                return f"Grammar type '{type_name}' missing '{key}'."

        if not isinstance(type_def["token_sets"], dict):
            return f"Grammar type '{type_name}': 'token_sets' must be an object."

        if not isinstance(type_def["states"], dict):
            return f"Grammar type '{type_name}': 'states' must be an object."

        start = type_def["start"]
        if start not in type_def["states"]:
            return f"Grammar type '{type_name}': start state '{start}' not found in states."

        for state_name, state_def in type_def["states"].items():
            if not isinstance(state_def, dict):
                return f"Grammar type '{type_name}': state '{state_name}' must be an object."

            for key in ("sets", "can_end", "transitions"):
                if key not in state_def:
                    return f"Grammar type '{type_name}': state '{state_name}' missing '{key}'."

            if not isinstance(state_def["sets"], list):
                return f"Grammar type '{type_name}': state '{state_name}': 'sets' must be a list."

            if not isinstance(state_def["can_end"], bool):
                return f"Grammar type '{type_name}': state '{state_name}': 'can_end' must be a boolean."

            if not isinstance(state_def["transitions"], dict):
                return f"Grammar type '{type_name}': state '{state_name}': 'transitions' must be an object."

            for set_name in state_def["sets"]:
                if set_name not in type_def["token_sets"]:
                    return f"Grammar type '{type_name}': state '{state_name}' references unknown set '{set_name}'."

            for trigger, target in state_def["transitions"].items():
                if trigger not in type_def["token_sets"]:
                    return f"Grammar type '{type_name}': state '{state_name}' transition references unknown set '{trigger}'."
                if target not in type_def["states"]:
                    return f"Grammar type '{type_name}': state '{state_name}' transition targets unknown state '{target}'."

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

    if len(functions) == 0:
        return "Functions definition must contain at least one function."

    for i, func in enumerate(functions):
        if not isinstance(func, dict):
            return f"Function at index {i} must be an object."

        for key in ("name", "description", "parameters", "returns"):
            if key not in func:
                return f"Function at index {i} missing '{key}'."

        if not isinstance(func["name"], str):
            return f"Function at index {i}: 'name' must be a string."

        if not isinstance(func["description"], str):
            return f"Function at index {i}: 'description' must be a string."

        if not isinstance(func["parameters"], dict):
            return f"Function '{
                func['name']}': 'parameters' must be an object."

        if not isinstance(func["returns"], dict):
            return f"Function '{func['name']}': 'returns' must be an object."

        if "type" not in func["returns"]:
            return f"Function '{func['name']}': 'returns' missing 'type'."

        for param_name, param_def in func["parameters"].items():
            if not isinstance(param_def, dict):
                return f"Function '{
                    func['name']}': parameter '{param_name}' must be an object."
            if "type" not in param_def:
                return f"Function '{
                    func['name']}': parameter '{param_name}' missing 'type'."

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

    if len(prompts) == 0:
        return "Prompts must contain at least one prompt."

    for i, prompt in enumerate(prompts):
        if not isinstance(prompt, dict):
            return f"Prompt at index {i} must be an object."

        if "prompt" not in prompt:
            return f"Prompt at index {i} missing 'prompt' key."

        if not isinstance(prompt["prompt"], str):
            return f"Prompt at index {i}: 'prompt' must be a string."

    return None


def check_data(
    grammar: Any,
    functions: Any,
    prompts: Any
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
