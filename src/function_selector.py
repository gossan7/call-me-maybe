from constrained_decoding import ConstrainedDecoder
from typing import Dict, List


def build_prompt(
    functions: List[Dict],
    user_prompt: str,
    label_map: Dict[str, str]
) -> str:

    prompt = ""

    for func in functions:
        params = ",".join(
            f"{v['type']}"
            for v in func["parameters"].values()
        )
        returns = func["returns"]["type"]
        label = label_map[func["name"]]
        prompt += f"F{label} ({params}) -> {returns}: {func['description']}\n"

    prompt += "Select the appropriate function for the prompt.\n"

    prompt += f"\nPrompt: {user_prompt}\n"
    prompt += "Function: \"F"

    return prompt


def select_function(
    functions: List[Dict],
    user_prompt: str,
    constrained_decoder: ConstrainedDecoder
) -> str:

    label_to_name: Dict[str, str] = {}
    name_to_label: Dict[str, str] = {}

    for i, func in enumerate(functions):
        label = f"{i}"
        name = func["name"]
        if name not in name_to_label:
            name_to_label[name] = label
            label_to_name[label] = name

    prompt = build_prompt(functions, user_prompt, name_to_label)
    answers = list(label_to_name.keys())

    answer = constrained_decoder.generate_from_trie(
        prompt=prompt,
        answers=answers,
        end_sequence="\""
    )

    return label_to_name[answer]
