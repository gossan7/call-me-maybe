from constrained_decoding import ConstrainedDecoder
from typing import Dict
import json


GRAMMAR_FILE = "C:\\Users\\queti\\Documents\\call-me-maybe\\grammar.json"
TYPES_NAMES = {
    "number": "float"
}


def build_prompt(
        user_prompt: str,
        function: Dict,
        extracted: Dict[str, str],
        current_param: str
) -> str:
    params = ", ".join(
        f"{k}: {TYPES_NAMES.get(v['type'], v['type'])}"
        for k, v in function["parameters"].items()
    )
    function_definition = f"Function({params}): {function['description']}"

    prompt = f"{function_definition}\nPrompt: {user_prompt}\n"
    prompt += "Extract parameters from the prompt.\n\n"
    elems = [f"{name} = {value}" for name, value in extracted.items()]
    elems.append(f"{current_param} = ")
    prompt += ", ".join(elems)

    return prompt


def extract_parameters(
        function: Dict,
        user_prompt: str,
        constrained_decoder: ConstrainedDecoder
) -> Dict[str, str]:
    extracted: Dict[str, str] = {}

    with open(GRAMMAR_FILE) as f:
        grammar = json.load(f)

    for parameter in function["parameters"]:
        prompt = build_prompt(
            user_prompt=user_prompt,
            function=function,
            extracted=extracted,
            current_param=parameter
        )

        end_set = [
            "\n",
            ","
        ]

        if function["parameters"][parameter]["type"] not in {"number", "integer", "bool"}:
            prompt += '\"'
            end_set = [
                '\"'
            ]

        parameter_grammar = grammar[function["parameters"][parameter]["type"]]

        answer = constrained_decoder.generate_from_grammar(
            prompt=prompt,
            grammar=parameter_grammar,
            end_set=end_set
        )

        extracted[parameter] = answer

    return extracted
