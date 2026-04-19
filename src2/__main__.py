import json
from typing import List, Dict, Tuple
from llm_sdk.llm_sdk import Small_LLM_Model

from structures import Functions
from extractor import Extractor


def parse_argv() -> Tuple[str, str, str]:
    return "Qwen/Qwen3-0.6B", "/home/vquetier/cc/m3/callmm/grammar.json", "/home/vquetier/cc/m3/callmm/data/input/functions_definition.json", "/home/vquetier/cc/m3/callmm/data/input/function_calling_tests.json"


def load_data(
    grammar_file: str,
    functions_file: str,
    prompts_file: str
) -> Tuple[Dict, Functions, List]:
    with open(grammar_file) as f:
        grammar = json.load(f)
    with open(functions_file) as f:
        functions = Functions(functions=json.load(f))
    with open(prompts_file) as f:
        prompts = json.load(f)

    return grammar, functions, prompts


def write_data(result: List[Dict]) -> None:
    pass


def main() -> None:
    model_name, grammar_file, functions_file, prompts_file = parse_argv()
    grammar, functions, prompts = load_data(
        grammar_file=grammar_file,
        functions_file=functions_file,
        prompts_file=prompts_file
    )

    llm = Small_LLM_Model(
        model_name=model_name
    )

    extractor = Extractor(
        llm=llm,
        grammar=grammar,
    )

    result: List[Dict] = []

    for user_prompt in prompts:
        answer = extractor.generate_answer(
            user_prompt=user_prompt,
            functions=functions
        )
        result.append(answer)

    write_data(result=result)


if __name__ == "__main__":
    main()
