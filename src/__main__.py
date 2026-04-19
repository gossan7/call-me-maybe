import json
import sys
from typing import List, Dict, Tuple
from llm_sdk.llm_sdk import Small_LLM_Model

from structures import Functions
from extractor import Extractor


def parse_argv() -> Tuple[str, str, str, str]:
    return "Qwen/Qwen3-0.6B", "/home/vquetier/cc/m3/callmm/grammar.json", "/home/vquetier/cc/m3/callmm/data/input/functions_definition2.json", "/home/vquetier/cc/m3/callmm/data/input/function_calling_tests4.json"


def load_data(
    grammar_file: str,
    functions_file: str,
    prompts_file: str
) -> Tuple[Dict, Functions, List[Dict]]:
    with open(grammar_file) as f:
        grammar = json.load(f)
    with open(functions_file) as f:
        functions = Functions(functions=json.load(f))
    with open(prompts_file) as f:
        prompts = json.load(f)

    return grammar, functions, prompts


def write_data(result: List[Dict]) -> None:
    with open("results.json", "w") as f:
        json.dump(result, f, indent=2)


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

    i = 0

    for user_prompt in prompts:
        i += 1
        print(f"{i}/{len(prompts)}:", user_prompt["prompt"])
        answer = extractor.generate_answer(
            user_prompt=user_prompt["prompt"],
            functions=functions
        )
        json.dump(obj=answer, fp=sys.stdout, indent=2)
        print('\n')
        result.append(answer)

    write_data(result=result)


if __name__ == "__main__":
    main()
