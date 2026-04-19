import json
import sys

from typing import List, Dict, Tuple
from dotenv import load_dotenv

from llm_sdk.llm_sdk import Small_LLM_Model

from spinner import Spinner
from extractor import Extractor
from structures import Functions

load_dotenv()


def hide_cursor() -> None:
    print("\033[?25l", end="", flush=True)


def show_cursor() -> None:
    print("\033[?25h", end="", flush=True)


def parse_argv() -> Tuple[str, str, str, str]:
    return "Qwen/Qwen3-0.6B", "/home/vquetier/cc/m3/callmm/grammar.json", "/home/vquetier/cc/m3/callmm/data/input/functions_definition2.json", "/home/vquetier/cc/m3/callmm/data/input/function_calling_tests2.json"


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
    hide_cursor()

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

    print("\nGenerating...")

    spinner = Spinner(total_prompts=len(prompts))
    spinner.start()

    for user_prompt in prompts:
        spinner.update_message()
        answer = extractor.generate_answer(
            user_prompt=user_prompt["prompt"],
            functions=functions
        )
        result.append(answer)
    spinner.stop()
    print()

    write_data(result=result)


if __name__ == "__main__":
    try:
        hide_cursor()
        main()
    except Exception as e:
        print(e)
    finally:
        show_cursor()
