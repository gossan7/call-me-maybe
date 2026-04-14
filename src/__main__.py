from function_selector2 import select_function
from sys import argv
from typing import Dict, List
from llm_sdk.llm_sdk import Small_LLM_Model
import json
import time


def get_data(filename):
    with open(filename) as f:
        data = json.load(f)
    return data


def main() -> None:
    # TODO: parse flags

    # TODO: verify json validity
    prompts_list: List[Dict[str, str]] = get_data(
        "/home/vquetier/cc/m3/callmm/data/input/function_calling_tests2.json"
    )

    prompts = [value for d in prompts_list for value in d.values()]

    data = get_data(
        "/home/vquetier/cc/m3/callmm/data/input/functions_definition2.json"
    )

    llm = Small_LLM_Model()

    # TODO: main loop
    total = 0.0
    print("\n===== Generation =====\n")
    for p in prompts:
        # TODO: select function name
        start = time.time()
        function = select_function(data, p, llm)
        end = time.time()
        print(p, function, f"time: {end - start}", sep='\n', end="\n\n")
        total += end - start
        # TODO: extract parameters from function
        # TODO: format answer as json
    print(f"total time: {total}")

    # TODO: write answers in a json


if __name__ == "__main__":
    # try:
    main()
    # except Exception as e:
    #     print(f"Unexpected exception: {e}")
