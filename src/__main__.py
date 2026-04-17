from function_selector import select_function
from parameter_extractor import extract_parameters
from sys import argv
from typing import Dict, List
from llm_sdk.llm_sdk import Small_LLM_Model
from constrained_decoding import ConstrainedDecoder
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
        "C:\\Users\\queti\\Documents\\call-me-maybe\\data\\input\\function_calling_tests.json"
    )

    prompts = [value for d in prompts_list for value in d.values()]

    functions = get_data(
        "C:\\Users\\queti\\Documents\\call-me-maybe\\data\\input\\functions_definition.json"
    )

    constrained_decoder = ConstrainedDecoder(
        llm=Small_LLM_Model()
    )

    # TODO: main loop
    print("\n===== Generation =====\n")
    total = 0.0
    for user_prompt in prompts:
        start = time.time()

        function = select_function(
            functions=functions,
            user_prompt=user_prompt,
            constrained_decoder=constrained_decoder
        )

        for elem in functions:
            if elem["name"] == function:
                function_dict = elem
                break

        parameters = extract_parameters(
            function=function_dict,
            user_prompt=user_prompt,
            constrained_decoder=constrained_decoder
        )

        end = time.time()

        print(user_prompt, function, parameters, f"time: {end - start}", sep='\n')

        print()

        total += end - start

        # TODO: format answer in a dict

    print(f"total time: {total}, prompt: {len(prompts)}")

    # TODO: write answers in a json


if __name__ == "__main__":
    # try:
    main()
    # except Exception as e:
    #     print(f"Unexpected exception: {e}")
