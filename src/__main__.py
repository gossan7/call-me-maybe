from function_selector import select_function
from sys import argv
from typing import Dict, List
from llm_sdk.llm_sdk import Small_LLM_Model
import json
import time


EXPECTED = {
    "What is the sum of 2 and 3?": "fn_add_numbers",
    "What is the sum of 265 and 345?": "fn_add_numbers",
    "Add 99 and 1": "fn_add_numbers",
    "Greet shrek": "fn_greet",
    "Greet john": "fn_greet",
    "Say hello to Alice": "fn_greet",
    "Reverse the string 'hello'": "fn_reverse_string",
    "Reverse the string 'world'": "fn_reverse_string",
    "Reverse 'abcdef'": "fn_reverse_string",
    "What is the square root of 16?": "fn_get_square_root",
    "Calculate the square root of 144": "fn_get_square_root",
    "Square root of 81": "fn_get_square_root",
    "Replace all numbers in \"Hello 34 I'm 233 years old\" with NUMBERS": "fn_substitute_string_with_regex",
    "Replace all vowels in 'Programming is fun' with asterisks": "fn_substitute_string_with_regex",
    "Substitute the word 'cat' with 'dog' in 'The cat sat on the mat with another cat'": "fn_substitute_string_with_regex",
    "Multiply 7 and 8": "fn_multiply_numbers",
    "What is 12 times 15?": "fn_multiply_numbers",
    "Calculate the product of 3 and 99": "fn_multiply_numbers",
    "Divide 100 by 4": "fn_divide_numbers",
    "What is 999 divided by 3?": "fn_divide_numbers",
    "Divide 50 by 7": "fn_divide_numbers",
    "How long is the string 'hello world'?": "fn_string_length",
    "What is the length of 'pneumonoultramicroscopicsilicovolcanoconiosis'?": "fn_string_length",
    "Count the characters in 'test'": "fn_string_length",
    "Convert 'hello world' to uppercase": "fn_to_uppercase",
    "Make 'Python is Great' all uppercase": "fn_to_uppercase",
    "Uppercase the string 'abc'": "fn_to_uppercase",
    "Convert 'HELLO WORLD' to lowercase": "fn_to_lowercase",
    "Make 'SCREAMING' lowercase": "fn_to_lowercase",
    "Lowercase 'ABC'": "fn_to_lowercase",
    "Is 'racecar' a palindrome?": "fn_is_palindrome",
    "Check if 'hello' is a palindrome": "fn_is_palindrome",
    "Is 'madam' a palindrome?": "fn_is_palindrome",
    "Raise 2 to the power of 10": "fn_power",
    "What is 3 to the power 4?": "fn_power",
    "Calculate 5 raised to the exponent 3": "fn_power",
    "Concatenate 'hello' and ' world'": "fn_concat_strings",
    "Join the strings 'foo' and 'bar'": "fn_concat_strings",
    "Put together 'good' and 'bye'": "fn_concat_strings",
    "What is the absolute value of -42?": "fn_absolute_value",
    "Absolute value of -7.5": "fn_absolute_value",
    "Get the absolute value of -100": "fn_absolute_value",
    "Is 42 even?": "fn_is_even",
    "Check if 7 is an even number": "fn_is_even",
    "Is 128 even?": "fn_is_even",
}


def get_data(filename):
    with open(filename) as f:
        data = json.load(f)
    return data


def main() -> None:
    # TODO: parse flags

    # TODO: verify json validity
    prompts_list: List[Dict[str, str]] = get_data(
        "/home/vquetier/cc/m3/callmm/data/input/function_calling_tests3.json"
    )

    prompts = [value for d in prompts_list for value in d.values()]

    data = get_data(
        "/home/vquetier/cc/m3/callmm/data/input/functions_definition2.json"
    )

    llm = Small_LLM_Model()

    # TODO: main loop
    total = 0.0
    print("\n===== Generation =====\n")
    valid = 0
    for p in prompts:
        start = time.time()
        function = select_function(data, p, llm)
        end = time.time()

        print(p, function, f"time: {end - start}", sep='\n')
        if function == EXPECTED[p]:
            print("Valid")
            valid += 1

        else:
            print("Invalid")

        print()
        total += end - start
        # TODO: extract parameters from function
        # TODO: format answer as json
    print(f"total score: {valid}/{len(prompts)}")
    print(f"total time: {total}")

    # TODO: write answers in a json


if __name__ == "__main__":
    # try:
    main()
    # except Exception as e:
    #     print(f"Unexpected exception: {e}")
