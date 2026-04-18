from typing import List, Dict, Tuple
from llm_sdk.llm_sdk import Small_LLM_Model

from extractor import Extractor


def parse_argv() -> Tuple[str, str, str]:
    pass


def load_data(
    grammar_file: str,
    functions_file: str,
    prompts_file: str
) -> Tuple[grammar, functions, prompts]:
    pass


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

    for prompt in prompts:
        answer = extractor.generate_answer(
            prompt=prompt,
            functions=functions
        )
        result.append(answer)

    write_data(result=result)
