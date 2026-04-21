import json
import atexit
import signal
import os
import argparse

from typing import List, Dict, Tuple

from llm_sdk import Small_LLM_Model

from .validation import check_data, DataError
from .spinner import Spinner
from .extractor import Extractor
from .structures import Functions


try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def hide_cursor() -> None:
    """Hide the terminal cursor."""
    print("\033[?25l", end="", flush=True)


def show_cursor() -> None:
    """Show the terminal cursor."""
    print("\033[?25h", end="", flush=True)


def parse_argv() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Function calling with constrained decoding."
    )
    parser.add_argument(
        "--functions_definition",
        type=str,
        default="data/input/functions_definition.json",
        help="Path to function definitions JSON file."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/input/function_calling_tests.json",
        help="Path to input prompts JSON file."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/output/function_calls.json",
        help="Path to output results JSON file."
    )
    parser.add_argument(
        "--grammar",
        type=str,
        default="data/grammar/grammar.json",
        help="Path to grammar definitions JSON file."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-0.6B",
        help="Model name to use for inference."
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Print generation details in real time."
    )

    return parser.parse_args()


def load_data(
    grammar_path: str,
    functions_path: str,
    prompts_path: str
) -> Tuple[Dict[str, Dict], List[Dict], List[Dict]]:
    """Load and parse a JSON file.

    Args:
        path: Path to the JSON file.

    Returns:
        Parsed JSON data, or None if loading failed.
    """
    with open(grammar_path) as f:
        grammar_data = json.load(f)

    with open(functions_path) as f:
        functions_data = json.load(f)

    with open(prompts_path) as f:
        prompts_data = json.load(f)

    return grammar_data, functions_data, prompts_data


def write_data(
    result: List[Dict],
    output_path: str
) -> None:
    """Write results to a JSON file.

    Args:
        result: List of result dictionaries.
        output_path: Path to the output file.
    """
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults written to {output_path}")


def main(spinner: Spinner) -> None:
    """Main entrypoint for the function calling pipeline.

    Args:
        spinner: Spinner instance for progress display.
    """
    args = parse_argv()

    grammar_data, functions_data, prompts_data = load_data(
        args.grammar, args.functions_definition, args.input
    )

    error = check_data(grammar_data, functions_data, prompts_data)
    if error:
        raise DataError(error)

    functions = Functions(functions=functions_data)

    llm = Small_LLM_Model(model_name=args.model)

    extractor = Extractor(
        llm=llm,
        grammar=grammar_data,
        spinner=spinner,
        debug_mode=args.debug
    )

    result: List[Dict] = []

    spinner.total_prompts = len(prompts_data)
    spinner.update_message()
    print("\nGenerating...")
    spinner.start()

    for user_prompt in prompts_data:
        answer = extractor.generate_answer(
            user_prompt=user_prompt["prompt"],
            functions=functions
        )
        result.append(answer)
        spinner.update_message()

    spinner.generated = True
    spinner.stop()

    write_data(result=result, output_path=args.output)


if __name__ == "__main__":
    try:
        spinner = Spinner()

        def cleanup() -> None:
            spinner.stop()
            show_cursor()

        atexit.register(cleanup)
        signal.signal(signal.SIGINT, lambda *_: (cleanup(), os._exit(0)))

        hide_cursor()
        main(spinner=spinner)

    except Exception as e:
        print(f"Error: {e}")
