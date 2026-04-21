from typing import Dict
from pydantic import BaseModel
from llm_sdk import Small_LLM_Model

from structures import Functions
from spinner import Spinner
from decoders import Decoder, TrieDecoder, GrammarDecoder, FreeDecoder


class Extractor(BaseModel):
    model_config = {"arbitrary_types_allowed": True}
    llm: Small_LLM_Model
    grammar: Dict
    spinner: Spinner
    debug_mode: bool = False

    def debug(self, text: str) -> None:
        """Print debug information above the spinner.

        Args:
            text: Debug text to display.
        """
        if self.debug_mode:
            self.spinner.print_above(text)

    def build_prompt_for_function(
        self,
        user_prompt: str,
        functions: Functions
    ) -> str:
        """Build the augmented prompt for function selection.

        Args:
            user_prompt: The user's natural language prompt.
            functions: Available functions definitions.

        Returns:
            Formatted prompt string for function selection.
        """
        prompt_list = []

        for func_name in functions.get_names_list():
            description = functions.get_description(name=func_name)
            prompt_list.append(
                f"{func_name}: {description}"
            )

        prompt_list.append(
            "Select the appropriate function for the prompt."
        )
        prompt_list.append(f"Prompt: {user_prompt}")
        prompt_list.append("Function: \"")

        return '\n'.join(prompt_list)

    def get_function(
        self,
        user_prompt: str,
        functions: Functions,
        answer: Dict
    ) -> None:

        decoder = TrieDecoder(
            llm=self.llm,
            end_sequences=['\"', ',', '\n'],
            sequences=functions.get_names_list()
        )

        prompt = self.build_prompt_for_function(
            user_prompt=user_prompt,
            functions=functions
        )

        answer["name"] = decoder.generate(prompt=prompt)

        params = functions.get_parameters(answer["name"])

        params_string = ", ".join(
            f"{k}: {v['type']}"
            for k, v in params.items()
        )

        self.debug(f"  Function: {answer['name']}({params_string})")

    def get_decoder_for_parameter(self, parameter_type: str) -> Decoder:
        match parameter_type:
            case "number" | "integer":
                return GrammarDecoder(
                    llm=self.llm,
                    end_sequences=['\"', ',', '\n'],
                    grammar=self.grammar[parameter_type]
                )
            case "boolean":
                return TrieDecoder(
                    llm=self.llm,
                    end_sequences=['\"', ',', '\n'],
                    sequences=["true", "false"]
                )
            case _:
                return FreeDecoder(
                    llm=self.llm,
                    end_sequences=['\"', '\n']
                )

    def build_prompt_for_parameter(
        self,
        user_prompt: str,
        functions: Functions,
        parameter: str,
        answer: Dict
    ) -> str:
        """Build the augmented prompt for parameter extraction.

        Args:
            user_prompt: The user's natural language prompt.
            functions: Available functions definitions.
            parameter: Name of the parameter to extract.
            answer: Current answer dict with function name
                and already extracted parameters.

        Returns:
            Formatted prompt string for parameter extraction.
        """
        func_name = answer["name"]
        func_params = functions.get_parameters(name=func_name)

        type_mapping = {
            "number": "float",
            "integer": "int",
            "boolean": "bool",
            "string": "str"
        }

        signature = ", ".join(
            f"{k}: {type_mapping.get(v['type'], v['type'])}"
            for k, v in func_params.items()
        )

        description = functions.get_description(name=func_name)

        prompt_list = [
            f"{func_name}({signature}): {description}",
            "Extract the parameter values from the prompt.",
            f"Prompt: {user_prompt}",
        ]

        extracted = [
            f"{k} = {v}" for k, v in answer["parameters"].items()
        ]
        extracted.append(f"{parameter} = ")

        prompt_list.append("Parameters: " + ", ".join(extracted))

        param_type = func_params[parameter]["type"]
        if param_type not in {"number", "integer", "boolean"}:
            prompt_list[-1] += '\"'

        return '\n'.join(prompt_list)

    def convert_parameter(
        self,
        result: str,
        parameter_type: str
    ) -> int | float | bool | str:
        match parameter_type:
            case "integer":
                return int(result)
            case "number":
                return float(result)
            case "boolean":
                return True if result == "true" else False
            case _:
                return result

    def get_parameters(
        self,
        user_prompt: str,
        functions: Functions,
        answer: Dict
    ) -> None:

        answer["parameters"] = {}

        for name, value in \
                functions.get_parameters(name=answer["name"]).items():
            decoder = self.get_decoder_for_parameter(
                parameter_type=value["type"]
            )

            prompt = self.build_prompt_for_parameter(
                user_prompt=user_prompt,
                functions=functions,
                parameter=name,
                answer=answer
            )

            result = decoder.generate(prompt=prompt)

            answer["parameters"][name] = self.convert_parameter(
                result=result,
                parameter_type=value["type"]
            )

            self.debug(f"    {name} = {answer['parameters'][name]}")

    def generate_answer(self, user_prompt: str, functions: Functions) -> Dict:
        answer = {
            "prompt": user_prompt,
        }

        self.debug(f"\n\033[36mPrompt:\033[0m {user_prompt}")

        self.get_function(
            user_prompt=user_prompt,
            functions=functions,
            answer=answer
        )

        self.get_parameters(
            user_prompt=user_prompt,
            functions=functions,
            answer=answer
        )

        return answer
