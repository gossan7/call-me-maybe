from typing import Dict
from pydantic import BaseModel
from llm_sdk.llm_sdk import Small_LLM_Model

from structures import Functions
from decoders import Decoder, TrieDecoder, GrammarDecoder, FreeDecoder


class Extractor(BaseModel):
    model_config = {"arbitrary_types_allowed": True}
    llm: Small_LLM_Model
    grammar: Dict

    def build_prompt_for_function(self, user_prompt: str, functions: Functions) -> str:
        prompt_list = []

        for name in functions.get_names_list():
            parameters = ", ".join(
                f"k: {v['type']}"
                for k, v in functions.get_parameters(name=name).items()
            )

            description = functions.get_description(name=name)

            returns = functions.get_returns(name=name)["type"]

            prompt_list.append(
                f"{name} ({parameters}) -> {returns}: {description}"
            )

        prompt_list.append("Select the appropriate function for the prompt.")
        prompt_list.append(f"Prompt: {user_prompt}")
        prompt_list.append("Function: \"")

        return '\n'.join(prompt_list)

    def get_function(self, user_prompt: str, functions: Functions, answer: Dict) -> None:
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

    def build_prompt_for_parameter(self, functions: Functions, parameter: str, answer: Dict) -> str:
        prompt_list = []

        name = answer["name"]

        parameters = ", ".join(
            f"k: {v['type']}"
            for k, v in functions.get_parameters(name=name).items()
        )

        description = functions.get_description(name=name)

        returns = functions.get_returns(name=name)["type"]

        prompt_list.append(
            f"{name} ({parameters}) -> {returns}: {description}"
        )

        params = [f"{name} = {value}" for name, value in answer["parameters"]]
        params.append(f"{parameter} = ")

        prompt_list.append(", ".join(params))

        if functions.get_parameters(name=name)[parameter]["type"] not in {
            "number",
            "integer",
            "boolean"
        }:
            prompt_list[-1] += '\"'

        return '\n'.join(prompt_list)

    def convert_parameter(self, result: str, parameter_type: str) -> int | float | bool | str:
        match parameter_type:
            case "integer":
                return int(result)
            case "number":
                return float(result)
            case "boolean":
                return True if result == "true" else False
            case _:
                return result

    def get_parameters(self, user_prompt: str, functions: Functions, answer: Dict) -> None:
        answer["parameters"] = {}
        for parameter in functions.get_parameters(name=answer["name"]):
            decoder = self.get_decoder_for_parameter(parameter_type=parameter["type"])

            prompt = self.build_prompt_for_parameter(
                functions=functions,
                answer=answer
            )

            result = decoder.generate(prompt=prompt)

            answer["parameters"][parameter["name"]] = self.convert_parameter(
                result=result,
                parameter_type=parameter["type"]
            )

    def generate_answer(self, user_prompt: str, functions: Functions) -> Dict:
        answer = {
            "prompt": user_prompt,
        }

        self.get_function(
            prompt=user_prompt,
            functions=functions,
            answer=answer
        )

        self.get_parameters(
            prompt=user_prompt,
            functions=functions,
            answer=answer
        )

        return answer
