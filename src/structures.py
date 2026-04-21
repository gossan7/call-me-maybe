from typing import List, Dict, Set, Any
from pydantic import BaseModel

from llm_sdk import Small_LLM_Model

from .tokenizer import Tokenizer


class Functions(BaseModel):
    functions: List[Dict[Any, Any]]
    functions_mapping: Dict[str, Dict[Any, Any]] = {}

    def model_post_init(self, _: Any) -> None:
        self.functions_mapping = {
            func["name"]: func
            for func in self.functions
        }

    def get_parameters(self, name: str) -> Dict[str, Dict[str, str]]:
        parameters = self.functions_mapping[name]["parameters"]
        assert isinstance(parameters, dict)
        return parameters

    def get_description(self, name: str) -> str:
        description = self.functions_mapping[name]["description"]
        assert isinstance(description, str)
        return description

    def get_returns(self, name: str) -> str:
        returns = self.functions_mapping[name]["returns"]["type"]
        assert isinstance(returns, str)
        return returns

    def get_names_list(self) -> List[str]:
        names_list = list(self.functions_mapping.keys())
        assert isinstance(names_list, list)
        return names_list


class TokenTrie(BaseModel):
    token_id: int = -1
    children: Dict[int, "TokenTrie"] = {}
    path_count: int = 0

    def add_path(
        self,
        tokens: List[int],
        end_sequences_tokens: List[List[int]]
    ) -> None:

        self.path_count += 1

        node = self
        for token in tokens:
            node.children.setdefault(token, TokenTrie(token_id=token))
            node = node.children[token]
            node.path_count += 1

        end = node
        for end_tokens in end_sequences_tokens:
            node = end
            for token in end_tokens:
                node.children.setdefault(token, TokenTrie(token_id=token))
                node = node.children[token]
                node.path_count += 1


class TokenGrammar(BaseModel):
    model_config = {"arbitrary_types_allowed": True}
    grammar: Dict[str, Any]
    end_set: List[str]
    llm: Small_LLM_Model
    tokenizer: Tokenizer
    state: str = "start"
    token_sets: Dict[str, Set[int]] = {}
    end_set_tokens: Set[int] = set()

    def model_post_init(self, _: Any) -> None:
        self.state: str = "start"

        self.token_sets = {
            curr_set: {
                token for elem in self.grammar["token_sets"][curr_set]
                for token in self.tokenizer.encode(elem)
            }
            for curr_set in self.grammar["token_sets"]
        }

        self.end_set_tokens = {
            token for seq in self.end_set
            for token in self.tokenizer.encode(seq)
        }

    def update_state(self, token: int) -> None:
        if token in self.end_set_tokens:
            return

        for set_name, set_content in self.token_sets.items():
            if token in set_content:
                token_set = set_name
                break

        self.state = \
            self.grammar["states"][self.state]["transitions"][token_set]

    def can_end(self) -> bool:
        can_end_flag = self.grammar["states"][self.state]["can_end"]
        assert isinstance(can_end_flag, bool)
        return can_end_flag

    def get_available_tokens(self) -> List[int]:
        token_set = set()

        for set_name, set_content in self.token_sets.items():
            if set_name in self.grammar["states"][self.state]["sets"]:
                token_set |= set_content

        token_list = list(token_set)

        if self.can_end():
            token_list.extend(list(self.end_set_tokens))

        return token_list
