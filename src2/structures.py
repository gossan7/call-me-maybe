from typing import List, Dict, Set
from pydantic import BaseModel

from llm_sdk.llm_sdk import Small_LLM_Model


class Functions(BaseModel):
    functions: List[Dict]
    functions_mapping: Dict[str, Dict] = {}

    def model_post_init(self, _):
        self.functions_mapping = {
            func["name"]: func
            for func in self.functions
        }

    def get_parameters(self, name: str) -> Dict:
        return self.functions_mapping[name]["parameters"]

    def get_description(self, name: str) -> str:
        return self.functions_mapping[name]["description"]

    def get_returns(self, name: str) -> str:
        return self.functions_mapping[name]["returns"]["type"]

    def get_names_list(self) -> List[str]:
        return list(self.functions_mapping.keys())


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
    grammar: Dict
    end_set: List[str]
    llm: Small_LLM_Model
    state: str = "start"
    token_sets: Dict[str, Set[int]] = {}
    end_set_tokens: Set[int] = set()

    def model_post_init(self, _) -> None:
        self.state: str = "start"

        self.token_sets = {
            curr_set: {
                token for elem in self.grammar["token_sets"][curr_set]
                for token in self.llm.encode(elem).tolist()[0]
            }
            for curr_set in self.grammar["token_sets"]
        }

        self.end_set_tokens = {
            token for seq in self.end_set
            for token in self.llm.encode(seq).tolist()[0]
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

    def can_end(self):
        return self.grammar["states"][self.state]["can_end"]

    def get_available_tokens(self) -> List[int] | None:
        token_set = set()

        for set_name, set_content in self.token_sets.items():
            if set_name == "any":
                return None
            if set_name in self.grammar["states"][self.state]["sets"]:
                token_set |= set_content

        token_list = list(token_set)

        if self.can_end():
            token_list.extend(list(self.end_set_tokens))

        return token_list
