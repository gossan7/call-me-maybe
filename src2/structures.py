from typing import List, Dict
from pydantic import BaseModel


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
    pass
