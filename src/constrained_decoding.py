from llm_sdk.llm_sdk import Small_LLM_Model
from pydantic import BaseModel
from typing import List, Dict


class TokenTrie(BaseModel):
    token_id: int = -1
    children: Dict[int, "TokenTrie"] = {}

    def add_path(self, tokens: List[int]) -> None:
        node = self
        for token in tokens:
            node.children.setdefault(token, TokenTrie(token_id=token))
            node = node.children[token]


class TokenGrammar(BaseModel):
    pass


class ConstrainedDecoder(BaseModel):
    model_config = {"arbitrary_types_allowed": True}
    llm: Small_LLM_Model

    def build_trie(self, strings: List[str], end_sequence: str) -> TokenTrie:
        root = TokenTrie()
        end_sequence_tokens = self.llm.encode(end_sequence).tolist()[0]

        for s in strings:
            tokens: List[int] = self.llm.encode(s).tolist()[0]
            tokens.extend(end_sequence_tokens)
            root.add_path(tokens=tokens)

        return root

    def generate_from_trie(
            self,
            prompt: str,
            answers: List[str],
            end_sequence: str
    ) -> str:

        root = self.build_trie(
            strings=answers,
            end_sequence=end_sequence
        )

        prompt_tokens: List[int] = self.llm.encode(prompt).tolist()[0]

        generated: List[int] = []
        node = root

        while node.children:
            logits = self.llm.get_logits_from_input_ids(prompt_tokens)

            valid_tokens = list(node.children.keys())
            best_token = max(valid_tokens, key=lambda t: logits[t])

            prompt_tokens.append(best_token)
            generated.append(best_token)
            node = node.children[best_token]

        answer = self.llm.decode(generated).removesuffix(end_sequence)

        return answer
