from typing import List, Dict
from abc import ABC, abstractmethod
from pydantic import BaseModel, Field
from llm_sdk.llm_sdk import Small_LLM_Model
from structures import TokenTrie, TokenGrammar


class Decoder(ABC, BaseModel):
    model_config = {"arbitrary_types_allowed": True}
    llm: Small_LLM_Model
    end_sequences: List[str]
    max_token: int = 50

    @abstractmethod
    def get_token_from_logits(self, logits: List[float]) -> int:
        pass

    @abstractmethod
    def is_generation_ending(self, generated: List[int]) -> bool:
        pass

    @abstractmethod
    def format_output(self, generated: List[int]) -> str:
        pass

    def generate(self, prompt: str) -> str:
        prompt_tokens: List[int] = self.llm.encode(prompt).tolist()[0]
        generated: List[int] = []

        while not self.is_generation_ending(generated=generated):
            logits = self.llm.get_logits_from_input_ids(prompt_tokens)
            token = self.get_token_from_logits(logits)
            generated.append(token)
            prompt_tokens.append(token)

        return self.format_output(generated=generated)


class TrieDecoder(Decoder):
    sequences: List[str]
    token_trie: TokenTrie = TokenTrie()
    node: TokenTrie = Field(init=False)

    def model_post_init(self, _) -> None:
        end_sequences_tokens: List[List[int]] = [
            self.llm.encode(end_sequence).tolist()[0]
            for end_sequence in self.end_sequences
        ]

        for sequence in self.sequences:
            tokens = self.llm.encode(sequence).tolist()[0]
            self.token_trie.add_path(
                tokens=tokens,
                end_sequences_tokens=end_sequences_tokens
            )

        self.node = self.token_trie

    def get_token_from_logits(self, logits) -> int:
        valid_tokens = list(self.node.children.keys())
        token = max(valid_tokens, key=lambda t: logits[t])
        self.node = self.node.children[token]
        return token

    def is_generation_ending(self, generated) -> bool:
        if self.node.path_count == 1:
            return True
        return False

    def format_output(self, generated):
        while self.node.children:
            token = list(self.node.children.keys())[0]
            generated.append(token)
            self.node = self.node.children[token]

        return self.llm.decode(generated)


class GrammarDecoder(Decoder):
    grammar: Dict
    token_grammar: TokenGrammar = TokenGrammar()


class FreeDecoder(Decoder):
    pass
