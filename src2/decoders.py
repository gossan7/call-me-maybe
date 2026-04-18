from typing import List, Dict
from abc import ABC, abstractmethod
from pydantic import BaseModel
from llm_sdk.llm_sdk import Small_LLM_Model
from structures import TokenTrie, TokenGrammar


class Decoder(ABC, BaseModel):
    model_config = {"arbitrary_types_allowed": True}
    llm: Small_LLM_Model
    end_sequences: List[str]

    @abstractmethod
    def get_token_from_logits(self, logits: List[float]) -> int:
        pass

    @abstractmethod
    def is_generation_ending(self, generated: List[int]):
        pass

    def generate(self, prompt: str) -> str:
        prompt_tokens: List[int] = self.llm.encode(prompt).tolist()[0]
        generated: List[int] = []


class TrieDecoder(Decoder):
    sequences: List[str]
    token_trie: TokenTrie = TokenTrie()


class GrammarDecoder(Decoder):
    grammar: Dict
    token_grammar: TokenGrammar = TokenGrammar()


class FreeDecoder(Decoder):
    pass
