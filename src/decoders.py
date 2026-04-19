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
    generated: int = Field(default=0, init=False)

    @abstractmethod
    def get_token_from_logits(self, logits: List[float]) -> int:
        pass

    def is_generation_ending(self, generated):
        if self.generated >= self.max_token:
            return True
        answer = self.llm.decode(generated)
        for end_seq in self.end_sequences:
            idx = answer.find(end_seq)
            if idx != -1:
                return True
        return False

    def format_output(self, generated: List[int]) -> str:
        answer = self.llm.decode(generated)

        for end_seq in self.end_sequences:
            idx = answer.find(end_seq)
            if idx != -1:
                answer = answer[:idx]
                break

        return answer

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
    node: TokenTrie = Field(TokenTrie(), init=False)

    def model_post_init(self, _) -> None:
        end_sequences_tokens: List[List[int]] = [
            self.llm.encode(end_seq).tolist()[0]
            for end_seq in self.end_sequences
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

        return super().format_output(generated=generated)


class GrammarDecoder(Decoder):
    grammar: Dict
    token_grammar: TokenGrammar | None = Field(default=None, init=False)

    def model_post_init(self, _):
        self.token_grammar = TokenGrammar(
            grammar=self.grammar,
            end_set=self.end_sequences,
            llm=self.llm
        )

    def get_token_from_logits(self, logits):
        valid_tokens = self.token_grammar.get_available_tokens()
        token = max(valid_tokens, key=lambda t: logits[t])
        self.token_grammar.update_state(token=token)
        return token

    def is_generation_ending(self, generated):
        if self.token_grammar.can_end() or self.generated >= self.max_token:
            return super().is_generation_ending(generated=generated)
        return False


class FreeDecoder(Decoder):

    def get_token_from_logits(self, logits):
        self.generated += 1
        return logits.index(max(logits))
