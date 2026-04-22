from typing import List, Dict, Any
from abc import ABC, abstractmethod
from pydantic import BaseModel, Field

from llm_sdk import Small_LLM_Model

from .structures import TokenTrie, TokenGrammar
from .tokenizer import Tokenizer


class Decoder(ABC, BaseModel):
    model_config = {"arbitrary_types_allowed": True}
    llm: Small_LLM_Model
    tokenizer: Tokenizer
    end_sequences: List[str]
    max_token: int = 50
    generated: int = Field(default=0, init=False)

    @abstractmethod
    def get_token_from_logits(self, logits: List[float]) -> int:
        """Pick the next token id from the raw model logits.

        Args:
            logits: One logit per token id from the model.

        Returns:
            The selected token id.
        """
        pass

    def is_generation_ending(self, generated: List[int]) -> bool:
        """Return True when generation should stop.

        Stops when the token budget is exhausted or when the decoded
        text contains any configured end sequence.

        Args:
            generated: Tokens generated so far.

        Returns:
            True if generation should end, False otherwise.
        """
        if self.generated >= self.max_token:
            return True
        answer = self.tokenizer.decode(generated)
        for end_seq in self.end_sequences:
            idx = answer.find(end_seq)
            if idx != -1:
                return True
        return False

    def format_output(self, generated: List[int]) -> str:
        """Decode generated tokens and truncate at the first end sequence.

        Args:
            generated: Tokens generated so far.

        Returns:
            The cleaned output string.
        """
        answer = self.tokenizer.decode(generated)

        for end_seq in self.end_sequences:
            idx = answer.find(end_seq)
            if idx != -1:
                answer = answer[:idx]
                break

        return answer

    def generate(self, prompt: str) -> str:
        """Run the full token-by-token generation loop for `prompt`.

        Args:
            prompt: The text prompt to condition generation on.

        Returns:
            The generated output string, truncated at an end sequence.
        """
        prompt_tokens: List[int] = self.tokenizer.encode(prompt)
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

    def model_post_init(self, _: Any) -> None:
        """Build the token trie from the allowed sequences."""
        end_sequences_tokens: List[List[int]] = [
            self.tokenizer.encode(end_seq)
            for end_seq in self.end_sequences
        ]

        for sequence in self.sequences:
            tokens = self.tokenizer.encode(sequence)
            self.token_trie.add_path(
                tokens=tokens,
                end_sequences_tokens=end_sequences_tokens
            )

        self.node = self.token_trie

    def get_token_from_logits(self, logits: List[float]) -> int:
        """Pick the highest-logit token among the current trie children.

        Args:
            logits: One logit per token id from the model.

        Returns:
            The selected token id.
        """
        valid_tokens = list(self.node.children.keys())
        token = max(valid_tokens, key=lambda t: logits[t])
        self.node = self.node.children[token]
        return token

    def is_generation_ending(self, generated: List[int]) -> bool:
        """Return True when the trie has narrowed down to a single path.

        Args:
            generated: Tokens generated so far (unused here).

        Returns:
            True once the trie node has no branching left.
        """
        if self.node.path_count == 1:
            return True
        return False

    def format_output(self, generated: List[int]) -> str:
        """Walk any remaining deterministic trie path before decoding.

        Args:
            generated: Tokens generated so far.

        Returns:
            The decoded and truncated output string.
        """
        while self.node.children:
            token = list(self.node.children.keys())[0]
            generated.append(token)
            self.node = self.node.children[token]

        return super().format_output(generated=generated)

    def generate(self, prompt: str) -> str:
        """Run generation while fast-forwarding through deterministic paths.

        Args:
            prompt: The text prompt to condition generation on.

        Returns:
            The generated output string.
        """
        prompt_tokens: List[int] = self.tokenizer.encode(prompt)
        generated: List[int] = []

        while not self.is_generation_ending(generated=generated):
            while len(self.node.children) == 1:
                token = list(self.node.children.keys())[0]
                generated.append(token)
                prompt_tokens.append(token)
                self.node = self.node.children[token]

            logits = self.llm.get_logits_from_input_ids(prompt_tokens)
            token = self.get_token_from_logits(logits)
            generated.append(token)
            prompt_tokens.append(token)

        return self.format_output(generated=generated)


class GrammarDecoder(Decoder):
    grammar: Dict[str, Any]
    token_grammar: TokenGrammar | None = Field(default=None, init=False)

    def model_post_init(self, _: Any) -> None:
        """Build the grammar state machine from the provided grammar dict."""
        self.token_grammar = TokenGrammar(
            grammar=self.grammar,
            end_set=self.end_sequences,
            llm=self.llm,
            tokenizer=self.tokenizer
        )

    def get_token_from_logits(self, logits: List[float]) -> int:
        """Pick the highest-logit token among those allowed by the grammar.

        Args:
            logits: One logit per token id from the model.

        Returns:
            The selected token id.
        """
        self.generated += 1
        assert isinstance(self.token_grammar, TokenGrammar)
        valid_tokens = self.token_grammar.get_available_tokens()
        token = max(valid_tokens, key=lambda t: logits[t])
        self.token_grammar.update_state(token=token)
        return token

    def is_generation_ending(self, generated: List[int]) -> bool:
        """Return True only when the grammar is in an acceptable end state.

        Args:
            generated: Tokens generated so far.

        Returns:
            True if generation may end, False otherwise.
        """
        assert isinstance(self.token_grammar, TokenGrammar)
        if self.token_grammar.can_end() or self.generated >= self.max_token:
            return super().is_generation_ending(generated=generated)
        return False


class FreeDecoder(Decoder):
    def get_token_from_logits(self, logits: List[float]) -> int:
        """Pick the single highest-logit token with no constraint at all.

        Args:
            logits: One logit per token id from the model.

        Returns:
            The argmax token id.
        """
        self.generated += 1
        return logits.index(max(logits))
