from typing import List, Dict, Set, Any
from pydantic import BaseModel

from llm_sdk import Small_LLM_Model

from .tokenizer import Tokenizer


class Functions(BaseModel):
    functions: List[Dict[Any, Any]]
    functions_mapping: Dict[str, Dict[Any, Any]] = {}

    def model_post_init(self, _: Any) -> None:
        """Build a name-keyed lookup table from the list of functions."""
        self.functions_mapping = {
            func["name"]: func
            for func in self.functions
        }

    def get_parameters(self, name: str) -> Dict[str, Dict[str, str]]:
        """Return the parameter schema of the function named `name`.

        Args:
            name: The function name.

        Returns:
            Mapping from parameter name to its schema definition.
        """
        parameters = self.functions_mapping[name]["parameters"]
        assert isinstance(parameters, dict)
        return parameters

    def get_description(self, name: str) -> str:
        """Return the human description of the function named `name`.

        Args:
            name: The function name.

        Returns:
            The description string.
        """
        description = self.functions_mapping[name]["description"]
        assert isinstance(description, str)
        return description

    def get_returns(self, name: str) -> str:
        """Return the declared return type of the function named `name`.

        Args:
            name: The function name.

        Returns:
            The return type string.
        """
        returns = self.functions_mapping[name]["returns"]["type"]
        assert isinstance(returns, str)
        return returns

    def get_names_list(self) -> List[str]:
        """Return the list of all available function names."""
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
        """Insert a token sequence followed by every end-sequence variant.

        Each node keeps a counter of how many paths pass through it so
        callers can detect when the trie has narrowed to a single choice.

        Args:
            tokens: The token ids of the main sequence.
            end_sequences_tokens: List of terminator sequences to append
                after the main sequence.
        """
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
    state: str = ""
    token_sets: Dict[str, Set[int]] = {}
    end_set_tokens: Set[int] = set()

    def model_post_init(self, _: Any) -> None:
        """Tokenise every grammar token set and the end-sequence alphabet."""
        self.state: str = self.grammar["start"]

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
        """Advance the state machine according to `token`.

        End-sequence tokens leave the state unchanged; every other token
        triggers the transition declared for its token set.

        Args:
            token: The token id that has just been emitted.
        """
        if token in self.end_set_tokens:
            return

        for set_name, set_content in self.token_sets.items():
            if token in set_content:
                token_set = set_name
                break

        self.state = \
            self.grammar["states"][self.state]["transitions"][token_set]

    def can_end(self) -> bool:
        """Return True when the current state is an accepting state."""
        can_end_flag = self.grammar["states"][self.state]["can_end"]
        assert isinstance(can_end_flag, bool)
        return can_end_flag

    def get_available_tokens(self) -> List[int]:
        """Return every token id allowed in the current state.

        The returned list is the union of every token set referenced by
        the current state, plus the end-sequence alphabet when the state
        is an accepting state.
        """
        token_set = set()

        for set_name, set_content in self.token_sets.items():
            if set_name in self.grammar["states"][self.state]["sets"]:
                token_set |= set_content

        token_list = list(token_set)

        if self.can_end():
            token_list.extend(list(self.end_set_tokens))

        return token_list
