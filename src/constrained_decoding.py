from llm_sdk.llm_sdk import Small_LLM_Model
from pydantic import BaseModel
from typing import List, Dict, Set


class TokenTrie(BaseModel):
    token_id: int = -1
    children: Dict[int, "TokenTrie"] = {}

    def add_path(self, tokens: List[int]) -> None:
        node = self
        for token in tokens:
            node.children.setdefault(token, TokenTrie(token_id=token))
            node = node.children[token]


class TokenGrammar(BaseModel):
    model_config = {"arbitrary_types_allowed": True}
    grammar: Dict
    end_set: List[str]
    llm: Small_LLM_Model
    state: str = "start"
    token_sets: Dict[str, Set[int]] = {}
    end_set_tokens: Set[int] = {}

    def model_post_init(self, _) -> None:
        self.state: str = "start"

        self.token_sets = {
            curr_set: {
                self.llm.encode(elem).tolist()[0][0]
                for elem in self.grammar["token_sets"][curr_set]
                if curr_set != "[ANY]"
            }
            for curr_set in self.grammar["token_sets"]
        }

        self.end_set_tokens = \
            self.llm.encode(self.end_set).tolist()[0][0]

    def update_state(self, token: int) -> None:
        for set_name, set_content in self.token_sets.items():
            if set_name == "[ANY]" or token in set_content:
                token_set = set_name
                break

        self.state = \
            self.grammar["states"][self.state]["transitions"][token_set]

    def get_available_tokens(self) -> List[int] | None:
        token_set = set()

        for set_name, set_content in self.token_sets.items():
            if set_name == "any":
                return None
            if set_name in self.grammar["states"][self.state]["sets"]:
                token_set |= set_content

        token_list = list(token_set)

        if self.grammar["states"][self.state]["can_end"]:
            token_list.extend(list(self.end_set_tokens))

        return token_list


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

    def generate_from_grammar(
            self,
            prompt: str,
            grammar: Dict,
            end_set: List[str]
    ) -> str:

        token_grammar = TokenGrammar(
            grammar=grammar,
            end_set=end_set,
            llm=self.llm
        )

        prompt_tokens: List[int] = self.llm.encode(prompt).tolist()[0]
        end_set_tokens = {
            self.llm.encode(end_elem).tolist()[0][0]
            for end_elem in end_set
        }

        generated: List[int] = []

        for _ in range(50):
            logits = self.llm.get_logits_from_input_ids(prompt_tokens)
            valid_tokens = token_grammar.get_available_tokens()
            if not valid_tokens:
                best_token = logits.index(max(logits))
            else:
                best_token = max(valid_tokens, key=lambda t: logits[t])

            if best_token in end_set_tokens:
                break

            prompt_tokens.append(best_token)
            generated.append(best_token)
            token_grammar.update_state(best_token)

        answer = self.llm.decode(generated)

        return answer
