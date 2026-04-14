from llm_sdk.llm_sdk import Small_LLM_Model
from typing import Dict, List


class TrieNode:
    def __init__(self, token_id: int) -> None:
        self.token_id: int = token_id
        self.children: Dict[int, "TrieNode"] = {}

    def add_child(self, node: "TrieNode") -> None:
        self.children[node.token_id] = node


def build_trie(
    labels: List[str],
    end_token: int,
    llm: Small_LLM_Model
) -> TrieNode:
    root = TrieNode(-1)

    for label in labels:
        tokens: List[int] = llm.encode(label).tolist()[0]
        tokens.append(end_token)
        node = root

        for token in tokens:
            if token not in node.children:
                node.add_child(TrieNode(token))
            node = node.children[token]

    return root


def constrained_generate(
    prompt_tokens: List[int],
    root: TrieNode,
    end_token: int,
    llm: Small_LLM_Model
) -> List[int]:
    generated: List[int] = []
    node = root

    while node.children:
        logits = llm.get_logits_from_input_ids(prompt_tokens)

        valid_tokens = list(node.children.keys())
        best_token = max(valid_tokens, key=lambda t: logits[t])

        if best_token == end_token:
            break

        prompt_tokens.append(best_token)
        generated.append(best_token)
        node = node.children[best_token]

    return generated


def build_prompt(
    functions: List[Dict],
    user_prompt: str,
    label_map: Dict[str, str]
) -> str:

    prompt = ""

    for func in functions:
        params = ",".join(
            f"{k}:{v['type']}"
            for k, v in func["parameters"].items()
        )
        returns = func["returns"]["type"]
        label = label_map[func["name"]]
        prompt += f"F{label} ({params}) -> {returns}: {func['description']}\n"

    prompt += "Select the appropriate function for the prompt.\n"

    prompt += f"\nPrompt: {user_prompt}\n"
    prompt += "Function: \"F"

    # print(prompt)

    return prompt


def select_function(
    functions: List[Dict],
    user_prompt: str,
    llm: Small_LLM_Model
) -> str:

    label_to_name: Dict[str, str] = {}
    name_to_label: Dict[str, str] = {}

    for i, func in enumerate(functions):
        label = f"{i}"
        name = func["name"]
        if name not in name_to_label:
            name_to_label[name] = label
            label_to_name[label] = name

    prompt = build_prompt(functions, user_prompt, name_to_label)
    prompt_tokens: List[int] = llm.encode(prompt).tolist()[0]

    end_token: int = llm.encode('"').tolist()[0][0]
    labels = list(label_to_name.keys())
    root = build_trie(labels, end_token, llm)

    generated = constrained_generate(prompt_tokens, root, end_token, llm)
    label = llm.decode(generated)

    return label_to_name[label]
