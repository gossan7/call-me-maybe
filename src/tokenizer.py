import json
from typing import Any, Dict, List, Tuple

import regex as re
from pydantic import BaseModel, PrivateAttr

from llm_sdk import Small_LLM_Model


def _build_byte_encoder() -> List[str]:
    """Return GPT-2's reversible byte to printable-unicode table.

    188 already-printable bytes map to themselves; the remaining 68 map
    to codepoints 256..323 so every byte becomes a single visible char.
    The returned list is indexed directly by byte value (0..255).
    """
    printable = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("\xa1"), ord("\xac") + 1))
        + list(range(ord("\xae"), ord("\xff") + 1))
    )

    mapped = printable.copy()

    extra = 0
    for b in range(256):
        if b not in printable:
            printable.append(b)
            mapped.append(256 + extra)
            extra += 1

    dict_mapping = dict(zip(printable, [chr(c) for c in mapped]))
    return [dict_mapping[i] for i in range(256)]


def _extract_pattern(pre_tok: Any) -> str:
    """Find the pre-tokenizer regex inside a tokenizer.json pre_tokenizer.

    The pre_tokenizer field is a tree of nodes. Only `Split` nodes carry
    a regex; other node types (Sequence, ByteLevel, ...) may contain
    children, so this function recurses into them.

    Args:
        pre_tok: A pre_tokenizer subtree from tokenizer.json.

    Returns:
        The first Regex pattern found, or the empty string when none.
    """
    if isinstance(pre_tok, dict):
        if pre_tok.get("type") == "Split":
            pat = pre_tok.get("pattern", {})
            if isinstance(pat, dict) and "Regex" in pat:
                return str(pat["Regex"])

        for child in pre_tok.get("pretokenizers", []):
            found = _extract_pattern(child)
            if found:
                return found

    return ""


class Tokenizer(BaseModel):
    model_config = {"arbitrary_types_allowed": True}
    llm: Small_LLM_Model

    _vocab: Dict[str, int] = PrivateAttr()
    _inv_vocab: Dict[int, str] = PrivateAttr()
    _merges: Dict[Tuple[str, str], int] = PrivateAttr()
    _byte_encoder: List[str] = PrivateAttr()
    _byte_decoder: Dict[str, int] = PrivateAttr()
    _pattern: re.Pattern = PrivateAttr()
    _bpe_cache: Dict[str, List[str]] = PrivateAttr()

    def model_post_init(self, _: object) -> None:
        """Load tokenizer.json and precompute every derived structure.

        Built here rather than with default_factory because most of the
        state depends on `self.llm`, which isn't available until pydantic
        finishes constructing the instance.
        """
        path = self.llm.get_path_to_tokenizer_file()
        with open(path, encoding="utf-8") as f:
            config: Dict[str, Dict[str, Any]] = json.load(f)

        model = config["model"]

        self._vocab = model["vocab"]
        self._inv_vocab = {v: k for k, v in self._vocab.items()}

        self._merges = {}
        for rank, entry in enumerate(model.get("merges", [])):
            if isinstance(entry, str):
                a, b = entry.split(" ", 1)
            else:
                a, b = entry
            self._merges[(a, b)] = rank

        self._byte_encoder = _build_byte_encoder()
        self._byte_decoder = {v: i for i, v in enumerate(self._byte_encoder)}

        self._pattern = re.compile(_extract_pattern(
            config.get("pre_tokenizer")
        ))

        self._bpe_cache = {}

    def _bpe(self, word: str) -> List[str]:
        """Apply BPE merges to a byte-encoded pre-token.

        Starts with `word` as a list of single characters, then repeatedly
        finds the adjacent pair with the lowest merge rank and merges
        every occurrence of it, until no more known merges apply.

        Args:
            word: The byte-encoded pre-token to split into BPE pieces.

        Returns:
            The list of BPE pieces (each piece is a key in the vocab).
        """
        if word in self._bpe_cache:
            return self._bpe_cache[word]

        tokens = list(word)

        while len(tokens) >= 2:
            best_rank: float = float("inf")
            best_pair = None

            for pair in zip(tokens, tokens[1:]):
                rank = self._merges.get(pair)
                if rank is not None and rank < best_rank:
                    best_rank = rank
                    best_pair = pair

            if best_pair is None:
                break

            first, second = best_pair
            merged: List[str] = []
            i = 0
            while i < len(tokens):
                if (
                    i < len(tokens) - 1
                    and tokens[i] == first
                    and tokens[i + 1] == second
                ):
                    merged.append(first + second)
                    i += 2

                else:
                    merged.append(tokens[i])
                    i += 1

            tokens = merged

        self._bpe_cache[word] = tokens
        return tokens

    def encode(self, text: str) -> List[int]:
        """Encode text into a list of token IDs.

        Pipeline: regex-split text into pre-tokens, byte-encode each
        pre-token to stand-in chars, BPE-merge, vocab-lookup.

        Args:
            text: The text to tokenise.

        Returns:
            The list of token ids.
        """
        ids: List[int] = []

        for pretoken in self._pattern.findall(text):
            assert isinstance(pretoken, str)

            encoded = "".join(
                self._byte_encoder[b] for b in pretoken.encode("utf-8")
            )

            for tok in self._bpe(encoded):
                ids.append(self._vocab[tok])

        return ids

    def decode(self, tokens: List[int]) -> str:
        """Decode a list of token IDs back into text.

        Args:
            tokens: The token ids to decode.

        Returns:
            The reconstructed text.
        """
        text = "".join(self._inv_vocab[t] for t in tokens)
        raw = bytes(self._byte_decoder[c] for c in text)
        return raw.decode("utf-8", errors="replace")
