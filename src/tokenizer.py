"""Byte-level BPE tokenizer built from a Hugging Face tokenizer.json."""

# json: to parse tokenizer.json.
# typing: standard annotations. Any is for objects we don't want to
#   type precisely (compiled regex, raw JSON nodes).
import json
from typing import Any, Dict, List, Tuple

# `regex` (not stdlib `re`) because we need \p{L}, \p{N}, etc., which
# stdlib re doesn't support. Aliased as `re` to read naturally.
import regex as re
from pydantic import BaseModel, PrivateAttr

from llm_sdk import Small_LLM_Model


def _build_byte_encoder() -> List[str]:
    """Reversible mapping: every byte value (0..255) -> one printable char.

    BPE wants to operate on strings, but it was trained on bytes. So we
    need a way to show any byte as a visible character without collisions.
    188 of the 256 bytes are already printable (ASCII and most Latin-1),
    so they map to themselves. The other 68 (control chars, space, etc.)
    get assigned codepoints 256..323 to avoid collisions with real chars.
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

    The pre_tokenizer field is a tree of nodes. We only care about
    `Split` nodes (which carry a regex) — if we find one, return its
    pattern. Other node types (Sequence, ByteLevel, ...) may contain
    children, so we recurse into them.
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
    """Byte-level BPE tokenizer built from a Hugging Face tokenizer.json."""
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
        """Pydantic's 'after-__init__' hook. Build all private state.

        We do it here rather than with default_factory because most of
        this depends on `self.llm`, which isn't available until the
        instance finishes constructing.
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

        Algorithm: start with the pre-token as a list of single chars.
        Repeatedly find the adjacent pair with the lowest merge rank,
        merge every occurrence of it, and loop. Stop when no adjacent
        pair is known in the merges table.
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
        """Text -> list of token IDs. Public entrypoint.

        Pipeline: regex-split text into pre-tokens, byte-encode each
        pre-token to stand-in chars, BPE-merge, vocab-lookup.
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
        """Token IDs -> text. Public entrypoint."""
        text = "".join(self._inv_vocab[t] for t in tokens)
        raw = bytes(self._byte_decoder[c] for c in text)
        return raw.decode("utf-8", errors="replace")
