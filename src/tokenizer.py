"""Portable byte-level BPE tokenizer driven by a Hugging Face tokenizer.json.

Works with any model whose tokenizer.json declares a BPE model and a
byte-level pre-tokenizer (Qwen, GPT-2, Llama 3, Mistral, DeepSeek, ...).
SentencePiece tokenizers are not supported.
"""

import json
from typing import Any, Dict, List, Optional, Tuple

import regex as re
from pydantic import BaseModel, PrivateAttr

from llm_sdk import Small_LLM_Model


_DEFAULT_GPT2_PATTERN = (
    r"'s|'t|'re|'ve|'m|'ll|'d"
    r"| ?\p{L}+| ?\p{N}+"
    r"| ?[^\s\p{L}\p{N}]+"
    r"|\s+(?!\S)|\s+"
)


def _build_byte_encoder() -> Dict[int, str]:
    """Return GPT-2's reversible byte -> printable-unicode mapping.

    188 already-printable bytes map to themselves; the remaining 68 map
    to codepoints 256..323 so every byte becomes one visible character.
    """
    printable = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("\xa1"), ord("\xac") + 1))
        + list(range(ord("\xae"), ord("\xff") + 1))
    )
    mapped = printable[:]
    extra = 0
    for b in range(256):
        if b not in printable:
            printable.append(b)
            mapped.append(256 + extra)
            extra += 1
    return dict(zip(printable, [chr(c) for c in mapped]))


def _extract_pretokenizer_pattern(pre_tok: Any) -> str:
    """Walk a tokenizer.json pre_tokenizer config and return its regex.

    Handles Sequence and Split node types. Falls back to the GPT-2
    default when the config doesn't name an explicit pattern.
    """
    if not isinstance(pre_tok, dict):
        return _DEFAULT_GPT2_PATTERN

    node_type = pre_tok.get("type")

    if node_type == "Sequence":
        for child in pre_tok.get("pretokenizers", []):
            pattern = _extract_pretokenizer_pattern(child)
            if pattern != _DEFAULT_GPT2_PATTERN:
                return pattern
        return _DEFAULT_GPT2_PATTERN

    if node_type == "Split":
        pat = pre_tok.get("pattern", {})
        if isinstance(pat, dict) and "Regex" in pat:
            return str(pat["Regex"])

    return _DEFAULT_GPT2_PATTERN


def _parse_merges(raw: Any) -> Dict[Tuple[str, str], int]:
    """Parse merges into a {pair: rank} dict.

    Supports both the legacy "a b" string format and the newer [a, b]
    list format seen in recent tokenizer.json files.
    """
    merges: Dict[Tuple[str, str], int] = {}
    if not isinstance(raw, list):
        return merges

    for rank, entry in enumerate(raw):
        if isinstance(entry, str):
            parts = entry.split(" ", 1)
            if len(parts) == 2:
                merges[(parts[0], parts[1])] = rank
        elif isinstance(entry, list) and len(entry) == 2:
            a, b = entry
            if isinstance(a, str) and isinstance(b, str):
                merges[(a, b)] = rank

    return merges


class Tokenizer(BaseModel):
    """Byte-level BPE tokenizer built from a Hugging Face tokenizer.json."""

    model_config = {"arbitrary_types_allowed": True}

    llm: Small_LLM_Model

    _vocab: Dict[str, int] = PrivateAttr()
    _inv_vocab: Dict[int, str] = PrivateAttr()
    _merges: Dict[Tuple[str, str], int] = PrivateAttr()
    _byte_encoder: Dict[int, str] = PrivateAttr()
    _byte_decoder: Dict[str, int] = PrivateAttr()
    _pattern: Any = PrivateAttr()
    _added_tokens: Dict[str, int] = PrivateAttr()
    _inv_added_tokens: Dict[int, str] = PrivateAttr()
    _added_splitter: Any = PrivateAttr()
    _bpe_cache: Dict[str, List[str]] = PrivateAttr()

    def model_post_init(self, _: object) -> None:
        """Load tokenizer.json and precompute derived structures."""
        config = self._load_config()

        self._load_model(config)
        self._load_added_tokens(config)

        self._byte_encoder = _build_byte_encoder()
        self._byte_decoder = {v: k for k, v in self._byte_encoder.items()}

        pattern = _extract_pretokenizer_pattern(config.get("pre_tokenizer"))
        self._pattern = re.compile(pattern)

        self._bpe_cache = {}

    def _load_config(self) -> Dict[str, Any]:
        """Read tokenizer.json from disk."""
        path = self.llm.get_path_to_tokenizer_file()
        with open(path, encoding="utf-8") as f:
            config = json.load(f)
        if not isinstance(config, dict):
            raise ValueError("tokenizer.json: root must be an object")
        return config

    def _load_model(self, config: Dict[str, Any]) -> None:
        """Extract vocab and merges from the config."""
        model = config.get("model", {})
        if model.get("type") != "BPE":
            raise ValueError(
                f"Unsupported tokenizer model type "
                f"{model.get('type')!r}: only byte-level BPE is supported"
            )

        vocab = model.get("vocab", {})
        if not isinstance(vocab, dict):
            raise ValueError("tokenizer.json: 'vocab' must be an object")

        self._vocab = {str(k): int(v) for k, v in vocab.items()}
        self._inv_vocab = {v: k for k, v in self._vocab.items()}
        self._merges = _parse_merges(model.get("merges", []))

    def _load_added_tokens(self, config: Dict[str, Any]) -> None:
        """Extract special/added tokens and build the splitter regex."""
        added: Dict[str, int] = {}
        for entry in config.get("added_tokens", []):
            if isinstance(entry, dict):
                content = entry.get("content")
                tid = entry.get("id")
                if isinstance(content, str) and isinstance(tid, int):
                    added[content] = tid

        self._added_tokens = added
        self._inv_added_tokens = {v: k for k, v in added.items()}

        if added:
            specials = sorted(added, key=len, reverse=True)
            self._added_splitter = re.compile(
                "(" + "|".join(re.escape(s) for s in specials) + ")"
            )
        else:
            self._added_splitter = None

    def _find_best_pair(
        self, tokens: List[str]
    ) -> Optional[Tuple[str, str]]:
        """Return the adjacent pair with the lowest merge rank, or None."""
        best_rank: float = float("inf")
        best_pair: Optional[Tuple[str, str]] = None

        for pair in zip(tokens, tokens[1:]):
            rank = self._merges.get(pair)
            if rank is not None and rank < best_rank:
                best_rank = rank
                best_pair = pair

        return best_pair

    def _merge_pair(
        self, tokens: List[str], pair: Tuple[str, str]
    ) -> List[str]:
        """Merge every non-overlapping occurrence of pair, left-to-right."""
        first, second = pair
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

        return merged

    def _bpe(self, word: str) -> List[str]:
        """Apply BPE merges to a single byte-encoded pre-token."""
        if word in self._bpe_cache:
            return self._bpe_cache[word]

        tokens = list(word)
        while len(tokens) >= 2:
            pair = self._find_best_pair(tokens)
            if pair is None:
                break
            tokens = self._merge_pair(tokens, pair)

        self._bpe_cache[word] = tokens
        return tokens

    def _byte_encode(self, pretoken: str) -> str:
        """Map each byte of pretoken to its printable-unicode stand-in."""
        return "".join(
            self._byte_encoder[b] for b in pretoken.encode("utf-8")
        )

    def _encode_segment(self, text: str) -> List[int]:
        """Encode a segment known to contain no added/special tokens."""
        ids: List[int] = []

        for pretoken in self._pattern.findall(text):
            encoded = self._byte_encode(pretoken)
            for tok in self._bpe(encoded):
                ids.append(self._vocab[tok])

        return ids

    def encode(self, text: str) -> List[int]:
        """Encode text into a list of token IDs.

        Added/special tokens are matched first and emitted as single IDs;
        remaining segments go through regex pre-tokenization, byte-to-
        unicode remapping, BPE merging, and vocab lookup.
        """
        if self._added_splitter is None:
            return self._encode_segment(text)

        ids: List[int] = []
        for segment in self._added_splitter.split(text):
            if not segment:
                continue
            if segment in self._added_tokens:
                ids.append(self._added_tokens[segment])
            else:
                ids.extend(self._encode_segment(segment))

        return ids

    def _byte_decode(self, text: str) -> str:
        """Byte-decode a unicode-remapped string back to utf-8 text."""
        raw = bytes(self._byte_decoder[c] for c in text)
        return raw.decode("utf-8", errors="replace")

    def decode(self, tokens: List[int]) -> str:
        """Decode token IDs back to text.

        Byte-level pieces are buffered and utf-8 decoded together so
        multi-byte characters split across tokens reconstruct correctly.
        Added/special tokens flush the buffer and emit literally.
        """
        pieces: List[str] = []
        buffer: List[str] = []

        for t in tokens:
            if t in self._inv_added_tokens:
                if buffer:
                    pieces.append(self._byte_decode("".join(buffer)))
                    buffer = []
                pieces.append(self._inv_added_tokens[t])
            else:
                buffer.append(self._inv_vocab[t])

        if buffer:
            pieces.append(self._byte_decode("".join(buffer)))

        return "".join(pieces)
