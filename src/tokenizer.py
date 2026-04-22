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
    # Start the mapping with bytes that ARE already printable characters.
    # Three ranges cover them:
    #   ord("!")..ord("~")     -> printable ASCII except space
    #   \xa1..\xac             -> Latin-1 printable range 1
    #   \xae..\xff             -> Latin-1 printable range 2
    #                             (skipping \xad, soft hyphen,
    # which is invisible)
    printable = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("\xa1"), ord("\xac") + 1))
        + list(range(ord("\xae"), ord("\xff") + 1))
    )

    # `mapped[i]` is the codepoint that will stand in for byte `printable[i]`.
    # For safe bytes it's the byte itself (identity mapping).
    mapped = printable.copy()

    # Now handle the unsafe bytes. For each byte 0..255 not already in
    # `printable`, append it and give it the next free codepoint starting
    # at 256. This way every byte gets exactly one visible stand-in.
    extra = 0
    for b in range(256):
        if b not in printable:
            printable.append(b)
            mapped.append(256 + extra)
            extra += 1

    # Pair them up into {byte_value: stand_in_char}. Examples:
    #   0x20 (space)   -> chr(256) = "Ġ"
    #   0x0A (newline) -> chr(266) = "Ċ"
    #   0x41 ('A')     -> "A" (unchanged)

    dict_mapping = dict(zip(printable, [chr(c) for c in mapped]))

    return [dict_mapping[i] for i in range(256)]


def _extract_pattern(pre_tok: Any) -> str:
    """Find the pre-tokenizer regex inside a tokenizer.json pre_tokenizer.

    The pre_tokenizer field is a tree of nodes. We only care about
    `Split` nodes (which carry a regex) — if we find one, return its
    pattern. Other node types (Sequence, ByteLevel, ...) may contain
    children, so we recurse into them.
    """
    # Guard: if this isn't a dict, there's nothing to look into.
    if isinstance(pre_tok, dict):
        # A Split node looks like {"type":"Split","pattern":{"Regex":"..."}}.
        # This is the only node that carries the regex we want.
        if pre_tok.get("type") == "Split":
            pat = pre_tok.get("pattern", {})
            if isinstance(pat, dict) and "Regex" in pat:
                return str(pat["Regex"])

        # Not a Split — might be a Sequence or similar container. Recurse
        # into its children (the "pretokenizers" key) and use the first
        # pattern we find.
        for child in pre_tok.get("pretokenizers", []):
            found = _extract_pattern(child)
            if found:
                return found

    # Nothing found on this branch. Return empty string; the caller will
    # compile it, which gives a regex that matches empty strings (fine
    # for our defaults — every model we care about actually has a Split).
    return ""


class Tokenizer(BaseModel):
    """Byte-level BPE tokenizer built from a Hugging Face tokenizer.json."""

    # We hold a compiled regex (not a pydantic-native type) on `_pattern`.
    # This flag tells pydantic to accept that kind of field without whining.
    model_config = {"arbitrary_types_allowed": True}

    # The only public field: the LLM wrapper. Everything else we need
    # (vocab, merges, regex, byte map) we derive from it in model_post_init.
    llm: Small_LLM_Model

    # All internal state is private. PrivateAttr keeps it outside pydantic's
    # validation/serialisation machinery — it's just instance state.
    _vocab: Dict[str, int] = PrivateAttr()          # str piece -> token ID
    _inv_vocab: Dict[int, str] = PrivateAttr()      # token ID -> str piece
    _merges: Dict[Tuple[str, str], int] = PrivateAttr()  # (a, b) -> rank
    _byte_encoder: List[str] = PrivateAttr()   # byte -> stand-in char
    _byte_decoder: Dict[str, int] = PrivateAttr()   # stand-in char -> byte
    _pattern: re.Pattern = PrivateAttr()            # compiled pre-tok regex
    _bpe_cache: Dict[str, List[str]] = PrivateAttr()  # word -> bpe pieces

    def model_post_init(self, _: object) -> None:
        """Pydantic's 'after-__init__' hook. Build all private state.

        We do it here rather than with default_factory because most of
        this depends on `self.llm`, which isn't available until the
        instance finishes constructing.
        """
        # Ask the SDK where tokenizer.json lives on disk, then load it.
        path = self.llm.get_path_to_tokenizer_file()
        with open(path, encoding="utf-8") as f:
            config: Dict[str, Dict[str, Any]] = json.load(f)

        # The "model" subtree holds vocab and merges. Bomb loudly if
        # the file is malformed — nothing below makes sense without it.
        model = config["model"]

        # vocab is {piece: id}. We need both directions; invert once here.
        self._vocab = model["vocab"]
        self._inv_vocab = {v: k for k, v in self._vocab.items()}

        # Merges is a LIST of pairs. The list position IS the rank —
        # rank 0 is applied first (highest priority), rank N last.
        # Older tokenizer.json uses "a b" strings, newer uses [a, b] lists.
        # Handle both by checking the type of each entry.
        self._merges = {}
        for rank, entry in enumerate(model.get("merges", [])):
            if isinstance(entry, str):
                a, b = entry.split(" ", 1)
            else:
                a, b = entry
            self._merges[(a, b)] = rank

        # Build the byte <-> stand-in character tables. Same for every
        # byte-level BPE tokenizer, but we compute per-instance for clarity.
        self._byte_encoder = _build_byte_encoder()
        self._byte_decoder = {v: i for i, v in enumerate(self._byte_encoder)}

        # Dig out the pre-tokenizer regex from the config tree, then
        # compile it once for use in encode().
        self._pattern = re.compile(_extract_pattern(
            config.get("pre_tokenizer")
        ))

        # BPE cache starts empty. Fills up as encode() runs and stays
        # populated for the lifetime of the tokenizer — common words
        # like "the", "fn_add", etc. benefit a lot.
        self._bpe_cache = {}

    def _bpe(self, word: str) -> List[str]:
        """Apply BPE merges to a byte-encoded pre-token.

        Algorithm: start with the pre-token as a list of single chars.
        Repeatedly find the adjacent pair with the lowest merge rank,
        merge every occurrence of it, and loop. Stop when no adjacent
        pair is known in the merges table.
        """
        # Cache hit: words recur in real text, skip the work.
        if word in self._bpe_cache:
            return self._bpe_cache[word]

        # Initial state: each character is its own "piece". BPE will
        # glue them into larger pieces.
        tokens = list(word)

        # Loop until no more merges are possible (or we're down to 1 token).
        while len(tokens) >= 2:

            # --- Step 1: find the best adjacent pair to merge. ---
            # "Best" = lowest rank in the merges table (lower = earlier
            # in the training, higher priority).
            # Using float("inf") as the initial best lets any real rank win.
            best_rank: float = float("inf")
            best_pair = None

            # zip(tokens, tokens[1:]) yields every adjacent (left, right) pair.
            for pair in zip(tokens, tokens[1:]):
                # Look up this pair's rank. None means "not a known merge".
                rank = self._merges.get(pair)
                if rank is not None and rank < best_rank:
                    best_rank = rank
                    best_pair = pair

            # If no adjacent pair is known, BPE has reached its fixed point.
            if best_pair is None:
                break

            # --- Step 2: merge every occurrence of best_pair. ---
            # This is done in a SINGLE pass, left-to-right, non-overlapping.
            # Example: pair=("l","l"), tokens=["l","l","l"] -> ["ll","l"],
            # not ["l","ll"] (because we consume the first occurrence first).
            first, second = best_pair
            merged: List[str] = []
            i = 0
            while i < len(tokens):
                # Condition: we're not at the last index, AND the next two
                # tokens match our pair. Then concatenate and jump by 2.
                if (
                    i < len(tokens) - 1
                    and tokens[i] == first
                    and tokens[i + 1] == second
                ):
                    merged.append(first + second)
                    i += 2
                else:
                    # No merge here. Keep the current token, move on.
                    merged.append(tokens[i])
                    i += 1

            # Replace tokens with the merged version and loop again to
            # find the next-best pair in this shorter list.
            tokens = merged

        # Memoise the final piece list so we skip the work next time.
        self._bpe_cache[word] = tokens
        return tokens

    def encode(self, text: str) -> List[int]:
        """Text -> list of token IDs. Public entrypoint.

        Pipeline: regex-split text into pre-tokens, byte-encode each
        pre-token to stand-in chars, BPE-merge, vocab-lookup.
        """
        ids: List[int] = []

        # Step 1: split the input into pre-tokens via the pattern.
        # findall returns all non-overlapping matches in order.
        for pretoken in self._pattern.findall(text):
            assert isinstance(pretoken, str)

            # Step 2: byte-encode. We go utf-8 first (so 'é' becomes 2
            # bytes, etc.), then map each byte to its stand-in char.
            # Result is a string that BPE can operate on.
            encoded = "".join(
                self._byte_encoder[b] for b in pretoken.encode("utf-8")
            )

            # Step 3: BPE the stand-in string into subword pieces.
            # Step 4: each piece must be in the vocab (it was trained
            # on exactly these pieces); look it up for its integer ID.
            for tok in self._bpe(encoded):
                ids.append(self._vocab[tok])

        return ids

    def decode(self, tokens: List[int]) -> str:
        """Token IDs -> text. Public entrypoint."""
        # Step 1: each ID maps to a stand-in-char string in the vocab.
        # Concatenate them into one long stand-in-char string.
        text = "".join(self._inv_vocab[t] for t in tokens)

        # Step 2: reverse the byte-encoding. Each char of `text` was a
        # stand-in for one byte; collect them back into real bytes.
        raw = bytes(self._byte_decoder[c] for c in text)

        # Step 3: those bytes are utf-8. Decode them into a real string.
        # errors="replace" makes invalid utf-8 become U+FFFD rather than
        # crashing — useful when decoding partial output mid-generation.
        return raw.decode("utf-8", errors="replace")
