*This project has been created as part of the 42 curriculum by vquetier.*

# call me maybe

## Description

`call me maybe` is an introduction to **function calling in LLMs**. Given a
natural-language prompt and a catalogue of function signatures, the program
picks the function the user means and fills in its arguments with the right
types — without ever letting the model emit free-form text.

For example:

- **Input prompt:** `"What is the sum of 40 and 2?"`
- **Output:** `{"name": "fn_add_numbers", "parameters": {"a": 40.0, "b": 2.0}}`

The challenge is that small models (we use the 0.6B-parameter
`Qwen/Qwen3-0.6B`) are unreliable at producing structured JSON when left to
their own devices. The project uses **constrained decoding** to force every
generated token to respect both the JSON structure and the schema declared in
`functions_definition.json`, pushing reliability to 100% even with a tiny
model.

## Instructions

Requirements: Python ≥ 3.10 and [uv](https://docs.astral.sh/uv/).

```bash
# Install dependencies and prepare the venv
make install

# Run with the default input files in data/input/
make run

# Or pass custom paths explicitly
uv run python -m src \
    --functions_definition data/input/functions_definition.json \
    --input data/input/function_calling_tests.json \
    --output data/output/function_calls.json
```

Other Makefile targets:

- `make debug` — run under `pdb`
- `make lint` — run flake8 and mypy with the mandatory flags
- `make lint-strict` — run mypy with `--strict`
- `make clean` — remove `__pycache__` and `.mypy_cache`
- `make fclean` — `clean` plus removes the `.venv`

The program accepts these CLI flags:

| Flag | Default | Purpose |
|------|---------|---------|
| `--functions_definition` | `data/input/functions_definition.json` | Function schema file |
| `--input` | `data/input/function_calling_tests.json` | Prompts to process |
| `--output` | `data/output/function_calls.json` | Where results are written |
| `--grammar` | `data/grammar/grammar.json` | Grammar state machines for numeric/integer decoding |
| `--model` | `Qwen/Qwen3-0.6B` | Hugging Face model identifier |
| `--debug` | off | Print intermediate generation steps |

## Example usage

```console
$ make run

Generating...
(33.5s) 11/11 Done !
Results written to data/output/function_calls.json

$ cat data/output/function_calls.json
[
  {
    "prompt": "What is the sum of 2 and 3?",
    "name": "fn_add_numbers",
    "parameters": {"a": 2.0, "b": 3.0}
  },
  {
    "prompt": "Reverse the string 'hello'",
    "name": "fn_reverse_string",
    "parameters": {"s": "hello"}
  },
  ...
]
```

## Algorithm explanation

Constrained decoding intercepts the LLM's token-by-token generation and
restricts it to only the tokens that remain valid at each step. The solution
uses three decoder strategies, picked per field.

**Function selection — `TrieDecoder`.**
All legal function names are inserted into a trie of token ids. At each step,
the decoder looks at the trie node it's currently on, restricts the logits to
the keys of that node's children, picks the argmax, and descends. When the
trie narrows to a single remaining path it fast-forwards to the end without
consulting the model. This guarantees the emitted name is exactly one of the
allowed functions.

**Numeric arguments — `GrammarDecoder`.**
Numbers and integers are described in `data/grammar/grammar.json` as small
finite-state machines. States declare which **token sets** (digits, sign,
dot) are valid transitions and which states are accepting. At each step the
decoder asks the grammar for the set of allowed token ids, picks the argmax
from that subset, and advances the FSM. The FSM guarantees every output is a
syntactically valid number.

**Boolean arguments — `TrieDecoder`.**
Same as function selection, with a two-branch trie (`true` / `false`).

**String arguments — `FreeDecoder`.**
No structural constraint — strings can contain almost anything. The decoder
picks the raw argmax at each step and stops when it hits a quote or newline.

Every decoder inherits from a common abstract `Decoder` that owns the
tokenizer, the end-sequence detection, and the main generation loop; each
subclass only overrides `get_token_from_logits` and (where needed)
`is_generation_ending` and `format_output`.

## Design decisions

- **Custom tokenizer.** `src/tokenizer.py` reimplements byte-level BPE from
  the model's `tokenizer.json` (vocab + merges + pre-tokenizer regex + byte
  encoder). This makes the decoder independent from the SDK's `encode` /
  `decode` wrappers and round-trip-matches them for our inputs.
- **Pydantic everywhere.** Every domain class (`Functions`, `TokenTrie`,
  `TokenGrammar`, `Decoder` and its subclasses, `Extractor`, `Tokenizer`,
  `Spinner`) is a `BaseModel`. Internal-only state uses `PrivateAttr`.
- **Separate extraction passes.** The function name and each parameter are
  generated in independent decoder runs with their own prompt. This keeps
  each constrained-decoding context small and easy to reason about.
- **Grammar as data.** The FSM for numbers is plain JSON (`data/grammar/`).
  Adding a new primitive type means adding a state machine to that file
  rather than writing code.
- **Graceful error handling.** Missing or malformed inputs raise a
  `DataError` with a clear message instead of a stack trace. The prompt
  loop continues past a single bad prompt.

## Performance analysis

Measured on a standard laptop running `Qwen/Qwen3-0.6B` on CPU against the
11-prompt demo input set.

- **Accuracy:** [X / 11] correct function selections and parameters
  ([X]% accuracy).
- **JSON validity:** 100%. Constrained decoding guarantees every output
  parses without error and matches the schema.
- **Throughput:** the full 11-prompt run completes in ~33 seconds on CPU,
  well under the 5-minute budget.
- **Reliability:** repeated runs produce identical outputs (argmax
  sampling, no temperature).

The custom tokenizer is a pure-Python pipeline; it's slightly slower than
the SDK's Rust-backed tokenizer on first invocation but benefits from a BPE
cache on repeated pre-tokens.

## Challenges faced

- **Silent tokenizer drift.** An early version of the custom tokenizer used
  a hardcoded GPT-2 pre-tokenizer regex, which *almost* matched Qwen3's but
  differed on whitespace handling. Outputs still decoded correctly, but the
  logits shifted enough that function selection started picking the wrong
  function on some prompts. The fix was to read the exact regex from the
  model's `tokenizer.json` rather than assume it.
- **BPE merge order.** The first merge loop picked the leftmost applicable
  pair rather than the globally lowest-rank pair. This produced valid-
  looking token sequences that still disagreed with the reference tokenizer.
  Once reading the merges dict by rank rather than by occurrence order, the
  round-trip tests passed.
- **Multi-byte UTF-8 on decode.** Because BPE pieces are byte-level, a
  single Unicode character can span two tokens. The decoder now concatenates
  byte-encoded pieces into one string before doing a single UTF-8 decode
  with `errors="replace"`.
- **Pydantic + threading.** `Spinner` needs a `threading.Thread` and a
  `threading.Lock`, neither of which are native pydantic types. Using
  `PrivateAttr` plus `model_post_init` was the cleanest way to initialise
  them after the pydantic model finishes constructing.

## Testing strategy

- **Round-trip tokenizer check.** For each varied prompt, assert
  `Tokenizer.encode(s) == llm.encode(s).tolist()[0]` and
  `Tokenizer.decode(Tokenizer.encode(s)) == s`. This catches any drift
  between the custom tokenizer and the reference implementation.
- **Schema-validation pass.** The `validation` module rejects malformed
  grammar / function / prompt files before the model is even loaded.
- **End-to-end demo runs.** The provided input files exercise: one-argument
  functions, multi-argument functions, numeric extraction, string
  extraction, quoted strings in prompts, and prompts that are ambiguous
  between multiple functions.
- **Edge-case prompts.** Very long source strings, special characters,
  large numbers, and prompts that don't obviously map to any function.

## Resources

**Documentation & articles**

- Hugging Face `tokenizers` documentation — structure of `tokenizer.json`.
- GPT-2 paper and its reference implementation — byte-level BPE and the
  `bytes_to_unicode` mapping.
- Qwen3 model card on Hugging Face.
- Outlines and lm-format-enforcer blog posts for background on constrained
  decoding (implementation done from scratch; no code taken from them).
- PEP 257 — docstring conventions.
- Pydantic v2 docs — `PrivateAttr` and `model_post_init`.

**AI usage**

I used Claude as a pair-programming assistant throughout the project. It
was helpful for:

- Explaining BPE, the byte-level byte-to-unicode mapping, and the rationale
  for pre-tokenization before I implemented them.
- Refactoring the tokenizer module once a working version existed.
- Drafting docstrings, and this README.
- Surfacing edge cases I hadn't considered (multi-byte UTF-8 on decode,
  max-token cutoffs mid-number, etc.).

I did **not** use AI for:

- The constrained-decoding trie and grammar algorithms — these were
  designed by hand.
- The final implementation of the BPE merge loop — I wrote it, compared it
  against the reference tokenizer's output, and fixed it myself when it
  drifted.

Every piece of AI-generated code in the project has been read, modified,
and tested by me. I can explain and justify every line.