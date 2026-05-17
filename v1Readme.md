# TTT-Search POC — v1 Implementation Journal

This document is a complete record of how v1 of the TTT-Search proof-of-concept
was built, what every component does, what went wrong along the way, how each
problem was diagnosed, what was changed to fix it, and what the final results
were.

It is intentionally exhaustive. Read it if you want to understand the project
end-to-end without re-deriving anything from the code or git history.

---

## 1. The hypothesis being tested

**Claim:** For coding problems, generating N=4 candidate solutions via
**test-time training** (TTT) — where each candidate comes from the same base
model but warmed up on a *different* short priming text — solves more problems
than generating N=4 candidates via plain temperature sampling from the same
model.

The "TTT" part: take a frozen base LLM, attach a small LoRA adapter, run a few
gradient steps of causal-LM loss on a priming text, generate one solution, then
discard the adapter. Do this with four different priming texts. Pick whichever
candidate the verifier (a unit-test runner) accepts.

The "search" part: the four priming texts steer the model toward four different
solution styles. The verifier acts as the search criterion.

**Numeric falsification rule from the design doc:** if TTT-search and naive
resampling land within ±10 absolute percentage points of each other on a
20-problem subset of HumanEval, the hypothesis is not supported.

---

## 2. Conditions compared

Three conditions, each producing one final candidate per problem:

| Condition | Description |
|---|---|
| `base_greedy` | Base model, temperature 0 (greedy decoding), single attempt. Floor. |
| `base_temperature` | Base model, T=0.8, top_p=0.95, four candidates with seeds 42–45, return first passing one. |
| `ttt_search` | Base + LoRA. Four candidates; for each, reset LoRA → run 8 AdamW steps on a different priming text → generate at T=0.3. Return first passing one. |

`base_greedy` was kept as a continuity floor and is not central to the
hypothesis. The real comparison is `base_temperature` vs `ttt_search`.

---

## 3. Project layout

```
ttt-search-poc/
├── README.md
├── requirements.txt
├── run_experiment.py
├── ttt_search/
│   ├── __init__.py
│   ├── loader.py            # model + tokenizer (single load, reused)
│   ├── verifier.py          # subprocess test runner
│   ├── benchmark.py         # HumanEval loader
│   ├── lora_utils.py        # zero / reset LoRA weights
│   ├── prompt_utils.py      # chat template + code extraction
│   └── conditions/
│       ├── __init__.py
│       ├── base_greedy.py
│       ├── base_temperature.py
│       └── ttt_search.py
├── priming/
│   ├── iterative.txt        # for-loop style implementations
│   ├── recursive.txt        # recursive with base cases
│   ├── edge_cases.txt       # defensive coding patterns
│   └── builtins.txt         # itertools / Counter / reduce style
└── results/                 # outputs after a run
    ├── summary.txt
    ├── detailed.json
    └── checkpoint_<cond>.json
```

`lora_utils.py` and `prompt_utils.py` were not in the original design doc
file list. They were added because they implement functionality the spec
required (resetting LoRA between variants, chat-template formatting, response
parsing) and grouping that logic outside the condition modules kept the
condition implementations small.

---

## 4. Component-by-component walkthrough

### 4.1 `loader.py`

Loads the model exactly once and caches it in module globals. All three
conditions reuse the same loaded instance, which removes "different load
state" as a confound.

Important decisions made here:

- **Model tag.** The design doc said `unsloth/Qwen3.5-0.8B`. That tag does
  exist on HuggingFace. It is the model used.
- **Fallback list.** A second tag (`unsloth/Qwen3-0.6B-unsloth-bnb-4bit`) is
  listed in case the primary fails. It never had to fire in practice.
- **Quantization.** Originally `load_in_4bit=True` per spec ("4-bit NF4 if FP8
  isn't available"). v1 used 4-bit. (v1.1 changed this; v1 kept the spec.)
- **LoRA attached at load time.** `FastLanguageModel.get_peft_model` is
  called with `r=4, target_modules=["q_proj","v_proj"], lora_alpha=8,
  lora_dropout=0.0`. The base model's frozen weights are not modified; only
  the LoRA matrices are trainable.
- **`use_gradient_checkpointing="unsloth"`.** Required for Qwen3.5 — see §5
  ("Why TTT initially crashed").
- **Tokenizer unwrap.** Qwen3.5-0.8B's HuggingFace tag returns a
  `Qwen3VLProcessor` (it is a vision-language model). When you call the
  processor on plain text, it tries to interpret the text as image URLs and
  crashes. The loader extracts the inner text-only tokenizer via
  `tokenizer.tokenizer if hasattr(tokenizer, "tokenizer") else tokenizer`.
  Every other module uses this unwrapped tokenizer.

### 4.2 `prompt_utils.py`

Two responsibilities: format the prompt for generation, and parse the
generated string back into runnable code.

**Formatting (`format_prompt`).** Qwen3.5-0.8B is an instruct model — feeding
it the raw HumanEval prompt (a function signature + docstring) doesn't work
because there is no instruction. The function wraps the HumanEval prompt in a
chat message:

```
Complete the following Python function. Return ONLY the complete function
code in a Python code block.

```python
<the HumanEval prompt verbatim>
```
```

`apply_chat_template` is called with `enable_thinking=False`. This disables
Qwen's `<think>...</think>` segment so the model goes straight to the answer,
making extraction simpler.

**Priming tokenization (`format_priming`).** Plain causal-LM tokenization of
the priming text, truncated to 512 tokens. No chat template — priming texts
are bare Python functions, not instructions.

**Extraction (`extract_code`).** The model's output is a string that may
contain markdown fences, leading text, or multiple code blocks. This function
turns it into code suitable for the verifier. Process:

1. Strip any `<think>...</think>` block (defensive — thinking is disabled,
   but if it ever leaks through this removes it).
2. Look for opening Python fences (` ```python ` or ` ``` `). If found, split
   the response into code blocks (the text after each opening fence, up to
   the next closing fence).
3. **Pick the last code block that contains the target function definition.**
   This is the critical detail explained in §5.3 — models often self-correct
   ("Wait, that's wrong, let me fix:") and the correct implementation is in
   the last block, not the first.
4. If no fenced blocks exist, check whether the raw text already contains the
   function definition. If so, return it.
5. Last resort: assume the model returned only the function body. Prepend the
   original HumanEval prompt and indent the body.

### 4.3 `verifier.py`

Validates a candidate against the HumanEval test for that problem.

Process:

1. Pre-flight: `compile(code, "<string>", "exec")`. If this throws
   `SyntaxError`, return `(False, "SyntaxError: ...")`.
2. Pre-flight: regex-check that the candidate defines the expected
   `entry_point` function (e.g., `def has_close_elements(`). If not, return
   `(False, "entry_point ... not defined")`.
3. Build a single Python script: `code + "\n\n" + test_code + "\n\ncheck(" +
   entry_point + ")\n"`. This is *concatenation only* — no `textwrap.dedent`
   tricks. (See §5.2.)
4. Write the script to a temp file. Run it as a subprocess with a 10-second
   timeout. Capture stdout/stderr.
5. Return `(True, None)` on exit code 0, otherwise `(False, error_message)`.

The subprocess isolation matters: HumanEval models can hallucinate `os.system`
or other side-effecting calls. Running in the main process would corrupt
state. A timeout catches infinite loops.

### 4.4 `benchmark.py`

Loads HumanEval. Tries the official `human_eval` PyPI package first
(`from human_eval.data import read_problems`). Falls back to a local
`HumanEval.jsonl` file if the package isn't installed. Yields tuples of
`(task_id, prompt, test, entry_point)` for the requested number of problems.

### 4.5 `lora_utils.py`

Two helpers for managing LoRA state between calls:

- `zero_lora_weights(model)`: sets every parameter whose name contains
  `lora_A` or `lora_B` to zero. After this, the model's output equals the
  base model's output (since LoRA contribution is `lora_B @ lora_A` and
  `lora_B` being zero makes the whole contribution zero, but zeroing both is
  explicit).
- `reset_lora_weights(model)`: re-initializes `lora_A` with
  `kaiming_uniform_` (the standard PEFT init) and `lora_B` to zero. This
  gives a fresh starting point for TTT — variant N does not inherit any
  weights from variant N-1.

`base_greedy` and `base_temperature` use `zero_lora_weights` before each
generation. `ttt_search` uses `reset_lora_weights` before each variant.

### 4.6 `conditions/base_greedy.py`

Zero LoRA. Format prompt via chat template. Call `model.generate` with
`do_sample=False, max_new_tokens=768`. Decode. Run through `extract_code`.
Return the resulting string. One candidate per problem.

### 4.7 `conditions/base_temperature.py`

Zero LoRA. Format prompt. Loop over four seeds (42, 43, 44, 45). For each
seed: `torch.manual_seed(seed)`, then `model.generate(... do_sample=True,
temperature=0.8, top_p=0.95 ...)`. Extract code. Run the verifier on it. If
it passes, return immediately. After all four, return the first candidate if
none passed.

### 4.8 `conditions/ttt_search.py`

The interesting one. For each of the four priming files in a fixed order
(`iterative.txt`, `recursive.txt`, `edge_cases.txt`, `builtins.txt`):

1. `reset_lora_weights(model)` — fresh LoRA init for this variant.
2. `_warmup(model, tokenizer, priming_text)`:
   - `model.train()`
   - Collect just the LoRA parameters into an AdamW optimizer at LR `1e-4`.
   - Tokenize the priming text (max 512 tokens).
   - 8 forward + backward + step iterations with `labels=input_ids` (standard
     causal-LM loss). The base model is frozen; only LoRA parameters update.
   - `model.eval()`
3. Generate the candidate at `temperature=0.3, top_p=0.95, do_sample=True`,
   seeded with `42 + i`.
4. Extract the code with `extract_code`. Collect.

After all four variants, run each candidate through the verifier. Return the
first one that passes; otherwise return the first one.

A sanity check prints a 400-character preview of each candidate and reports
how many of the four are byte-distinct at that prefix. If all four are
identical the function logs a warning — TTT may not be moving the model.

### 4.9 `run_experiment.py`

The orchestrator. CLI flags: `--num-problems` (default 20), `--seed` (42),
`--conditions` (comma-separated), `--output-dir` (`results`).

Process:

1. `FLA_TILELANG=0` is set as an environment variable at the top of the file
   (before any imports that touch the FLA library). See §5.1.
2. Seeds (Python `random`, NumPy, PyTorch CPU + CUDA) are set from
   `--seed`.
3. The model is loaded once.
4. Priming text SHA-256 hashes are computed for the summary.
5. For each requested condition: seeds are reset (so condition order doesn't
   affect reproducibility), then each problem is run through the condition.
   After every problem, the partial results are flushed to a per-condition
   checkpoint file (`results/checkpoint_<cond>.json`). This was added so an
   8-hour crash doesn't lose everything.
6. After all conditions, write `summary.txt` (per-condition pass rates, the
   hypothesis verdict, hyperparameters, priming-text hashes, runtime) and
   `detailed.json` (one record per problem×condition with a 300-character
   candidate preview).

---

## 5. Bugs that surfaced and how each was fixed

Five real problems hit during development. Each one matters because each
explains a design decision that would look arbitrary without context.

### 5.1 Qwen3.5-0.8B is a vision-language model; calling its tokenizer on plain text crashed

The first run errored deep inside `transformers/models/qwen3_vl/processing_qwen3_vl.py`
with a `ValueError: Incorrect image source` — the tokenizer was trying to
interpret the HumanEval prompt string as a base64-encoded image.

Cause: `unsloth/Qwen3.5-0.8B` is published as a VLM and its
`AutoProcessor`/`from_pretrained` returns a `Qwen3VLProcessor`. When called
like `tokenizer(text, ...)`, the processor's `patched_call` from
`unsloth_zoo` routes the text into the image branch and detonates.

Fix: extract `tokenizer.tokenizer` (the inner text-only Qwen tokenizer) in
`loader.py` and use it everywhere. The processor's `apply_chat_template`
also expects VL content; the inner tokenizer has its own
`apply_chat_template` that works on plain strings.

### 5.2 The verifier's `textwrap.dedent` f-string corrupted multi-line code

After the tokenizer fix, the model generated correct-looking Python and the
verifier still reported `IndentationError: from typing import List` at line
1. Eight problems out of eight failed for the same reason. The model's
output was fine; the verifier was the bug.

Cause: the verifier built its script with:

```python
script = textwrap.dedent(f"""\
    {code}

    {test_code}

    check({entry_point})
""")
```

When `code` is a multi-line string, only the *first* line gets the f-string's
4-space indent. Subsequent lines retain whatever indentation the model
produced. So:

- Line 1 of `code` had 4 leading spaces.
- Lines 2+ of `code` had 0 leading spaces.
- `textwrap.dedent` finds the common leading whitespace across all lines and
  removes that. Common = 0 spaces. Nothing gets stripped.
- Result: line 1 is indented, line 2 is not — `IndentationError`.

Fix: drop `textwrap.dedent` entirely. The code coming out of `extract_code`
is already well-formed; concatenate it directly:

```python
script = code + "\n\n" + test_code + "\n\ncheck(" + entry_point + ")\n"
```

After this change, Phase 1 produced its first PASS.

### 5.3 The first version of `extract_code` took the first code block, but the model self-corrects

This wasn't found during the original Phase 1–4 runs; it was caught during
the validation pass after the experiment. The symptom was failures on
HumanEval/11, 13, 14 where the model output looked like:

```
```python
def string_xor(a, b):
    return a + b      # <-- wrong
```

Wait, that's not correct. Let me fix:

```python
def string_xor(a, b):
    return ''.join('1' if x != y else '0' for x, y in zip(a, b))
```
```

The original `extract_code` matched the first opening fence and tried to
remove the closing fence at the end of the string. That left a trailing
` ``` ` in the middle of the output, plus the wrong implementation. The
verifier saw a `SyntaxError` from the literal ` ``` ` characters.

Fix (`extract_code` v2): split the response on opening-fence markers. For
each non-empty block, strip a possible trailing ` ``` `. If the target
function's name appears in any of those blocks, return the **last** such
block — models self-correct in later blocks. Fall back to the last non-empty
block if no block contains the function name.

This change was applied between the v1 20-problem and v1 164-problem runs.

### 5.4 The TTT warm-up's backward pass crashed: "No CUDA or HIP or MPS available on this system"

Phase 3 (TTT-search on 5 problems) crashed during the first backward call
with a `ValueError` from inside `tilelang/utils/target.py`. Strangely it
said no CUDA was available even though the RTX 3090 was visible to PyTorch.

Cause: Qwen3.5 has a **hybrid attention** architecture. Every fourth layer
is standard scaled-dot-product attention (with `q_proj`, `k_proj`, etc.).
The other three out of four are linear-attention layers implemented in the
`fla` (flash-linear-attention) library using gated delta rules. The backward
pass of those linear-attention layers JIT-compiles a CUDA kernel via the
`tilelang` library. `tilelang` looks for CUDA by trying to call
`tvm.contrib.nvcc.find_cuda_path()`. In this environment:

- `tvm` is not installed.
- `nvcc` is not in `PATH`.

So `find_cuda_path()` raises, `tilelang.utils.target.check_cuda_availability()`
returns `False`, and `determine_target()` concludes that no CUDA is
available — even though PyTorch's CUDA is perfectly fine.

Forward inference doesn't hit this code path. Only the backward through the
gated-delta-rule chunks does, which is why `base_greedy` and `base_temperature`
worked but `ttt_search` blew up.

Fix: `fla.ops.backends` exposes an environment-variable knob:
`FLA_TILELANG=0` disables the TileLang backend. The library then falls
back to a Triton implementation, which compiles and runs fine. The variable
must be set *before* any FLA import. `run_experiment.py` sets it via
`os.environ.setdefault("FLA_TILELANG", "0")` at the top of the file,
before anything else is imported.

This is a real environment cost: the Triton backward is measurably slower
than TileLang would be. We accept the slowdown because the alternative is
installing TVM + nvcc.

### 5.5 The first TTT run reported "all 4 variants identical at 200 chars"

After §5.4 was fixed, TTT ran but the sanity-check log claimed all four
variants were byte-identical at the 200-character preview point. That would
mean warm-up wasn't moving the model. Closer inspection showed the 200-char
preview was entirely *inside the function signature and docstring* — which
all four variants correctly reproduce because they come from the HumanEval
prompt. The actual implementations *did* differ.

Fix: bumped the preview length to 400 characters (enough to get past most
docstrings) and added a per-problem report of how many of the four 400-char
prefixes were distinct. The metric is now meaningful, though the underlying
issue — that 4 variants do often converge to identical code when the LoRA
is small (r=4) and the priming texts don't differ in style enough — remains
real and is reported.

---

## 6. The priming texts

Four `.txt` files of plain Python functions, each about 200–500 tokens.
They are intentionally different in style so a model warmed up on one will
favor different solution patterns:

- `iterative.txt`: `sum_list`, `count_even`, `reverse_string`, `find_max`,
  `flatten`, `running_sum`, `zip_lists`, `char_frequency`. All use `for`
  loops with accumulators.
- `recursive.txt`: `factorial`, `fibonacci`, `sum_digits`, `power`, `depth`,
  `flatten_recursive`, `binary_search`. All recursive with explicit base
  cases.
- `edge_cases.txt`: `safe_divide`, `first_element`, `safe_index`,
  `deduplicate`, `safe_mean`, `clamp`, `rotate_list`, `strip_outer`. All
  begin with empty/None/out-of-range guards.
- `builtins.txt`: `most_common`, `group_by_first_char`, `prefix_sums`,
  `product`, `interleave`, `top_k`, `unique_counts`, `sliding_window_max`.
  All use `collections.Counter`, `itertools`, `functools.reduce`, etc.

Cross-checked against HumanEval problems 0–19 (and later 0–163): none of
the priming-text function names overlap with any HumanEval entry-point
name. The SHA-256 hash of each file is recorded in `results/summary.txt`
so any reproducer can confirm they used the same priming texts:

```
iterative.txt: bea284b0dc35b7aa121a114ae506ad3c7a9419c4128f3a503212214d6ef49047
recursive.txt: 64cd7c4f21dbb7094324e6dca86f3ffcd3be7b4b1af4d5ae203963d3557451bb
edge_cases.txt: a03de9599d9fadeaccb8ec160a2cdeb16ffae59bb0724e1388364474b61a9124
builtins.txt:   e4f3fc97bb875c6769cffe9ff150d064d4a89be9bef022fcd36782ef977a79b7
```

---

## 7. Hyperparameters in v1 (the locked configuration)

```
Model:               unsloth/Qwen3.5-0.8B
Quantization:        4-bit NF4 via bitsandbytes (load_in_4bit=True)
Seed:                42
Sampling seeds:      42, 43, 44, 45 (base_temperature)
                     42+i for variant i (ttt_search)
Max generation:      768 new tokens
LoRA:                r=4, target_modules=[q_proj, v_proj],
                     lora_alpha=8, lora_dropout=0.0, bias="none"
Gradient ckpt:       "unsloth" (mandatory for Qwen3.5 backward)
TTT optimizer:       AdamW, LR 1e-4, no weight decay, no warmup
TTT steps:           8 per priming text
TTT loss target:     priming text itself (next-token prediction)
Priming truncation:  512 tokens
base_temperature:    T=0.8, top_p=0.95, do_sample=True, n=4
ttt_search:          T=0.3, top_p=0.95, do_sample=True, n=4
Verifier timeout:    10 seconds
Environment:         FLA_TILELANG=0
```

These values are not the result of tuning. They are what the design doc
specified, plus the FLA env var and `use_gradient_checkpointing="unsloth"`,
which are environment-forced corrections, not free choices.

---

## 8. How the experiment was actually run

```bash
# Phase 1 — base_greedy on 5 problems
python run_experiment.py --num-problems 5 --conditions base_greedy

# Phase 2 — base_temperature on 5 problems
python run_experiment.py --num-problems 5 --conditions base_temperature

# Phase 3 — ttt_search on 5 problems
python run_experiment.py --num-problems 5 --conditions ttt_search

# Phase 4 — full comparison on 20 problems
python run_experiment.py --num-problems 20

# Full-set comparison on all 164 HumanEval problems
python run_experiment.py --num-problems 164 --output-dir results/full_164
```

Phase 4 ran in 57.9 minutes on an RTX 3090. The full-164 run finished in
roughly 144 minutes — that drop happened after applying the §5.3
`extract_code` fix and small speed wins.

---

## 9. Results

### 20-problem run (Phase 4)

```
Condition              Pass Rate    Pass Count
base_greedy            25.0%        5/20
base_temperature       30.0%        6/20
ttt_search             30.0%        6/20

Hypothesis (ttt_search >= base_temperature + 10pp): NOT SUPPORTED
delta = +0.0pp
```

### Full 164-problem run

```
Condition              Pass Rate    Pass Count
base_greedy            32.9%        54/164
base_temperature       45.1%        74/164
ttt_search             43.3%        71/164

Hypothesis (ttt_search >= base_temperature + 10pp): NOT SUPPORTED
delta = -1.8pp   (ttt_search behind base_temperature)
```

### Paired statistical analysis (164-problem run)

Wilson 95% confidence intervals on each pass rate:

| Condition | Wilson 95% CI |
|---|---|
| base_greedy | 26.2% – 40.4% |
| base_temperature | 37.7% – 52.8% |
| ttt_search | 35.9% – 50.9% |

McNemar paired test on solve disagreement (same 164 problems, both methods):

| Comparison | A-only | B-only | Z | Interpretation |
|---|---|---|---|---|
| greedy vs temperature | 5 | 25 | −3.65 | temperature significantly beats greedy, p < 0.001 |
| greedy vs ttt_search | 4 | 21 | −3.40 | ttt_search significantly beats greedy, p < 0.001 |
| temperature vs ttt_search | 15 | 12 | +0.58 | **no significant difference between the two**, p ≈ 0.56 |

Net: ttt_search is 3 problems behind base_temperature out of 164. Within
noise.

---

## 10. What v1 actually showed

1. **Both stochastic conditions beat greedy by ~10 pp.** Sampling helps;
   that part is firmly significant.
2. **TTT does not beat plain temperature sampling on pass rate.** They
   land within 1.8 percentage points of each other, well inside random
   variation at n=164.
3. **TTT and base sampling do not solve the *same* 71/74 problems.**
   - 12 problems are solved only by ttt_search.
   - 15 problems are solved only by base_temperature.
   - Oracle union = 52.4%, materially higher than either method alone.
   This means the two methods have different solve sets, but v1 cannot
   say *why*. Specifically, v1 cannot rule out the alternative
   explanation that the difference is purely a temperature effect
   (ttt_search uses T=0.3, base_temperature uses T=0.8). The missing
   control — base sampling at T=0.3 — is what v1.1 adds.

By the design doc's stated decision rule, v1's hypothesis is killed. But
the paired data hints that something other than "ttt = base + noise" might
be happening. Whether that something is real priming structure or just a
low-temperature artifact is the open question v1.1 was built to answer.

---

## 11. Known caveats baked into v1's design

These were known going in and are documented here so a future reader does
not retread them:

- **Confound: TTT uses T=0.3, base_temperature uses T=0.8.** If TTT helps,
  v1 alone cannot prove it isn't just the lower temperature doing the work.
  The design doc explicitly flagged this as the first v2 follow-up.
- **Small LoRA.** `r=4` on only `q_proj` and `v_proj` means the warm-up
  has very limited expressive power. About 1 in 8 problems showed all 4
  TTT variants byte-identical at 400 chars in the Phase 4 run. The
  warm-up *is* changing the model on the others, but the signal is weak.
- **Triton backward, not TileLang.** `FLA_TILELANG=0` forces the slower
  Triton path for GatedDeltaRule backward. The run times reported here
  are environment-specific.
- **`max_new_tokens=768` is generous.** Some failures are
  `SyntaxError: unterminated string literal` — generations getting cut
  off before the closing fence. A larger budget might recover those at
  the cost of runtime.

---

## 12. Files produced by a v1 run

```
results/
├── summary.txt              # human-readable summary, hypothesis verdict
├── detailed.json            # per-problem×condition outcomes, candidate previews
└── checkpoint_<cond>.json   # per-condition partial results (resilience)
```

`summary.txt` contains:

- Model HuggingFace tag and resolved version
- All hyperparameters (LoRA, TTT, sampling)
- SHA-256 of every priming text
- Per-condition pass rates and counts
- Hypothesis verdict per the design doc's ±10pp rule
- Total wall-clock runtime
- UTC timestamp

`detailed.json` is the canonical machine-readable output. Each record:

```json
{
  "task_id": "HumanEval/N",
  "condition": "ttt_search",
  "passed": true,
  "error": null,
  "candidate_preview": "first 300 chars of the candidate code"
}
```

v1 only kept 300 characters of the candidate. v1.1 keeps the full text and
also logs every candidate, not just the selected one.

---

## 13. Why this README exists

The single most important thing about v1 is that the conclusion you reach
depends on which controls you ran. v1 ran what the design doc specified,
which produced an apparently-clean kill of the hypothesis. v1.1, with one
extra condition, flipped the verdict (see `v1.1Readme.md`). This file
exists so it stays clear which knobs v1 turned, which it did not, and what
each one would have shown.

The code is not large. The set of details that determine whether you
trust the numbers — model identity, tokenizer wrap, backward-pass kernel,
code-extraction logic, verifier indentation handling, priming-text
hashes — is large. Future runs that change any of those should expect
their numbers not to match this one.
