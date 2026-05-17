# TTT-Search Experiments

A four-experiment research arc on test-time training (TTT) and inference-time
sampling for small LLMs on HumanEval. Built on `unsloth/Qwen3.5-0.8B`,
24 GB RTX 3090, BF16, single GPU. Each experiment either confirmed or killed
a hypothesis, and the final answer ended up being unrelated to where the
project started.

Detailed per-experiment journals:
- [`v1Readme.md`](v1Readme.md) — TTT-search v1: full design, every bug, every fix
- [`v1.1Readme.md`](v1.1Readme.md) — adding the temperature×priming factorial
- [`v1.2Readme.md`](v1.2Readme.md) — the noise-floor control that flipped v1.1's verdict

This top-level README is the executive summary.

---

## 1. The original hypothesis

**Claim (H1):** For coding problems, generating N=4 candidate solutions via
*test-time training* — where each candidate comes from the same base model
warmed up on a *different* short priming text via a brief LoRA adapter
update — solves more problems than generating N=4 candidates via plain
temperature sampling.

Falsification rule (from the v1 design doc): if TTT-search and naive
resampling land within ±10 absolute percentage points of each other on a
20-problem subset of HumanEval, the hypothesis is not supported.

The intuition: priming the model on style-specific code (iterative,
recursive, defensive, builtins-heavy) should bias each variant toward a
different solution approach, expanding the effective search.

---

## 2. The arc, in one table

| Experiment | What it added | Headline result | What it actually concluded |
|---|---|---|---|
| **v1** | 3 conditions on 164 HumanEval problems | `ttt_search` 43.3% vs `base_temperature` 45.1% | Δ = −1.8 pp at n=164. Hypothesis killed by the ±10 pp rule. But TTT and base solve 12/15 different problems, leaving an open question. |
| **v1.1** | Two new conditions to complete a 2×2 temperature × priming factorial; per-variant logging; auto-analyzer with locked-in verdict matrix | At matched temperature, sym diff D = 17 (T=0.3) / D = 34 (T=0.8) between TTT and base | "H1 + H2 HOLD" per the matrix — but the matrix had no noise reference |
| **v1.2** | The missing control: a *second* base-temperature run with seeds 46–49 to measure D_noise | D_ttt ≈ D_noise (17 vs 15 at T=0.3, 34 vs 36 at T=0.8) | Priming is **noise**. v1.1's apparent "real diversity" was just seed shuffle. Mixing methods at n=8 doesn't beat best pure. Verdict: **"Whole direction dead. Just sample more."** |
| **Sampling Dynamics Sweep (N × T)** | 5 temperatures × 16 candidates per problem on the full 164 | T=0.8 at N=16: 65.9%, dominating every budget. T=1.0 is the highest "safe" temperature (compile rate 4.8% vs T=1.2's 11.1%) | **Recipe locked in: sample at T=0.8 for any budget. Curves not plateaued at N=16; more samples still help.** |

---

## 3. v1 — TTT-search vs naive resampling

**What was run** (3 conditions, 164 HumanEval problems):

| Condition | Setup |
|---|---|
| `base_greedy` | T=0, single attempt. Floor. |
| `base_temperature` | T=0.8, top_p=0.95, n=4 candidates (seeds 42–45). Best-of-4 with verifier-based first-pass selection. |
| `ttt_search` | T=0.3, top_p=0.95. For each of 4 priming texts (`iterative`, `recursive`, `edge_cases`, `builtins`): reset LoRA → 8 AdamW steps at LR=1e-4 on the priming text → generate 1 candidate. Best-of-4 selection. |

**LoRA config:** r=4, target_modules=[q_proj, v_proj], lora_alpha=8, lora_dropout=0.0
**Model:** `unsloth/Qwen3.5-0.8B` (BF16, no quantization)
**Verifier:** subprocess with 10s timeout, compile + entry-point pre-flight

**Real bugs discovered and fixed during v1** (in `v1Readme.md` §5):

1. Qwen3.5-0.8B is a vision-language model — its processor crashed on plain text. Fixed by extracting `tokenizer.tokenizer` (inner text-only tokenizer).
2. The verifier's `textwrap.dedent` f-string was corrupting multi-line code (only the first line got the indent prefix). Fixed by concatenating directly.
3. `extract_code` initially picked the *first* fenced code block. Models self-correct ("Wait, let me fix:") in *later* blocks. Fixed by selecting the last block containing the function definition.
4. TTT's backward pass crashed on Qwen3.5's hybrid attention (gated delta rule + TileLang JIT requires `nvcc`/TVM). Fixed by `FLA_TILELANG=0` to fall back to the Triton backend.
5. Initial "all variants identical" sanity check was reading inside the docstring. Fixed by extending the preview to 400 chars (past most docstrings).

**Results:**

```
Condition              Pass Rate    Pass Count
base_greedy            32.9%        54/164
base_temperature       45.1%        74/164
ttt_search             43.3%        71/164
```

By the ±10 pp falsification rule: **hypothesis not supported** (Δ = −1.8 pp, with `ttt_search` *behind* `base_temperature`).

But the paired analysis was suspicious:
- 12 problems only `ttt_search` solves
- 15 problems only `base_temperature` solves
- Oracle union: 52.4%

The two methods solve *overlapping but different* problem sets. v1 cannot say whether that's "priming bites" or just "lower temperature samples differently."

---

## 4. v1.1 — the 2×2 factorial

**Question:** v1 confounded two knobs (LoRA warm-up *and* generation
temperature). v1.1 separates them by completing the factorial:

| | T=0.3 | T=0.8 |
|---|---|---|
| **No priming** (LoRA zeroed) | `base_temp_lowT` 🆕 | `base_temp_highT` (= v1's `base_temperature`) |
| **With priming** (TTT warm-up) | `ttt_lowT` (= v1's `ttt_search`) | `ttt_highT` 🆕 |

Plus per-variant logging (full candidate code, `source_label`, `passed`,
`error`, `generation_seconds` for each of the 4 candidates per problem),
plus a `ttt_search/analysis.py` module that pre-computes pairwise solve
sets, oracle unions, per-source-label attribution, and a locked-in verdict
matrix.

**Performance work:** switched off 4-bit bitsandbytes quantization (the
spec recommended it for 8 GB GPUs; on a 24 GB RTX 3090 the 1.6 GB BF16
model fits comfortably and the dequantization overhead is removed).
Batched the 4 temperature-sampling candidates with `num_return_sequences=4`
for a 2.5× speedup on the temperature conditions. Both pipelines verified
by re-checking that v1's deterministic conditions produced identical
outputs.

**Results (164 problems, 263 min):**

```
Condition         Pass Rate    Pass Count
base_greedy        32.9%       54/164
base_temp_lowT     42.7%       70/164   ← new control
base_temp_highT    45.1%       74/164
ttt_lowT           43.3%       71/164
ttt_highT          46.3%       76/164   ← new
```

Pass-rate deltas are tiny (+1, +2 in TTT's favor). But the **symmetric
differences** are not:

```
At T=0.3:   D_ttt = 17 (only_ttt=9,  only_base=8)
At T=0.8:   D_ttt = 34 (only_ttt=18, only_base=16)
```

By the v1.1 verdict matrix (`D ≥ 10` AND `delta ≥ −2`), the analyzer
declared **"H1 + H2 HOLD: priming creates real diversity."**

Subtlety v1.1 missed: the matrix had no reference point. Two runs of
`base_temp_lowT` with *different seeds* might disagree on 15-ish problems
for free. If they do, the D = 17 finding is sampling variance dressed in
priming clothes.

---

## 5. v1.2 — the noise-floor control

The missing reference. v1.2 generates two seedB conditions:

| New | Setup |
|---|---|
| `base_temp_lowT_seedB` | Identical to v1.1's `base_temp_lowT` except seeds 46–49 instead of 42–45 |
| `base_temp_highT_seedB` | Same idea at T=0.8 |

These are run on the full 164 problems. Same model, same priming-text
hashes (verified), same verifier, same chat template, same everything
except the four sampling seeds.

Then `analyze_v12.py` compares:

```
D_ttt    = sym_diff(ttt_lowT,        base_temp_lowT_seedA)
D_noise  = sym_diff(base_temp_lowT_seedA, base_temp_lowT_seedB)
```

If `D_ttt ≈ D_noise`, priming is doing what seed shuffling already does —
**no signal**.

Decision rule (locked in §4 of the v1.2 design doc, *before* the run):

| Condition | Verdict |
|---|---|
| `D_ttt ≥ D_noise + 8` | priming is real |
| `|D_ttt − D_noise| ≤ 5` | priming is noise |
| `D_ttt ≤ D_noise − 5` | priming reduces diversity |

**Results:**

```
D_ttt_lowT    = 17,   D_noise_lowT  = 15   →  |17 − 15| = 2 ≤ 5  →  priming is noise
D_ttt_highT   = 34,   D_noise_highT = 36   →  |34 − 36| = 2 ≤ 5  →  priming is noise
```

Both temperatures agree. v1.1's "real diversity" finding is **fully
explained by seed variance**. The priming texts were doing nothing
beyond what a different RNG seed would have done for free.

Plus a compute-matched ensemble check at n=8 candidates per problem:

```
pure_lowT_n8       (seedA ∪ seedB at T=0.3)  =  75/164  (45.7%)
pure_highT_n8      (seedA ∪ seedB at T=0.8)  =  90/164  (54.9%)  ← best
mixed_temp_n8      (lowT_seedA ∪ highT_seedA) =  84/164  (51.2%)
mixed_ttt_temp_n8  (ttt_lowT ∪ highT_seedA)   =  87/164  (53.0%)
mixed_ttt_only_n8  (ttt_lowT ∪ ttt_highT)     =  85/164  (51.8%)
```

Mixing temperatures is *worse* than doubling up on T=0.8 alone.

**Final verdict** (`results/v12_full/analysis_v12.txt`):

> **VERDICT: Whole direction dead. Just sample more.**

That is the executive answer to the original hypothesis. TTT-search adds
nothing that better seed selection at the same temperature couldn't do
for free. The follow-up question — *how* should you sample more? — is
what the Sampling Dynamics Sweep answers.

---

## 6. Sampling Dynamics Sweep — N × T surface mapping

**Motivation:** v1.2 closed TTT but left an empirical recipe gap. We had
2 data points on the temperature curve (T=0.3 and T=0.8) at 2 sample
counts (N=4 and N=8). We did not know where the temperature peak is at
N=16, where the N-curve plateaus per temperature, where the compile-fail
"garbage onset" is, or what the compute-optimal (N, T) is for a given
budget.

**Design:** Generate **16 candidates per problem at each of 5
temperatures** {0.3, 0.5, 0.8, 1.0, 1.2}, then derive pass@N for any
N ≤ 16 from the same candidate pool.

To avoid re-generating data v1.1 + v1.2 already produced:

| Temperature | Already have | Add (sweep) | Total |
|---|---|---|---|
| T=0.3 | 8 (v1.1+v1.2) | 8 (seeds 50–57) | 16 |
| T=0.5 | 0 | 16 (seeds 42–57) | 16 |
| T=0.8 | 8 (v1.1+v1.2) | 8 (seeds 50–57) | 16 |
| T=1.0 | 0 | 16 (seeds 42–57) | 16 |
| T=1.2 | 0 | 16 (seeds 42–57) | 16 |

Total new candidates: 10,496. Runtime: ~4 hours.

**Analyzer** (`analyze_sweep.py`) pools candidates by temperature across
v1.1, v1.2, and sweep `detailed.json` files, then computes:

- Empirical pass@N (first-N candidates, any-pass)
- Unbiased pass@N (HumanEval-paper estimator: `1 − C(M−c, n) / C(M, n)`)
- Compile-failure rate per T
- Mean pairwise Levenshtein per T (with `python-Levenshtein` C extension)
- Saturation analysis (where each curve flattens)
- Compute-optimal (N, T) table for budgets ∈ {4, 8, 16}

**Results — pass@N (empirical, first-N candidates, n=164):**

| N | T=0.3 | T=0.5 | T=0.8 | T=1.0 | T=1.2 |
|---|---|---|---|---|---|
| 1 | 31.7% | 29.3% | 26.2% | 19.5% | 19.5% |
| 2 | 38.4% | 37.2% | 33.5% | 27.4% | 25.6% |
| 4 | 42.7% | 44.5% | **45.1%** | 36.0% | 30.5% |
| 8 | 45.7% | 51.8% | **54.9%** | 48.8% | 41.5% |
| 12 | 49.4% | 57.3% | **61.0%** | 54.9% | 47.6% |
| 16 | 51.8% | 59.8% | **65.9%** | 59.8% | 48.8% |

**Compile-failure rate** (Q3 — "garbage onset"):

```
T=0.3:  3.32%   OK
T=0.5:  2.40%   OK
T=0.8:  3.51%   OK
T=1.0:  4.80%   OK     ← highest safe temperature
T=1.2: 11.13%   GARBAGE
```

**Within-pool diversity** (mean pairwise normalized Levenshtein):

```
T=0.3: 0.183   T=0.5: 0.226   T=0.8: 0.298   T=1.0: 0.346   T=1.2: 0.453
```

Diversity climbs monotonically with T; pass rate does not. The sweet
spot is T=0.8 — enough sampling spread to find different solutions
without producing broken Python.

**Q1–Q4 final answers:**

```
Q1 (Temperature peak at N=16):     T=0.8, pass 108/164 (65.9%)
Q2 (Plateau per T):                 None of the five curves plateaued at N=16
                                    (pass@16 − pass@8 ≥ 10 for every T)
Q3 (Highest safe temperature):      T=1.0 (compile rate 4.8%)
Q4 (Compute-optimal):
    Budget  4: (N=4,  T=0.8) → 74/164  (45.1%)
    Budget  8: (N=8,  T=0.8) → 90/164  (54.9%)
    Budget 16: (N=16, T=0.8) → 108/164 (65.9%)
```

**Practical recipe (the deliverable):**

> For best-of-K on Qwen3.5-0.8B + HumanEval: sample K candidates at
> `T=0.8`, `top_p=0.95`, then run them through a verifier and pick the
> first that passes. Expected pass rates: 45% at K=4, 55% at K=8, 66% at
> K=16. Curves don't plateau within K ≤ 16 — bigger K keeps helping.

Plots: `results/sweep/sweep_plots/plot_passN_curves.png` and
`plot_surface.png`.

---

## 7. The final picture, in one paragraph

Started by asking whether priming a small LoRA scratchpad on style-themed
code (`iterative.txt`, `recursive.txt`, …) makes a verifier-guided
best-of-N system better. After three experiments controlling progressively
tighter, the answer is: priming does nothing that random seed variation
already does for free, and once that's known, the actually-useful
follow-up question — "OK so what *does* matter?" — gets a clean two-knob
answer: temperature 0.8 and as much N as you can afford.

---

## 8. Layout

```
ttt-search-experiments/
├── README.md                    # this file
├── v1Readme.md                  # full v1 journal (design, bugs, results)
├── v1.1Readme.md                # v1.1 journal: factorial + per-variant logging
├── v1.2Readme.md                # v1.2 journal: noise floor + ensemble check
├── requirements.txt
├── run_experiment.py            # one orchestrator for all conditions
├── analyze_v12.py               # v1.2 Q1+Q2 analyzer (verdict line)
├── analyze_sweep.py             # sweep N×T analyzer (pass@N table, Q1-Q4)
├── ttt_search/
│   ├── loader.py                # FastLanguageModel + inner text tokenizer
│   ├── prompt_utils.py          # chat template formatting, robust extract_code
│   ├── verifier.py              # subprocess sandbox, compile + entry-point pre-flight
│   ├── benchmark.py             # HumanEval loader
│   ├── lora_utils.py            # zero / reset LoRA weights between conditions
│   ├── analysis.py              # v1.1 analyzer (pairwise solve-sets, verdict matrix)
│   └── conditions/
│       ├── base_greedy.py
│       ├── base_temperature.py  # parameterized by temperature, seeds
│       └── ttt_search.py        # parameterized by generation_temperature
├── priming/                     # four priming texts (hash-locked from v1)
│   ├── iterative.txt
│   ├── recursive.txt
│   ├── edge_cases.txt
│   └── builtins.txt
└── results/
    ├── full_164/                # v1 final run (3 conditions × 164)
    ├── v11_full/                # v1.1 final run (5 conditions × 164)
    ├── v12_full/                # v1.2 final run (2 seedB conditions × 164)
    └── sweep/                   # sampling-dynamics sweep (5 conditions × 164)
        ├── detailed.json
        ├── sweep_analysis.txt
        ├── checkpoints/
        └── sweep_plots/         # the two PNGs
```

---

## 9. Reproduction

```bash
# Install
pip install -r requirements.txt
pip install python-Levenshtein   # for analyze_sweep.py diversity metric

# v1 (3 conditions on 164, ~2.4 hours)
python run_experiment.py --num-problems 164 --output-dir results/full_164 \
    --conditions base_greedy,base_temp_highT,ttt_lowT

# v1.1 (all 5 conditions on 164, ~4.4 hours)
python run_experiment.py --num-problems 164 --output-dir results/v11_full

# v1.2 (2 noise-floor conditions on 164, ~2 hours) + analyzer
python run_experiment.py --num-problems 164 --output-dir results/v12_full \
    --conditions base_temp_lowT_seedB,base_temp_highT_seedB --skip-analysis
python analyze_v12.py

# Sampling Dynamics Sweep (5 sweep conditions on 164, ~4 hours) + analyzer
python run_experiment.py --num-problems 164 --output-dir results/sweep \
    --conditions sweep_T03,sweep_T05,sweep_T08,sweep_T10,sweep_T12 --skip-analysis
python analyze_sweep.py
```

Environment notes (per `v1Readme.md` §5):

- BF16, no quantization (`load_in_4bit=False`, `dtype=torch.bfloat16`)
- `use_gradient_checkpointing="unsloth"` (required for Qwen3.5's hybrid attention backward)
- `FLA_TILELANG=0` (Triton fallback because `nvcc`/TVM not available)
- Inner text tokenizer: `tokenizer.tokenizer if hasattr(tokenizer, "tokenizer") else tokenizer`

These were not free choices; they are environment corrections documented
in the v1 journal.

---

## 10. What generalizes and what doesn't

These findings are specific to **Qwen3.5-0.8B on HumanEval**. They
generalize tentatively to *"small instruct-tuned LLMs on Python-coding
benchmarks with strict verifier-based correctness"*. They do **not**
necessarily generalize to:

- Larger models (the T=0.8 sweet spot likely moves lower as model capability rises)
- Tasks without verifiers (best-of-N needs a selection rule)
- Non-coding tasks (the answer-space topology is different)
- Models with different chat templates or thinking modes

When applying these recipes elsewhere, treat them as starting hypotheses
and re-run a small slice of the sweep on the new (model, benchmark)
pair before committing.
