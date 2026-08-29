# 0001. Local Executive / Orchestrator model for the Moeen adaptation

- **Status:** Accepted
- **Date:** 2026-08-29
- **Deciders:** Repository owner (takhlesbh)
- **Scope:** Selects the local Executive/Orchestrator model for the planned
  **Moeen** adaptation of Open Executive. Nothing else.

## Scope

Open Executive is the host/foundation codebase. Moeen is a planned adaptation
built on it, not yet represented as a named product, module, or configuration
anywhere in this repository. This ADR records a model-selection decision **for
the Moeen adaptation**; it is not a statement by or about upstream Open
Executive, whose Executive and eight specialists remain on the Anthropic Claude
API with the prompt-caching contract in `prompts/cache_manager.py` untouched.

## Decision

**Qwen3.5 9.65B is the current LOCAL EXECUTIVE / ORCHESTRATOR selection for the
Moeen adaptation**, on a 16 GB Apple M4 development machine.

It was selected on four measured grounds:

1. **Evidence discipline** — in the investment test it restated only what the
   submission and market data stated, keeping judgment in the ASSESSMENT section.
2. **Zero unsupported or derived numbers in the investment test** — every figure
   in its output traces to an input. It produced none of its own.
3. **Strict JSON reliability** — parseable on the first attempt, no markdown
   fence, top-level keys exactly `claims`, `conflicts`, `missing_evidence`.
4. **Correct tool calling and correct multi-turn tool state** — well-formed call
   with correct arguments, faithful consumption of the injected result, and the
   value still held correctly two turns later.

The Executive seat selects for trustworthiness under structure, not for
reasoning power. A model that reasons impressively but invents numbers is worse
here than one that reasons plainly and declines to derive.

## This ADR Does Not Authorize or Establish

- **No provider, runtime, or application integration.** No code was written or
  changed. Wiring a local model into any runtime path is a separate decision
  requiring its own ADR, provider work, and tests.
- **No change to upstream Open Executive's default model policy.**
- **Qwen3.5 as the best reasoning model** — it failed the scheduling test, and
  failed it differently on different runs.
- **Qwen3.5 as a financial calculation authority** — no model selected here is
  one; see below.
- **Qwen3.5 as the future specialist model** — separate criteria, separate
  decision; Ministral 3 is a live candidate for that seat.
- **Qwen3.5 as the final production or server model** — this is a local
  development selection. Server and production policy is undecided.

## Architectural Consequence — Calculation Authority

**LLM OUTPUT IS NOT CALCULATION AUTHORITY.**

This rule outlives the model choice that produced it and binds regardless of
which model occupies the Executive seat.

Any material computed investment figure must be produced by a **deterministic
calculation/tool layer with explicit inputs and provenance** — inputs used,
formula applied, and computing component must all be recoverable from the
record. This covers, where applicable: **IRR, NPV, payback, margins, percentage
gaps, scenario calculations, financial sensitivities, and reconciliations.**

**The Executive may:** route work; identify conflicts; identify missing
evidence; request calculations; interpret tool results; compare scenarios;
synthesize evidence.

**The Executive must not** be treated as the authoritative source of computed
financial figures.

Neither model tested qualifies as one. The rejected model produced a monthly
revenue figure **11.80x** the correct value at a controlled temperature, in
fluent prose, inside an otherwise competent analysis, with nothing signalling
the error — and at a lower temperature it was closer but **still** wrong.
Lowering temperature reduces error magnitude; it does not confer authority. The
selected model earns no exemption either: it won the investment test by
**declining to derive**, not by computing correctly, and its arithmetic was
wrong on every run where it attempted any. The rule protects against the seat,
not against a particular occupant.

## Model and Runtime Identity

| | Selected | Rejected for this seat |
| --- | --- | --- |
| Model | Qwen3.5 9.65B | Ministral 3 8B Instruct 2512 |
| Params | 9,653,104,368 (`general.parameter_count`) | 8.49 B |
| Quant | GGUF v3, `file_type 15` (Q4_K_M), arch `qwen35` | GGUF V3, `file_type 15` (Q4_K-M), arch `mistral3`, 4.89 BPW |
| Provenance | Ollama registry `qwen3.5:latest`, ID `6488c96fa5fa` | Unsloth (`general.quantized_by`), `repo_url https://huggingface.co/unsloth`, imatrix 238 entries / 82 chunks |
| Weights hash | model blob `sha256:dec52a44569a2a25341c4e4d3fee25846eed4f6f0b936278e3a3c900bb99d37c` (6,594,462,816 B); manifest config `sha256:be595b49fe22012bd1f5605ec14c7ffa58331783a88a4fd8c22e5fc8ec42cf9f` | file `sha256:5dbc3647eb563b9f8d3c70ec3d906cce84b86bb35c5e0b8a36e7df3937ab7174` (5,198,386,720 B) |
| Runtime | Ollama 0.32.5, native `/api/chat` | `llama-server` build `1 (b4d6c7d8f)`, AppleClang 21.0.0.21000099, Darwin arm64 (bundled with Ollama 0.32.5), OpenAI-compatible `/v1/chat/completions` |
| Arch notes | 32 blocks, embd 4096, ff 12288, 16 heads, k/v len 256, `full_attention_interval 4`, hybrid `ssm.*` stack, 27-block vision tower, ctx_train 262144 | 34 blocks, embd 4096, ff 14336, GQA 32/8, k/v len 128, **no sliding window** (`n_swa 0`), vocab 131072 (tekken), YaRN x16 over orig ctx 16384, ctx_train 262144 |

**Hardware:** Apple M4, 16384 MiB unified; Metal device MTL0 reporting 12124 MiB
(12123 MiB free). Ministral load: 35/35 layers offloaded, 4950.05 MiB model +
2176.00 MiB KV (f16) + 116.01 MiB compute on Metal, `n_slots = 1`.

**Sampling — identical across both arms:** `temperature 0.3`, `top_p 0.95`,
`presence_penalty 0.0`, `max output 2048`, `context 16384`, concurrency 1. A
supplementary Ministral-only run at `temperature 0.05` covered T1/T3/T4.
Qwen was run with `think=false` (see weakness 4 below).

## Benchmark Results

Ground truth: **T1** = 182.25 (A 60, B 47.25, C 75). **T2** = 300,000 before /
408,000 after / +108,000, closure beneficial. **T3-T4** = 6,400 x 85% = 5,440
positions x SAR 145/mo = **SAR 788,800/mo (SAR 9,465,600/yr)** against SAR
11,500,000/yr opex; planted conflicts 9,000-vs-6,400 positions and an 85%-vs-61%
gap of **24 percentage points**. **T5** = 63. **T6** = 61, gap 24 pp.

| Test | Qwen3.5 | Ministral 3 |
| --- | --- | --- |
| T1 scheduling | **128.25** (t=0.3); **180** on an earlier run — both wrong, different failure modes | **537** (t=0.3); **121.25** (t=0.05) — both wrong |
| T2 Arabic | `finish=length` — hit the 2048-token cap, truncated mid-figure at `408,`; 123.15 s; no final answer | `finish=stop`, 824 tok, 44.88 s; **correct** 300,000 -> 408,000, +108,000 |
| T3 investment | 551 tok; **zero derived figures**; both conflicts flagged; requested a 61%-occupancy sensitivity | t=0.3: **"SAR 9,306,000/month"** vs true 788,800 = **11.80x**; "14% above" for a 24-pt gap. t=0.05: 788,600/mo and 9,463,200/yr — near-correct route and conclusion, still arithmetically wrong |
| T4 strict JSON | `parsed=true`, `fence=false`, keys exactly `[claims, conflicts, missing_evidence]` | `parsed=false`, **fenced** with ` ```json ` at **both** t=0.3 and t=0.05 |
| T5 tool round trip | correct call + correct 63 | correct call + correct 63 |
| T6 multi-turn state | correct 61, correct **24 percentage points** | correct 61, correct **24 percentage points** |
| Mean decode | 16.55 t/s (16.02-17.05); 3,377 tok / 219.7 s | **19.40 t/s** (18.54-20.31); 3,169 tok / 178.1 s — **+17.2%** |
| Memory pressure | level **1 -> 2** from T4 on; compressor **2,509 -> 3,307 MiB**, rising monotonically; free 61-150 MiB | level **1 throughout**; compressor **297-301 MiB, flat**; free 134-266 MiB |

T3 and T4 together are decisive: the Executive is a component other components
parse, and a fenced response is a parse failure in the orchestrator seat. T5/T6
are recorded as a qualification both models met, not as a differentiator.

## Ministral 3 — Genuine Advantages

Real, measured, and recorded so this decision can be revisited honestly.
Ministral beat Qwen3.5 on every dimension here.

- **Better Arabic result** — complete, correct, terminated answer in 824 tokens
  where Qwen produced no final answer at all. The clearest single-test win in
  the comparison, and it went to the model that was not selected.
- **~17% faster decode** — 19.40 vs 16.55 t/s mean, with no run slower than
  Qwen's fastest.
- **Better memory-pressure behaviour** — pressure level 1 throughout, compressor
  flat at ~300 MiB against Qwen's climb to 3,307 MiB.
- **No truncation** — every generation terminated on its own.
- **Materially improved investment analysis at t=0.05** — reached the correct
  qualitative conclusion (revenue falls short of opex) by a defensible route,
  where at t=0.3 it reached the opposite impression via fabricated figures.

## Ministral 3 — Blocking Weaknesses for the Executive Seat

- **Fabricated / incorrect derived investment figures at a controlled
  temperature**, unhedged and unflagged.
- **Revenue calculation error of ~11.8x** — asserted SAR 9,306,000/month against
  a true 788,800, then reasoned onward from the inflated figure. At t=0.05 the
  error shrank but persisted (788,600 vs 788,800; 9,463,200 vs 9,465,600), and
  it still described a 24-percentage-point gap as "24% above".
- **Deterministic JSON code fencing** — reproducible at both temperatures, which
  makes it worse than stochastic: a systematic violation of the one output
  contract the orchestrator seat must honour.
- **Failed scheduling reasoning** — 537 and 121.25 against 182.25; the t=0.3
  answer mixes units, treating minutes as output.

**Disposition: retained as a possible future specialist candidate; NOT selected
as Executive in this decision.** Its advantages are exactly what a bounded
specialist needs — one that is fenced by its caller and never asked to compute
trips neither disqualifying wire. No specialist decision is made here.

## Qwen3.5 — Weaknesses

The known costs of this selection.

1. **Failed scheduling arithmetic** — 128.25 against 182.25, with two of three
   subtotals wrong (A computed as 36 via a spurious "5 x 36 = 180 minutes"; C as
   45).
2. **Unstable numerical answer across runs** — an earlier run of the identical
   verbatim prompt returned **180** with a *different* error profile (A and C
   correct, B wrong via 5 h instead of 5.25 h). Two runs, two wrong answers, two
   failure modes. The instability disqualifies it for calculation duty as much
   as the errors do.
3. **Arabic truncation and self-correction** — ran to the token cap without
   terminating, cut off mid-figure, after repeatedly re-deriving the same
   quantity and interrogating its own logic rather than converging. It was on
   the correct track when the cap ended it, which is the problem: correctness
   that does not terminate is unusable.
4. **`think=true` over-deliberation risk** — on the same scheduling prompt with
   thinking enabled, **non-terminating deliberation**: `done_reason=length` with
   **empty content** at 2048 tokens (119.14 s) and again at a raised 6144-token
   cap on three attempts (~360 s each), burning 18,451-21,556 characters of
   thinking and emitting nothing. The same model with `think=false` answered in
   291 tokens / 17.09 s. Prompt-specific rather than universal — other prompts
   terminated normally with thinking on — but a live hazard, and the reason the
   comparison ran with `think=false`.
5. **Higher memory pressure than Ministral** — level 2 from T4 onward,
   compressor to 3,307 MiB. On a 16 GB machine this is the constraint most
   likely to force a revisit.
6. **T4 semantic claim-taxonomy looseness** — the JSON was structurally perfect
   but the labelling was not: it typed the promoter's own "85% utilisation
   within 12 months" statement as `derived` when it is a `source_fact`, and
   emitted "Revenue assumption of SAR 145 ... **is valid**" as a `source_fact`
   with `supported: true` — the source states the assumption, it does not
   establish its validity.

## Limitations of the Evidence

- **Prompt comparability.** Only `p1.txt` (T1) survived verbatim from the
  original Qwen smoke battery; that harness preserved responses, not prompts.
  **T2 through T6 were reconstructed** from the preserved Qwen response bodies
  and recorded observations of the planted facts and pass criteria.
  Reconstruction cannot be byte-identical. **Mitigation: both models were re-run
  on the identical reconstructed prompts** under identical sampling, so every
  head-to-head above is internally valid. What cannot be claimed is continuity
  with the original run's absolute numbers; where an original-run figure is
  cited (T1 = 180, the `think=true` non-termination) it is used only as a
  drift/variance signal, never as a head-to-head data point.
- **Serving-stack difference.** Ollama's native API for Qwen against
  llama-server's OpenAI-compatible endpoint for Ministral. Sampling was matched,
  but templating, tokenization, and default-parameter handling differ — prompt
  token counts differ substantially as a result (T1: 201 vs 719 tokens). **Some
  portion of the 17.2% throughput delta may be stack rather than model.**
- **Single run per model per test at t=0.3.** The T1 instability surfaced only
  because an earlier independent run existed; per-test variance is otherwise
  unmeasured. The t=0.05 supplementary run is Ministral-only and cannot support
  a head-to-head claim.
- **Swap totals are not comparable across arms** (3072 MB vs 4096 MB backing
  file); the compressor figure is the reliable memory signal.
- Both models are 4-bit quantized; findings do not transfer to
  higher-precision weights of either.
- A `GGML_ASSERT` abort in `ggml-metal-device.m:622` occurred during
  interrupt-driven llama-server shutdown after the suite completed — a teardown
  artifact; all results were written to disk beforehand.

## Consequences

- Moeen development targets Qwen3.5 9.65B in the local Executive/Orchestrator
  seat. No integration work is authorized by this ADR.
- The calculation-authority rule is binding on the Moeen architecture from this
  point. Designing the deterministic calculation layer is **prerequisite** work,
  not follow-up work, for any Moeen feature surfacing a material computed
  figure.
- The Executive must be prompted and wrapped on the assumption that it will get
  arithmetic wrong; prompts should route computation out rather than invite it.
- Memory pressure level 2 is the operating envelope on a 16 GB M4. Substantial
  concurrent workloads are a risk to the local development loop.
- `think=true` is not enabled on the Executive path without a per-prompt
  termination guarantee and a hard token cap.
- Upstream Open Executive is unaffected.
- DeepSeek-R1 14B was rejected earlier at feasibility-audit stage on memory
  grounds for this hardware and never reached behavioural testing.

## Revisit Criteria

1. Qwen3.5's memory pressure prevents normal concurrent development work, or a
   pressure-driven failure occurs in ordinary use.
2. The `think=true` non-termination behaviour appears on an Executive-path
   prompt with thinking disabled, or a runtime update changes thinking-mode
   behaviour.
3. A Ministral 3 revision, quantization, or grammar-constrained decoding setup
   resolves the JSON fencing violation — constrained decoding would neutralize
   the single most disqualifying finding against it, and its speed and memory
   advantages are real.
4. Target hardware gains unified memory, relaxing the constraint that eliminated
   DeepSeek-R1 14B.
5. The deterministic calculation layer ships, lowering the cost of an Executive
   that computes badly and shifting weight toward routing, evidence discipline,
   and structured output.
6. A repeat-run study measures per-test variance, or the original T2-T6 prompts
   are recovered — either would strengthen or overturn the comparability basis
   above.
