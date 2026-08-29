# DECIDE-section training plan

Goal: distill the NPC decision policy (`Brain.decide()` in `sim/brain.py`) into a
small local model the game can run as its brain — faster and lighter than the
9.6 GB `gemma4` teacher, while behaving sensibly.

Status legend: ✅ done · 🔜 ready to run · ⏸ blocked on a decision · 📝 plan only

---

## 0. The data reality (why we regenerate)

The pre-existing corpus (`extract_dataset.py` over `runs/*.jsonl`, 1718 rows) is
**not usable** for the current game:

- **96% old schema** — `idle` + 3 actions, no `say`/`recall`/`reason`. The live
  prompt has 6 actions and a required `reason`.
- **97% `move`**, 0 `interact`/`use`/`recall`.

So step 1 regenerates a clean, schema-current set. ✅ pipeline built.

---

## 1. Data generation (distillation)  ✅ built · ⏸ teacher choice

`train/gen_scenarios.py` synthesizes diverse *situations* and runs each through
the **real** `Brain.decide()` against a teacher, logging a normal run file.
`train/extract_dataset.py` then produces `train/data/sft.jsonl`.

```bash
# local teacher (free, but gemma4 is degenerate — see below)
python train/gen_scenarios.py --n 400 --teacher ollama

# Claude teacher (recommended — needs: pip install anthropic + ANTHROPIC_API_KEY)
python train/gen_scenarios.py --n 800 --teacher claude
python train/extract_dataset.py --out train/data/sft.jsonl
```

**Critical finding:** `gemma4` chose `none` for 40/40 situations (incl. HP 45,
approaching agent). It is an always-idle policy under this prompt — useless as a
teacher. **Use `claude-sonnet-5`** (or `claude-haiku-4-5` to save ~half). Cost is
trivial: ~$0.65–$1.30 per 400 scenarios.

We balance **situations, not action labels** — a faithful survival-only teacher
*should* pick `move`/`none` most of the time; forcing uniform actions would teach
it to act without reason. `use` is intentionally absent (snapshot exposes no
inventory).

Target for a first real train: **800–1500 scenarios**, then inspect the action
and situation distribution before training.

---

## 2. Environment setup (GPU)  📝 plan only (user chose "not yet")

Current `torch` is **CPU-only** (`2.8.0+cpu`). To train on the RTX 3080 Ti (12 GB):

```bash
pip uninstall -y torch
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install "transformers>=4.57" peft trl datasets accelerate bitsandbytes
python -c "import torch; print(torch.cuda.is_available())"   # must print True
```

`bitsandbytes` on Windows: recent versions ship Windows wheels; if it fails, the
fallback is WSL2 or training in fp16 without 4-bit (a 3B model in fp16 LoRA still
fits 12 GB at short sequence lengths).

---

## 3. Student model + LoRA  📝

**Base: `Qwen2.5-3B-Instruct`** — strong at structured JSON, ~6 GB in 4-bit, fast
inference, comfortable in 12 GB with LoRA. (Alternatives: `Qwen2.5-1.5B-Instruct`
for speed, `google/gemma-2-2b-it` to stay in the teacher's family.)

LoRA config (starting point):

- 4-bit NF4 quant (`bitsandbytes`), bf16 compute
- LoRA `r=16`, `alpha=32`, `dropout=0.05`, targets = attention + MLP proj layers
- seq len 1024 (prompts are short), batch 8 + grad-accum to ~32 effective
- lr 2e-4 cosine, 2–3 epochs, warmup 3%
- mask the prompt; train only on the assistant JSON tokens (completion-only)

The dataset is chat triples `{system, user, assistant}` — apply the student's chat
template; the assistant target is the canonical action JSON (with `reason`).

`train/train_lora.py` (to be written when GPU stack is in) will use TRL
`SFTTrainer` with `DataCollatorForCompletionOnlyLM`.

---

## 4. Eval harness  📝

Behavior, not just loss. Hold out ~10% of scenarios and measure:

- **JSON validity rate** — `validate_intent()` accepts the raw output.
- **Action-match / situation sanity** — e.g. wounded-with-threat → not `none`;
  safe-idle → `none`. Reuse the situation families as labeled probes.
- **Agreement with teacher** on the held-out set.

A cheap first gate: generate on 50 held-out prompts, run each through
`validate_intent`, print the action distribution vs the teacher's.

---

## 5. Serve the student in-game  📝

Two options:

1. **Export to GGUF → Ollama** (matches current `OllamaProvider`, zero game
   changes): merge LoRA into the base, convert with `llama.cpp`
   `convert_hf_to_gguf.py`, `quantize` to Q4_K_M, write a `Modelfile`, `ollama
   create npc-decide`. Point `config.json` `"model"` at it.
2. **Direct HF inference provider** — add a `TransformersProvider` implementing
   `chat_json`; heavier to keep resident but no conversion step.

Recommend option 1 for parity with the existing stack.

---

## Immediate next actions

1. ⏸ **Decide the teacher** (Sonnet 5 recommended) and set up `anthropic` + key.
2. 🔜 Generate 800–1500 scenarios with that teacher; inspect distribution.
3. 📝 When ready to train: do §2 (GPU stack), then §3 script, §4 eval, §5 serve.
