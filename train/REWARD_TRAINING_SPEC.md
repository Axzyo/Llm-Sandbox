# Reward-based training — build spec

Implementation spec for a **drive-conditioned survival policy** trained by
**expert iteration** on an objective, world-computed reward. Hand this to the
implementer (Fable). Read `CLAUDE.md` (the six rules) and `DESIGN.md` (philosophy)
first — they are binding. This document records design decisions already made in
discussion so you don't re-litigate them; confirm the "Open decisions" at the end
before large work.

## 0. Goal in one paragraph

Train ONE small local model — a **drive-conditioned policy** `π(goal-set | state,
drives)` — that, given an NPC's drive weights in its context, makes decisions that
maximize that NPC's drive-weighted reward. The reward is computed by CODE from
measurable world outcomes (survival = meters stay high; curiosity = new things
learned; later: power = possessions/strength). No second/judge LLM — the world is
the judge. Training method is **expert iteration** (run episodes → keep the
high-reward trajectories → fine-tune to imitate them → repeat), NOT PPO to start.

## 1. Design decisions (fixed — do not redesign)

- **Drives are both input and reward weights.** `Entity.drives` (e.g.
  `{"survival":1.0,"curiosity":0.5}`) already appears in the snapshot the policy
  reads. The SAME weights weight the reward: `reward = Σ_d drives[d]·signal_d`. One
  model conditions on drives; per-NPC reward is the training signal. **Never train
  per-NPC** — spawn a *spread* of drive profiles so the one model generalizes.
- **Reward = drive-weighted sum of per-drive OUTCOME signals.** Each drive owns one
  function `(npc, world) -> float` that MEASURES a fact (meters, novelty gained),
  never judges the action's intent (rule 1). An action credits whatever drives its
  *outcomes* served — no action classification, ever.
- **No judge LLM.** Survival/novelty are objective and code-measurable.
- **Expert iteration, not PPO** (rule 4 — simplest that works). Escalate only if it
  plateaus.
- **Reuse the existing sim** (rule 2). The game already runs perception → think →
  goal execution → needs/death, and already logs `prompt`/`response` pairs
  (`sim/journal.py`; `train/extract_dataset.py` pairs them into goal-sets). Build
  the runner on that, don't fork it.

## 2. Existing systems to build ON (read these)

- `sim/entities.py` — `Entity` has `hp`, `hunger`, `thirst`, `inventory`, `drives`,
  `goals`.
- `sim/needs.py` — `tick_needs(entity, dt)`: hunger/thirst drain; health drains at
  empty, regens when both > `REGEN_THRESHOLD`; 0 hp handled by death.
- `sim/terrain.py` — `build_test_map`, `place_resources`, `tick_resources`,
  `interact_with`, `use_item`, `ITEMS`. Effects are DATA on objects/items.
- `sim/brain.py` — `Brain.decide(snapshot) -> list[Goal]`; `build_context`
  (data-only, adds `familiar` flags); `filter_response`/`validate_goals`/
  `goals_to_obj` (strict I/O); `_familiar_types()`; episodic `MemoryStore` +
  `SpatialMemory`. The policy is the `provider` the Brain calls
  (`provider.chat_json(system, user)`).
- `main.py` — the real loop: `build_snapshot`, perception (`PerceptionTracker`,
  `perceive_tiles`, `maintain_spatial`), `advance_goals`, `reap_dead`,
  `think_worker`/`think_q`/`result_q`. It is pygame-coupled — see task B.
- `train/extract_dataset.py` — turns run logs into goal-set SFT rows
  (`{system,user,assistant}`). Reuse/extend for reward filtering.
- `sim/provider.py` — `OllamaProvider.chat_json`. gemma4 already emits valid
  goal-sets under stakes (proven).

## 3. Components to build

### A. Reward module — `sim/reward.py`
- Per-drive component fns `(npc, world) -> float`, and a registry:
  ```python
  REWARD_COMPONENTS = {"survival": survival_reward, "curiosity": curiosity_reward}
  def reward(npc, world) -> float:
      return sum(npc.drives.get(d, 0.0) * fn(npc, world) for d, fn in REWARD_COMPONENTS.items())
  ```
- `survival_reward` = `min(hp, hunger, thirst)/100` while alive (only as safe as the
  worst meter). Death ends reward accrual → living longer + topped-off = more return.
- `curiosity_reward` = **novelty gained this step**, measured from memory: count of
  types newly become-familiar (`Brain._familiar_types` grew) and/or novel memories
  written. A discrete "discovery" signal; do NOT reward raw tiles-stepped-on
  (gameable). Requires the runner to track the before/after familiar-set per NPC.
- Leave a commented `# "power": power_reward` seam. Do NOT build power now.
- Constants (weights inside signals, discount) are module-level, minimal (rule 4).
- Tests: a topped-off NPC scores ~1·survival; a starving one ~0; learning a new
  type yields curiosity > 0; a `{survival:1,curiosity:0}` NPC's reward ignores
  novelty.

### B. Headless sim step — refactor `main.py`
- Extract the per-tick sim logic (needs, resources, perception, thinking dispatch,
  goal execution, death) OUT of the pygame loop into a reusable `sim/engine.py`
  (or `step_world(...)`), so BOTH `main.py` (rendered) and the episode runner
  (headless) call it. This is rule-2 reuse, not a fork. Keep `main.py` behavior
  identical; smoke-test still `SMOKE OK`.
- The think dispatch currently uses a background thread + queues. For headless
  batch rollouts a synchronous think is simpler and reproducible — allow the engine
  to think synchronously (call `brain.decide` inline) so episodes are deterministic
  given a seed.

### C. Episode runner — `train/run_episodes.py`
- Run N headless episodes. Each: fresh map + `place_resources(rng)`, spawn K NPCs
  with **randomized drive weights** (task also: a drive-sampler), each with a Brain
  whose `provider` is the current policy (round 0 = gemma4).
- Step the engine to a time budget or until all NPCs dead. Each step, for each
  living NPC accumulate discounted `reward(npc, world)`; track its familiar-set
  delta for curiosity.
- Log, per NPC: its drives, its full trajectory of `(context, goal-set)` decisions
  (the journal already emits these — attach the NPC's return), and its total
  return + survival time + cause of death.
- Output: run logs (reuse the journal format) + a per-NPC score index.

### D. Reward-filtered dataset — extend `train/extract_dataset.py`
- Same pairing/goal-set logic, but keep only decisions from **high-return
  trajectories**. Because returns differ in scale across drive profiles, normalize
  PER PROFILE: bucket NPCs by (rounded) drive profile and keep the top quantile
  within each bucket (so you keep *good survivalists* AND *good explorers*, not
  just whichever profile scores higher numerically).
- Emit the usual `{system,user,assistant}` rows (assistant = `goals_to_obj`).

### E. Training + GPU stack
- Install: CUDA `torch`, `peft`, `trl`, `bitsandbytes` (RTX 3080 Ti, 12 GB; see
  `train/TRAINING_PLAN.md` §2). bitsandbytes on Windows may need WSL/fp16 fallback.
- `train/train_lora.py`: LoRA SFT (TRL `SFTTrainer`, completion-only masking on the
  assistant goal-set) on the filtered dataset. Base: `Qwen2.5-3B-Instruct` (4-bit).
- Serve the trained student back as a rollout provider: implement a
  `TransformersProvider.chat_json` (or export merged→GGUF→Ollama) so the next round
  runs episodes with the improved policy.

### F. Expert-iteration orchestrator — `train/expert_iter.py`
- Loop: rollout (C) with current policy → score → filter (D) → LoRA train (E) →
  swap policy → repeat. Round 0 policy = gemma4. Persist each round's checkpoint +
  the baseline/eval numbers.

### G. Baseline + eval (do this FIRST — it's cheap and it's the yardstick)
- Run (C) with gemma4 as the policy across varied drives; report mean return,
  survival time, novelty discovered, per drive profile. This is BOTH the eval
  harness we lacked and the number every training round must beat.

## 4. Constraints (from CLAUDE.md)

1. Reward signals MEASURE state; never authored judgments or per-turn prose.
2. Reuse interact/inventory/memory/perception; the runner reuses the engine.
3. Delete anything you stop using; no "just in case" scaffolding.
4. Simplest thing that works: expert iteration before PPO; no power reward yet; no
   extra tunables/abstractions beyond the reward registry.
5. Keep the goal-set I/O contract strict (`filter_response`).
6. `smoke_test.py` stays green; new modules get coverage. Run `python smoke_test.py`
   → `SMOKE OK` after each change.

## 5. Suggested build order

1. G (baseline) needs A + B + C, so: **A → B → C → G** (measure gemma4).
2. **D → E → F** (the training loop), then iterate and watch G's numbers climb.

## 6. Open decisions to confirm with the project owner

- **Reward shaping specifics:** exact survival curve (min-meter linear vs a penalty
  for any meter under a threshold), curiosity signal (per new type vs per novel
  memory), discount factor, death penalty magnitude. Start simple, tune against G.
- **Episode scale:** map size, NPC count per episode, time budget, episodes/round —
  bounded by rollout throughput (gemma4 is slow; the distilled student is the point
  at which rollouts get cheap).
- **Student base model** (`Qwen2.5-3B-Instruct` assumed) and whether to serve via
  Transformers or GGUF/Ollama.
- **Drive-profile sampling distribution** (uniform over survival/curiosity, or
  weighted toward realistic mixes).
