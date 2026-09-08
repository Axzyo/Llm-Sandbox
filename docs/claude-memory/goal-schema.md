---
name: goal-schema
description: "LLM output is now goal-sets (ordered action list + one importance), not single actions"
metadata: 
  node_type: memory
  type: project
  originSessionId: a8dcbd6f-f51d-4e85-b4e6-c423df7a339c
  modified: 2026-08-29T09:47:52.892Z
---

The NPC brain's output schema was finalized on 2026-08-27 (user decision): the LLM no longer emits one action per think. It emits a **goal set**.

- A **Goal** = an ordered `actions` list (the same per-action schema: move/interact/say/inventory) + ONE `importance` (0–10) + one `reason`. Importance is **per goal-set, not per action** (user chose this explicitly over per-action importance).
- LLM reply shape: `{"goals":[{"actions":[<action>,...],"importance":<0-10>,"reason":"..."}]}`. `wait` and `recall` are standalone thinking-layer replies, NOT goals.
- `sim/goals.py`: `Goal` has `actions`, `importance`, `reason`, `status`, `step` (cursor into actions), `started_step`; `current_action`/`advance()` step through the plan. `GoalList` sorts by (importance desc, seq asc); higher-importance goals preempt and the old plan resumes at its `step`. A failed action abandons the whole plan.
- `sim/brain.py`: `validate_goals()` parses the goal-set shape (tolerates bare list / bare action dict / single goal dict); `decide()` now returns `list[Goal]` (empty = do nothing), runs the recall loop (bounded MAX_RECALLS=2, then one corrective retry), and `wait` → `[]`. `validate_intent()` still exists for the injected/scripted-intent bridge (`goal_from_intent`, importance 10.0 for injected).
- `main.py`: `advance_goals` works the top goal's `current_action`; `progress_move(goal, action_obj, ...)` durative, `enact_instant(action_obj, ...)` one-tick. Verified end-to-end: a move→say plan executes and completes.
- `smoke_test.py` updated to the new shape; **all pass**.

**`train/extract_dataset.py` REWRITTEN for goal-sets (2026-08-27):** now uses `filter_response`; keeps only `kind=="goals"` responses; assistant target = `json.dumps(goals_to_obj(goals))` (new `brain.goals_to_obj` serializer, whole importances render as ints). Skips recall/look/wait/bad (counted in the report). Data plan = **live gameplay logs + goal-sets only** (user's choice). `train/gen_scenarios.py` + `train/teachers.py` (synthetic path) were DELETED (2026-08-27 audit, rule 3 dead code) — data source is live logs, not synthetic; see [[decide-training-teacher]].

**Why:** captures the schema the whole game + any future training now depends on.
**How to apply:** treat the LLM contract as goal-sets everywhere; fix extract_dataset before regenerating training data.
