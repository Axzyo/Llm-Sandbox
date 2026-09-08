---
name: llm-latency-fix
description: Ollama calls must use 127.0.0.1 not localhost (Windows IPv6 ::1 stall adds ~2s); streaming is wired for dialogue
metadata: 
  node_type: memory
  type: project
  originSessionId: 8201b9c2-3b96-43a4-bf15-fdd0998cb731
  modified: 2026-08-25T20:30:04.876Z
---

NPC LLM response latency was ~2.3s per call; the cause was `localhost` resolving to IPv6 `::1` first on Windows and stalling before falling back to IPv4. Using `127.0.0.1` drops a call from 2.31s → 0.17s. Fixed in sim/provider.py (normalizes `localhost`→`127.0.0.1`) and config.json, plus `keep_alive:"10m"` to keep the model warm.

After the fix: actions ~0.3–0.5s, dialogue ~0.6–0.8s — all sub-1s. Model is `gemma4` (8B Q4) via local Ollama.

Dialogue replies also **stream** (`OllamaProvider.chat_json_stream` + `_text_value_so_far` extracts the `params.text` field out of the streaming JSON envelope). The NPC types out live in the chat HUD; first token ~0.04–0.35s warm. Streaming is display-only — the LLM still makes every decision. See [[memory-system-design]].