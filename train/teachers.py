"""Teacher providers for distillation data generation.

A teacher is any object exposing `chat_json(system, user) -> dict` (the same
interface `Brain` already calls). We reuse the game's own `OllamaProvider` for
local teachers, and add a thin Anthropic (Claude) teacher for higher-quality
demonstrations when an API key is available.

The teacher is the quality ceiling of the distilled student: a stronger teacher
here is the single biggest lever on final behavior.
"""
import json
import os
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sim.provider import OllamaProvider  # noqa: E402


class AnthropicTeacher:
    """Claude as a distillation teacher. Implements chat_json(system, user).

    Uses the official Anthropic SDK (lazy-imported so this module still loads
    without it when only the local teacher is used). Credentials resolve the
    normal SDK way: ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN, or an `ant auth
    login` profile. The game's own validate_intent still gates whatever comes
    back, so a stray non-action response is simply dropped downstream.
    """

    def __init__(self, model: str = "claude-sonnet-5", max_tokens: int = 512,
                 temperature: float = 0.4, timeout: float = 60.0):
        try:
            import anthropic  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "the Claude teacher needs the Anthropic SDK: pip install anthropic"
            ) from exc
        self._anthropic = anthropic
        self.client = anthropic.Anthropic(timeout=timeout)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    def chat_json(self, system: str, user: str) -> dict:
        # Sonnet 5 / the 4.6+ family removed sampling params (temperature/top_p/
        # top_k); passing them errors. Diversity comes from the varied prompts.
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system + "\n\nReply with exactly one JSON object and nothing else.",
            messages=[{"role": "user", "content": user}],
        )
        text = "".join(
            b.text for b in resp.content if getattr(b, "type", None) == "text"
        ).strip()
        # tolerate ```json fences or stray prose: grab the first {...} span
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end != -1 and end > start:
            text = text[start:end + 1]
        parsed = json.loads(text)
        if not isinstance(parsed, dict):
            raise ValueError("teacher output was not a JSON object")
        return parsed


class OpenAICompatTeacher:
    """Any OpenAI-compatible chat endpoint as a teacher (Moonshot/Kimi, OpenRouter,
    etc.). Implements chat_json(system, user). Dependency-free (raw HTTP) since
    these are not Anthropic endpoints. Credentials come from an env var.
    """

    def __init__(self, base_url: str, model: str, api_key_env: str,
                 max_tokens: int = 512, temperature: float = 0.4, timeout: float = 60.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = os.environ.get(api_key_env)
        if not self.api_key:
            raise RuntimeError(
                f"no API key found in ${api_key_env}; set it, e.g.\n"
                f'  setx {api_key_env} "sk-..."   (open a new terminal after)'
            )
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout

    def chat_json(self, system: str, user: str) -> dict:
        body = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system",
                 "content": system + "\n\nReply with exactly one JSON object and nothing else."},
                {"role": "user", "content": user},
            ],
        }
        req = urllib.request.Request(
            self.base_url + "/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "content-type": "application/json",
                "authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        text = (data["choices"][0]["message"]["content"] or "").strip()
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end != -1 and end > start:
            text = text[start:end + 1]
        parsed = json.loads(text)
        if not isinstance(parsed, dict):
            raise ValueError("teacher output was not a JSON object")
        return parsed


def make_teacher(kind: str, cfg: dict):
    """kind: 'ollama' (local, free) | 'claude' (Anthropic) | 'kimi' (Moonshot,
    OpenAI-compatible) | 'openai_compat' (any OpenAI-style endpoint)."""
    if kind == "ollama":
        return OllamaProvider(
            cfg.get("ollama_url", "http://127.0.0.1:11434"),
            cfg.get("model", "gemma4"),
            temperature=cfg.get("temperature", 0.4),
            num_predict=cfg.get("num_predict", 400),
            keep_alive=cfg.get("keep_alive", "30m"),
        )
    if kind == "claude":
        return AnthropicTeacher(
            model=cfg.get("teacher_model", "claude-sonnet-5"),
            temperature=cfg.get("temperature", 0.4),
        )
    if kind == "kimi":
        return OpenAICompatTeacher(
            base_url=cfg.get("kimi_base_url", "https://api.moonshot.ai/v1"),
            model=cfg.get("teacher_model", "kimi-k3"),
            api_key_env=cfg.get("kimi_key_env", "MOONSHOT_API_KEY"),
            temperature=cfg.get("temperature", 0.4),
        )
    if kind == "openai_compat":
        return OpenAICompatTeacher(
            base_url=cfg["teacher_base_url"],
            model=cfg["teacher_model"],
            api_key_env=cfg.get("teacher_key_env", "OPENAI_API_KEY"),
            temperature=cfg.get("temperature", 0.4),
        )
    raise ValueError(f"unknown teacher kind: {kind!r}")
