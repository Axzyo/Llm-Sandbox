import json
import urllib.request
from urllib.parse import urlparse

_ESCAPES = {'"': '"', "\\": "\\", "/": "/", "b": "\b", "f": "\f", "n": "\n", "r": "\r", "t": "\t"}


def _text_value_so_far(raw: str):
    """Decode the params.text string value as far as it has streamed in.

    Returns (decoded_so_far, closed) once the `"text"` key has appeared, or
    None if it hasn't yet. Stops cleanly on an incomplete trailing escape so
    a chunk boundary mid-escape never emits a broken character.
    """
    key = raw.find('"text"')
    if key == -1:
        return None
    start = raw.find('"', key + 6)  # opening quote of the value
    if start == -1:
        return None
    out = []
    j = start + 1
    n = len(raw)
    while j < n:
        c = raw[j]
        if c == "\\":
            if j + 1 >= n:
                break  # escape split across chunks; wait for more
            nxt = raw[j + 1]
            if nxt == "u":
                if j + 6 > n:
                    break
                out.append(chr(int(raw[j + 2:j + 6], 16)))
                j += 6
                continue
            out.append(_ESCAPES.get(nxt, nxt))
            j += 2
            continue
        if c == '"':
            return "".join(out), True
        out.append(c)
        j += 1
    return "".join(out), False


class AnthropicProvider:
    """chat_json served by Claude via the Anthropic API — a strong TEACHER policy
    for round-0 rollouts (it reasons over the survival signal where the local
    models idle). Same contract as OllamaProvider so the Brain is unchanged.

    Sonnet 5 specifics honored here: temperature/sampling params are rejected
    (never sent); the big static system prompt is cache-marked so repeated calls
    in an episode pay ~10% for it. The key is read from the environment by the
    SDK (ANTHROPIC_API_KEY / an `ant` profile) — never passed in code."""

    def __init__(self, model: str = "claude-sonnet-5", num_predict: int = 400,
                 effort: str = "low"):
        import anthropic
        self.client = anthropic.Anthropic()
        self.model = model
        self.num_predict = num_predict
        self.effort = effort            # low is plenty for a per-turn goal-set; keeps cost/latency down
        self._use_effort = True         # dropped automatically if this SDK/model rejects it

    def warm(self) -> None:
        pass                            # hosted API: no local model to preload

    def chat_json(self, system: str, user: str) -> dict:
        kw = dict(
            model=self.model,
            max_tokens=self.num_predict,
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": user}],
        )
        if self._use_effort:
            kw["output_config"] = {"effort": self.effort}
        try:
            resp = self.client.messages.create(**kw)
        except Exception as exc:        # SDK (TypeError) or server (400) rejects effort -> drop it, retry once
            if self._use_effort and ("output_config" in str(exc) or "effort" in str(exc)
                                     or isinstance(exc, TypeError)):
                self._use_effort = False
                kw.pop("output_config", None)
                resp = self.client.messages.create(**kw)
            else:
                raise
        if resp.stop_reason == "refusal":
            raise ValueError(f"model refused: {getattr(resp.stop_details, 'category', '?')}")
        text = "".join(b.text for b in resp.content if b.type == "text").strip()
        if text.startswith("```"):      # tolerate a fenced ```json ... ``` reply
            text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        parsed = json.loads(text)       # strict: a non-JSON reply raises, like the others
        if not isinstance(parsed, dict):
            raise ValueError("model output was not a JSON object")
        return parsed


class TransformersProvider:
    """Same chat_json contract as OllamaProvider, served by a local HF checkpoint
    (optionally with a LoRA adapter) — how a trained student rolls out episodes.
    torch/transformers/peft import lazily at construction, so the sim proper
    never needs the GPU stack installed."""

    def __init__(self, model_path: str, adapter: str | None = None,
                 temperature: float = 0.2, num_predict: int = 400):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.model = model_path            # the name, matching OllamaProvider.model
        self.temperature = temperature
        self.num_predict = num_predict
        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        try:
            from transformers import BitsAndBytesConfig
            quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
            net = AutoModelForCausalLM.from_pretrained(model_path, quantization_config=quant,
                                                       device_map="auto")
        except Exception:                  # no bitsandbytes (Windows) -> fp16 fallback
            net = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16,
                                                       device_map="auto")
        if adapter:
            from peft import PeftModel
            net = PeftModel.from_pretrained(net, adapter)
        self._net = net.eval()

    def chat_json(self, system: str, user: str) -> dict:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        inputs = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(self._net.device)
        with self._torch.no_grad():
            out = self._net.generate(
                inputs,
                max_new_tokens=self.num_predict,
                do_sample=self.temperature > 0,
                temperature=self.temperature if self.temperature > 0 else None,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        text = self.tokenizer.decode(out[0][inputs.shape[1]:], skip_special_tokens=True).strip()
        parsed = json.loads(text)          # strict: a non-JSON reply raises, like OllamaProvider
        if not isinstance(parsed, dict):
            raise ValueError("model output was not a JSON object")
        return parsed


class OllamaProvider:
    def __init__(self, url: str, model: str, temperature: float = 0.2, num_predict: int = 400,
                 timeout: int = 120, keep_alive: str = "10m"):
        parsed = urlparse(url if "://" in url else "http://" + url)
        host = parsed.hostname or "127.0.0.1"
        # localhost resolves to IPv6 ::1 first on Windows; Ollama listens on IPv4,
        # so the connect stalls ~2s before falling back. Force IPv4 to avoid it.
        if host == "localhost":
            host = "127.0.0.1"
        port = parsed.port or 11434
        self.url = f"{parsed.scheme or 'http'}://{host}:{port}"
        self.model = model
        self.temperature = temperature
        self.num_predict = num_predict
        self.timeout = timeout
        self.keep_alive = keep_alive

    def warm(self) -> None:
        """Fire-and-forget: load the model into memory so the first real call
        isn't a multi-second cold start. Safe to call in a background thread."""
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False,
            "keep_alive": self.keep_alive,
            "options": {"num_predict": 1},
        }
        req = urllib.request.Request(
            self.url + "/api/chat",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                resp.read()
        except Exception:
            pass

    def chat_json(self, system: str, user: str) -> dict:
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "format": "json",
            "think": False,
            "keep_alive": self.keep_alive,
            "options": {"temperature": self.temperature, "num_predict": self.num_predict},
        }
        req = urllib.request.Request(
            self.url + "/api/chat",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        content = data["message"]["content"].strip()
        parsed = json.loads(content)
        if not isinstance(parsed, dict):
            raise ValueError("model output was not a JSON object")
        return parsed

    def chat_json_stream(self, system: str, user: str, on_delta=None) -> dict:
        """Same as chat_json but streams; on_delta(text) fires with each new
        chunk of the params.text value as it generates. Returns the final dict."""
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": True,
            "format": "json",
            "think": False,
            "keep_alive": self.keep_alive,
            "options": {"temperature": self.temperature, "num_predict": self.num_predict},
        }
        req = urllib.request.Request(
            self.url + "/api/chat",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        raw = ""
        emitted = 0
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            for line in resp:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                piece = obj.get("message", {}).get("content", "")
                if piece:
                    raw += piece
                    if on_delta is not None:
                        res = _text_value_so_far(raw)
                        if res is not None:
                            decoded, _closed = res
                            if len(decoded) > emitted:
                                on_delta(decoded[emitted:])
                                emitted = len(decoded)
                if obj.get("done"):
                    break
        parsed = json.loads(raw.strip())
        if not isinstance(parsed, dict):
            raise ValueError("model output was not a JSON object")
        return parsed
