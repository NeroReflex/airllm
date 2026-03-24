from __future__ import annotations

import base64
import io
import json
import re
import threading
import time
import uuid
from typing import Any

import requests
from transformers import AutoConfig

from airllm import AutoModel

from .config import Settings


_QUIET_SENTINEL = object()


class ServerRunner:
    def __init__(self, settings: Settings) -> None:
        self.settings: Settings = settings
        self.model = None
        self.tokenizer = None
        self.loaded_model_id = None
        self.effective_max_seq_len = None
        self.model_lock = threading.Lock()

    def _infer_max_seq_len_from_model(self, model_id: str) -> int:
        """Infer the largest practical context length from model config."""
        try:
            if self.settings.hf_token:
                config = AutoConfig.from_pretrained(
                    model_id, token=self.settings.hf_token, trust_remote_code=True
                )
            else:
                config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        except Exception:
            # Fallback keeps behaviour predictable when config cannot be fetched.
            return 1024

        candidates: list[int] = []

        # Common config keys used by different architectures.
        for key in (
            "max_position_embeddings",
            "n_positions",
            "seq_length",
            "max_sequence_length",
            "max_seq_len",
            "max_seq_length",
            "context_length",
            "n_ctx",
        ):
            value = getattr(config, key, None)
            if isinstance(value, int) and value > 0:
                candidates.append(value)

        # Some models encode the extended length inside rope_scaling.
        rope_scaling = getattr(config, "rope_scaling", None)
        if isinstance(rope_scaling, dict):
            rope_max = rope_scaling.get("max_position_embeddings")
            if isinstance(rope_max, int) and rope_max > 0:
                candidates.append(rope_max)

            original = rope_scaling.get("original_max_position_embeddings")
            factor = rope_scaling.get("factor")
            if (
                isinstance(original, int)
                and original > 0
                and isinstance(factor, (int, float))
                and factor > 0
            ):
                candidates.append(int(original * float(factor)))

        # Discard placeholder infinite values seen in some tokenizer configs.
        sane_candidates = [v for v in candidates if v < 10_000_000]
        if not sane_candidates:
            return 1024
        return max(sane_candidates)

    def load_model_if_needed(self, model_id: str | None = None) -> None:
        target_model: str = model_id or self.settings.model_id
        if self.model is not None and self.loaded_model_id == target_model:
            return

        with self.model_lock:
            if self.model is not None and self.loaded_model_id == target_model:
                return

            layers_per_batch: Any = self.settings.layers_per_batch
            if layers_per_batch.isdigit():
                layers_per_batch = int(layers_per_batch)

            max_seq_len = self.settings.max_seq_len
            if max_seq_len is None:
                max_seq_len = self._infer_max_seq_len_from_model(target_model)

            self.model = AutoModel.from_pretrained(
                target_model,
                device=self.settings.device,
                max_seq_len=max_seq_len,
                prefetching=self.settings.prefetching,
                layers_per_batch=layers_per_batch,
                hf_token=self.settings.hf_token or None,
            )
            self.tokenizer: Any | None = getattr(self.model, "tokenizer", None)
            self.loaded_model_id: str = target_model
            self.effective_max_seq_len = max_seq_len
            self._dump_model_serve_info(layers_per_batch=layers_per_batch)

    def _chat_template_mode_for_dump(self) -> str:
        cfg = (self.settings.chat_template or "").strip()
        if cfg.lower() in ("none", "false", "0", "off", "no"):
            return "disabled (legacy ROLE: content formatting)"
        if cfg and cfg.lower() not in ("auto",):
            return f"custom file: {cfg}"

        tok = self.tokenizer
        built_in = getattr(tok, "chat_template", None) if tok is not None else None
        if isinstance(built_in, str) and built_in.strip():
            return "auto (model built-in tokenizer chat_template)"
        return "auto (no tokenizer chat_template found; fallback to legacy format)"

    def _dump_model_serve_info(self, layers_per_batch: Any) -> None:
        """Print effective serving settings after a model is loaded."""
        summary = [
            "[airllm] model loaded for serving",
            f"  model: {self.loaded_model_id}",
            f"  device: {self.settings.device}",
            f"  context_window: {self.effective_max_seq_len}",
            f"  max_new_tokens(default): {self.settings.max_new_tokens}",
            f"  temperature(default): {self.settings.temperature}",
            f"  top_p(default): {self.settings.top_p}",
            f"  prefetching: {self.settings.prefetching}",
            f"  layers_per_batch: {layers_per_batch}",
            f"  lazy_load_model: {self.settings.lazy_load_model}",
            f"  chat_template: {self._chat_template_mode_for_dump()}",
        ]
        print("\n".join(summary), flush=True)

    # ------------------------------------------------------------------
    # Prompt formatting
    # ------------------------------------------------------------------

    def _flatten_messages_to_prompt(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> tuple[str, list[Any], bool]:
        """Convert an OpenAI-style messages list into a text prompt + images.

        Applies the model's Jinja2 chat template when available — the same
        mechanism used by vLLM's ``--chat-template`` / llama.cpp's ``--jinja``
        flags.  Falls back to a simple ``ROLE: content`` concatenation when no
        template is available or when explicitly disabled via
        ``AIRLLM_CHAT_TEMPLATE=none``.

        Returns ``(prompt_str, images, used_chat_template)``.
        """
        # 1. Separate image payloads from the text content.
        images: list[Any] = []
        text_messages: list[dict[str, Any]] = []
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, list):
                text_chunks: list[str] = []
                for part in content:
                    if part.get("type") == "text":
                        text_chunks.append(part.get("text", ""))
                    elif part.get("type") == "image_url":
                        img = self._load_image_from_url(
                            part.get("image_url", {}).get("url", "")
                        )
                        if img is not None:
                            images.append(img)
                            text_chunks.append("<|image|>")
                text_messages.append({**msg, "content": " ".join(text_chunks).strip()})
            else:
                text_messages.append(msg)

        # 2. Build the text prompt via the chat template or naive fallback.
        prompt, used_template = self._apply_chat_template(
            text_messages,
            tools=tools,
            tool_choice=tool_choice,
        )
        return prompt, images, used_template

    def _apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> tuple[str, bool]:
        """Apply a Jinja2 chat template to text-only messages.

        Resolution order (mirrors vLLM's chat-template logic):

        1. ``AIRLLM_CHAT_TEMPLATE`` / ``--chat-template`` is a file path →
           load and use that Jinja2 template file.
        2. ``AIRLLM_CHAT_TEMPLATE`` is ``"none"`` / ``"false"`` → skip
           templating and return the naive-formatter output.
        3. Otherwise (default: empty / ``"auto"``) → attempt the model's
           built-in ``tokenizer.chat_template`` from
           ``tokenizer_config.json``.
        4. Hard fallback → naive ``ROLE: content`` concatenation.

        Returns ``(prompt_str, used_template_flag)``.
        """
        cfg = (self.settings.chat_template or "").strip()

        # Explicit opt-out.
        if cfg.lower() in ("none", "false", "0", "off", "no"):
            return self._naive_format(messages), False

        tokenizer = self.tokenizer

        # Custom Jinja2 template file.
        template_str: str | None = None
        if cfg and cfg.lower() not in ("", "auto"):
            try:
                with open(cfg) as fh:
                    template_str = fh.read()
            except OSError as exc:
                raise ValueError(
                    f"Cannot read chat template file '{cfg}': {exc}"
                ) from exc

        # Attempt HuggingFace tokenizer.apply_chat_template.
        try:
            kwargs: dict[str, Any] = {
                "tokenize": False,
                "add_generation_prompt": True,
            }
            if template_str is not None:
                kwargs["chat_template"] = template_str
            if tools:
                kwargs["tools"] = tools
            if tool_choice is not None:
                kwargs["tool_choice"] = tool_choice
            prompt: str = tokenizer.apply_chat_template(messages, **kwargs)
            return prompt, True
        except Exception:
            # Tokenizer has no chat_template, jinja2 is unavailable, or the
            # model's template does not support the supplied tool schema.
            return self._naive_format(messages), False

    def _naive_format(self, messages: list[dict[str, Any]]) -> str:
        """Legacy ``ROLE: content\\n…\\nASSISTANT:`` prompt format."""
        parts: list[str] = []
        for msg in messages:
            role = msg.get("role", "user").upper()
            content = msg.get("content") or ""
            parts.append(f"{role}: {content}")
        parts.append("ASSISTANT:")
        return "\n".join(parts)

    def _extract_tool_calls_from_completion(
        self,
        completion: str,
    ) -> tuple[str, list[dict[str, Any]]]:
        """Parse MiniMax-style XML tool calls into OpenAI-compatible objects."""
        block_re = re.compile(
            r"<minimax:tool_call>(.*?)</minimax:tool_call>", re.DOTALL
        )
        invoke_re = re.compile(r'<invoke\s+name="([^"]+)">(.*?)</invoke>', re.DOTALL)
        param_re = re.compile(
            r'<parameter\s+name="([^"]+)">(.*?)</parameter>', re.DOTALL
        )

        tool_calls: list[dict[str, Any]] = []
        for block in block_re.findall(completion or ""):
            for name, invoke_body in invoke_re.findall(block):
                args_obj: dict[str, Any] = {}
                for key, raw_value in param_re.findall(invoke_body):
                    value = raw_value.strip()
                    try:
                        args_obj[key] = json.loads(value)
                    except Exception:
                        args_obj[key] = value
                tool_calls.append(
                    {
                        "id": f"call_{uuid.uuid4().hex[:24]}",
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": json.dumps(args_obj, ensure_ascii=False),
                        },
                    }
                )

        cleaned = block_re.sub("", completion or "").strip()
        return cleaned, tool_calls

    # ------------------------------------------------------------------
    # Image helpers
    # ------------------------------------------------------------------

    def _load_image_from_url(self, url: str) -> Any | None:
        try:
            from PIL import Image
        except ModuleNotFoundError:
            return None

        if not url:
            return None
        if url.startswith("data:image"):
            try:
                b64: str = url.split(",", 1)[1]
                raw: bytes = base64.b64decode(b64)
                return Image.open(io.BytesIO(raw)).convert("RGB")
            except Exception:
                return None

        try:
            r: requests.Response = requests.get(url, timeout=20)
            r.raise_for_status()
            return Image.open(io.BytesIO(r.content)).convert("RGB")
        except Exception:
            return None

    def _set_quiet_generation(self, enabled: bool) -> Any:
        """Toggle backend-specific generation progress/log output."""
        if self.model is None or not enabled:
            return None
        previous = getattr(self.model, "quiet_generation", _QUIET_SENTINEL)
        setattr(self.model, "quiet_generation", True)
        return previous

    def _restore_quiet_generation(self, previous: Any, enabled: bool) -> None:
        if self.model is None or not enabled or previous is None:
            return
        if previous is _QUIET_SENTINEL:
            try:
                delattr(self.model, "quiet_generation")
            except AttributeError:
                pass
        else:
            setattr(self.model, "quiet_generation", previous)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def generate_chat(
        self,
        messages: list[dict[str, Any]],
        model_id: str | None,
        max_tokens: int,
        temperature: float,
        top_p: float,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        suppress_output: bool = False,
    ) -> dict[str, Any]:
        self.load_model_if_needed(model_id)
        prompt, images, used_template = self._flatten_messages_to_prompt(
            messages,
            tools=tools,
            tool_choice=tool_choice,
        )

        if images and hasattr(self.model, "get_processor"):
            processor = self.model.get_processor()
            inputs = processor(text=prompt, images=images, return_tensors="pt")
            previous_quiet = self._set_quiet_generation(suppress_output)
            try:
                output_ids = self.model.generate(
                    input_ids=inputs["input_ids"].to(self.settings.device),
                    attention_mask=(
                        inputs["attention_mask"].to(self.settings.device)
                        if inputs.get("attention_mask") is not None
                        else None
                    ),
                    pixel_values=inputs.get("pixel_values"),
                    aspect_ratio_ids=inputs.get("aspect_ratio_ids"),
                    aspect_ratio_mask=inputs.get("aspect_ratio_mask"),
                    cross_attention_mask=inputs.get("cross_attention_mask"),
                    max_new_tokens=max_tokens,
                    use_cache=False,
                )
            finally:
                self._restore_quiet_generation(previous_quiet, suppress_output)
            n_in = int(inputs["input_ids"].shape[-1])
            completion_ids = output_ids[0][n_in:]
            completion = processor.decode(
                completion_ids, skip_special_tokens=True
            ).strip()
            prompt_tokens = n_in
            completion_tokens = int(completion_ids.shape[-1])
        else:
            toks = self.tokenizer(
                [prompt],
                return_tensors="pt",
                truncation=True,
                max_length=self.effective_max_seq_len,
                add_special_tokens=not used_template,
            )
            input_ids = toks["input_ids"].to(self.settings.device)
            attention_mask = (
                toks["attention_mask"].to(self.settings.device)
                if toks.get("attention_mask") is not None
                else None
            )
            kwargs: dict[str, Any] = {
                "max_new_tokens": max_tokens,
                "use_cache": False,
            }
            if temperature > 0:
                kwargs["do_sample"] = True
                kwargs["temperature"] = temperature
                kwargs["top_p"] = top_p
            if attention_mask is not None:
                kwargs["attention_mask"] = attention_mask
            previous_quiet = self._set_quiet_generation(suppress_output)
            try:
                output_ids = self.model.generate(input_ids, **kwargs)
            finally:
                self._restore_quiet_generation(previous_quiet, suppress_output)
            completion_ids = output_ids[0][input_ids.shape[-1] :]
            completion = self.tokenizer.decode(
                completion_ids, skip_special_tokens=True
            ).strip()
            prompt_tokens = int(input_ids.shape[-1])
            completion_tokens = int(completion_ids.shape[-1])

        clean_text, tool_calls = self._extract_tool_calls_from_completion(completion)
        finish_reason = "tool_calls" if tool_calls else "stop"

        return {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.loaded_model_id,
            "completion_text": clean_text,
            "tool_calls": tool_calls,
            "finish_reason": finish_reason,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }

    def generate_chat_streaming(
        self,
        messages: list[dict[str, Any]],
        model_id: str | None,
        max_tokens: int,
        temperature: float,
        top_p: float,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        suppress_output: bool = False,
    ) -> tuple[dict[str, Any], Any, threading.Thread]:
        """Start streaming chat generation using TextIteratorStreamer.

        Returns ``(meta, streamer, thread)`` immediately.  The caller must
        consume *streamer* to receive tokens in real time, then join *thread*.
        Falls back to a single-shot generator if the model does not support
        the ``streamer`` kwarg (e.g. the MLX backend).
        """
        from transformers import TextIteratorStreamer

        self.load_model_if_needed(model_id)
        prompt, images, used_template = self._flatten_messages_to_prompt(
            messages, tools=tools, tool_choice=tool_choice
        )

        toks = self.tokenizer(
            [prompt],
            return_tensors="pt",
            truncation=True,
            max_length=self.effective_max_seq_len,
            add_special_tokens=not used_template,
        )
        input_ids = toks["input_ids"].to(self.settings.device)
        attention_mask = (
            toks["attention_mask"].to(self.settings.device)
            if toks.get("attention_mask") is not None
            else None
        )
        prompt_tokens = int(input_ids.shape[-1])

        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_special_tokens=True,
            timeout=120.0,
        )

        gen_kwargs: dict[str, Any] = {
            "input_ids": input_ids,
            "max_new_tokens": max_tokens,
            "use_cache": False,
            "streamer": streamer,
        }
        if attention_mask is not None:
            gen_kwargs["attention_mask"] = attention_mask
        if temperature > 0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p

        def _run_generate() -> None:
            previous_quiet = self._set_quiet_generation(suppress_output)
            try:
                self.model.generate(**gen_kwargs)
            finally:
                self._restore_quiet_generation(previous_quiet, suppress_output)

        thread = threading.Thread(target=_run_generate, daemon=True)
        thread.start()

        meta = {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "created": int(time.time()),
            "model": self.loaded_model_id,
            "prompt_tokens": prompt_tokens,
        }
        return meta, streamer, thread

    def generate_completion(
        self,
        prompt: str,
        model_id: str | None,
        max_tokens: int,
        temperature: float,
        top_p: float,
        suppress_output: bool = False,
    ) -> dict[str, Any]:
        self.load_model_if_needed(model_id)
        toks = self.tokenizer(
            [prompt],
            return_tensors="pt",
            truncation=True,
            max_length=self.effective_max_seq_len,
        )
        input_ids = toks["input_ids"].to(self.settings.device)
        attention_mask = (
            toks["attention_mask"].to(self.settings.device)
            if toks.get("attention_mask") is not None
            else None
        )

        kwargs: dict[str, Any] = {
            "max_new_tokens": max_tokens,
            "use_cache": False,
        }
        if temperature > 0:
            kwargs["do_sample"] = True
            kwargs["temperature"] = temperature
            kwargs["top_p"] = top_p
        if attention_mask is not None:
            kwargs["attention_mask"] = attention_mask

        previous_quiet = self._set_quiet_generation(suppress_output)
        try:
            output_ids = self.model.generate(input_ids, **kwargs)
        finally:
            self._restore_quiet_generation(previous_quiet, suppress_output)
        completion_ids = output_ids[0][input_ids.shape[-1] :]
        completion = self.tokenizer.decode(
            completion_ids, skip_special_tokens=True
        ).strip()
        prompt_tokens = int(input_ids.shape[-1])
        completion_tokens = int(completion_ids.shape[-1])

        return {
            "id": f"cmpl-{uuid.uuid4().hex}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": self.loaded_model_id,
            "completion_text": completion,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }

    def synthesize_speech(self, text: str, model_id: str | None) -> bytes:
        try:
            from scipy.io.wavfile import write as wav_write
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "SciPy is required for audio synthesis output"
            ) from exc

        self.load_model_if_needed(model_id)
        if not hasattr(self.model, "tts"):
            raise RuntimeError(
                "Current model backend does not support text-to-speech"
            )

        wav_tensor = self.model.tts(text)
        wav_np = wav_tensor.detach().cpu().numpy()

        bio = io.BytesIO()
        wav_write(bio, 16000, wav_np)
        return bio.getvalue()
