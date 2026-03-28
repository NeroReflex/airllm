from __future__ import annotations

import base64
import io
import json
import queue
import re
import threading
import time
import uuid
from typing import Any

import requests
from transformers import AutoConfig

from airllm import AutoModel

from .config import Settings

try:
    from openai_harmony import (
        HarmonyEncodingName,
        Role as HarmonyRole,
        StreamableParser,
        load_harmony_encoding,
    )
except ImportError:
    HarmonyEncodingName = None
    HarmonyRole = None
    StreamableParser = None
    load_harmony_encoding = None


_QUIET_SENTINEL = object()


class _HarmonyFinalChannelStreamer:
    """Queue-backed streamer that exposes only GPT-OSS final-channel deltas."""

    def __init__(self, encoding: Any, prompt_token_count: int, timeout: float = 120.0):
        if StreamableParser is None or HarmonyRole is None:
            raise RuntimeError("openai-harmony streaming support is unavailable")
        self._parser = StreamableParser(encoding, HarmonyRole.ASSISTANT)
        self._prompt_tokens_to_skip = max(0, int(prompt_token_count))
        self._timeout = timeout
        self._queue: queue.Queue[Any] = queue.Queue()
        self._stop_signal = object()

    def put(self, value: Any) -> None:
        tokens = self._normalize_tokens(value)
        if self._prompt_tokens_to_skip:
            if len(tokens) <= self._prompt_tokens_to_skip:
                self._prompt_tokens_to_skip -= len(tokens)
                return
            tokens = tokens[self._prompt_tokens_to_skip :]
            self._prompt_tokens_to_skip = 0

        for token in tokens:
            try:
                self._parser.process(int(token))
            except Exception:
                continue
            delta = self._parser.last_content_delta
            if delta and self._parser.current_channel == "final":
                self._queue.put(delta)

    def end(self) -> None:
        try:
            self._parser.process_eos()
        except Exception:
            pass
        self._queue.put(self._stop_signal)

    def __iter__(self):
        return self

    def __next__(self) -> str:
        while True:
            try:
                value = self._queue.get(timeout=self._timeout)
            except queue.Empty:
                # Harmony streams can have long silences while still generating.
                # Keep waiting until we receive text or an explicit stop signal.
                continue
            if value is self._stop_signal:
                raise StopIteration
            return value

    @staticmethod
    def _normalize_tokens(value: Any) -> list[int]:
        if hasattr(value, "tolist"):
            value = value.tolist()
        if isinstance(value, tuple):
            value = list(value)
        if isinstance(value, list):
            if value and isinstance(value[0], list):
                value = value[0]
            return [int(token) for token in value]
        return [int(value)]


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
            config_kwargs = {}
            if self.settings.hf_token:
                config_kwargs["token"] = self.settings.hf_token
            try:
                config = AutoConfig.from_pretrained(
                    model_id,
                    trust_remote_code=False,
                    **config_kwargs,
                )
            except Exception:
                config = AutoConfig.from_pretrained(
                    model_id,
                    trust_remote_code=True,
                    **config_kwargs,
                )
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

    def _uses_harmony_format(self) -> bool:
        model_id = (self.loaded_model_id or "").lower()
        if "gpt-oss" in model_id:
            return True

        template = getattr(self.tokenizer, "chat_template", None)
        if not isinstance(template, str):
            return False
        return "<|channel|>" in template and "<|start|>assistant" in template

    def _get_harmony_encoding(self) -> Any | None:
        if not self._uses_harmony_format() or load_harmony_encoding is None:
            return None
        encoding = getattr(self, "_harmony_encoding", None)
        if encoding is None:
            encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
            self._harmony_encoding = encoding
        return encoding

    def _harmony_content_to_text(self, content: Any) -> str:
        if isinstance(content, str):
            return content
        if not isinstance(content, list):
            return ""

        parts: list[str] = []
        for part in content:
            if isinstance(part, dict):
                if part.get("type") == "text":
                    parts.append(part.get("text", ""))
                elif "text" in part:
                    parts.append(str(part["text"]))
                elif "content" in part:
                    parts.append(str(part["content"]))
            else:
                text = getattr(part, "text", None)
                if text is not None:
                    parts.append(text)
        return "".join(parts)

    def _parse_harmony_completion_tokens(
        self,
        completion_ids: Any,
    ) -> tuple[str, str | None, list[dict[str, Any]], str] | None:
        encoding = self._get_harmony_encoding()
        if encoding is None or HarmonyRole is None:
            return None

        if hasattr(completion_ids, "tolist"):
            completion_ids = completion_ids.tolist()
        token_list = [int(token) for token in completion_ids]
        if not token_list:
            return "", None, [], "stop"

        try:
            messages = encoding.parse_messages_from_completion_tokens(
                token_list,
                role=HarmonyRole.ASSISTANT,
                strict=False,
            )
        except Exception:
            return None

        reasoning_chunks: list[str] = []
        final_chunks: list[str] = []
        fallback_chunks: list[str] = []
        tool_calls: list[dict[str, Any]] = []

        for message in messages:
            data = message.to_dict() if hasattr(message, "to_dict") else message
            if not isinstance(data, dict):
                continue

            role = data.get("role")
            role_name = getattr(role, "value", role)
            if role_name != "assistant":
                continue

            channel = data.get("channel")
            recipient = data.get("recipient")
            text = self._harmony_content_to_text(data.get("content"))

            if isinstance(recipient, str) and recipient.startswith("functions."):
                function_name = recipient.split(".", 1)[1]
                arguments = text.strip() or "{}"
                try:
                    arguments = json.dumps(json.loads(arguments), ensure_ascii=False)
                except Exception:
                    arguments = json.dumps(arguments, ensure_ascii=False)
                tool_calls.append(
                    {
                        "id": f"call_{uuid.uuid4().hex[:24]}",
                        "type": "function",
                        "function": {
                            "name": function_name,
                            "arguments": arguments,
                        },
                    }
                )
                continue

            if channel == "analysis":
                if text:
                    reasoning_chunks.append(text)
            elif channel == "final":
                if text:
                    final_chunks.append(text)
            elif text:
                fallback_chunks.append(text)

        completion_text = "\n\n".join(final_chunks or fallback_chunks).strip()
        if not completion_text and reasoning_chunks and not tool_calls:
            completion_text = "\n\n".join(reasoning_chunks).strip()
        reasoning_content = "\n\n".join(reasoning_chunks).strip() or None
        finish_reason = "tool_calls" if tool_calls else "stop"
        return completion_text, reasoning_content, tool_calls, finish_reason

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
            # For plain chat on harmony models, force final-channel generation
            # so user-facing interactive mode does not stall on long analysis.
            if (
                self._uses_harmony_format()
                and not tools
                and tool_choice is None
                and prompt.rstrip().endswith("<|start|>assistant")
            ):
                prompt = f"{prompt.rstrip()}<|channel|>final<|message|>"
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

    def _extract_reasoning_from_completion(
        self,
        completion: str,
    ) -> tuple[str, str | None]:
        think_re = re.compile(r"<think>(.*?)</think>", re.DOTALL)

        reasoning_chunks = [chunk.strip() for chunk in think_re.findall(completion or "")]
        cleaned = think_re.sub("", completion or "").strip()
        reasoning = "\n\n".join(chunk for chunk in reasoning_chunks if chunk)
        return cleaned, reasoning or None

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
            else:
                kwargs["do_sample"] = False
            if attention_mask is not None:
                kwargs["attention_mask"] = attention_mask
            previous_quiet = self._set_quiet_generation(suppress_output)
            try:
                output_ids = self.model.generate(input_ids, **kwargs)
            finally:
                self._restore_quiet_generation(previous_quiet, suppress_output)
            completion_ids = output_ids[0][input_ids.shape[-1] :]
            prompt_tokens = int(input_ids.shape[-1])
            completion_tokens = int(completion_ids.shape[-1])

        completion = self.tokenizer.decode(
            completion_ids, skip_special_tokens=True
        ).strip()

        harmony_parsed = self._parse_harmony_completion_tokens(completion_ids)
        if harmony_parsed is not None:
            clean_text, reasoning_content, tool_calls, finish_reason = harmony_parsed
            # When the prompt already forced `<|channel|>final<|message|>`,
            # completion tokens may be raw content without harmony headers.
            # Keep decoded text instead of returning an empty assistant turn.
            if not clean_text and completion:
                clean_text = completion
        else:
            clean_text, tool_calls = self._extract_tool_calls_from_completion(completion)
            clean_text, reasoning_content = self._extract_reasoning_from_completion(clean_text)
            finish_reason = "tool_calls" if tool_calls else "stop"

        return {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.loaded_model_id,
            "completion_text": clean_text,
            "reasoning_content": reasoning_content,
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

        harmony_encoding = self._get_harmony_encoding()
        use_harmony_streamer = harmony_encoding is not None
        if use_harmony_streamer:
            streamer = _HarmonyFinalChannelStreamer(
                harmony_encoding,
                prompt_token_count=prompt_tokens,
                timeout=120.0,
            )
        else:
            streamer = TextIteratorStreamer(
                self.tokenizer,
                skip_special_tokens=True,
                skip_prompt=True,
                timeout=None,
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
        else:
            gen_kwargs["do_sample"] = False

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
        else:
            kwargs["do_sample"] = False
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
