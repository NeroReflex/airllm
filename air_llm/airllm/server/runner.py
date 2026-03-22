import base64
import io
import threading
import time
import uuid
from typing import Any

import requests
from PIL import Image
from scipy.io.wavfile import write as wav_write

from airllm import AutoModel

from .config import Settings


class ServerRunner:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.model = None
        self.tokenizer = None
        self.loaded_model_id = None
        self.model_lock = threading.Lock()

    def load_model_if_needed(self, model_id: str | None = None):
        target_model = model_id or self.settings.model_id
        if self.model is not None and self.loaded_model_id == target_model:
            return

        with self.model_lock:
            if self.model is not None and self.loaded_model_id == target_model:
                return

            layers_per_batch: Any = self.settings.layers_per_batch
            if layers_per_batch.isdigit():
                layers_per_batch = int(layers_per_batch)

            self.model = AutoModel.from_pretrained(
                target_model,
                device=self.settings.device,
                max_seq_len=self.settings.max_seq_len,
                prefetching=self.settings.prefetching,
                layers_per_batch=layers_per_batch,
                hf_token=self.settings.hf_token or None,
            )
            self.tokenizer = getattr(self.model, "tokenizer", None)
            self.loaded_model_id = target_model

    def _flatten_messages_to_prompt(self, messages: list[dict[str, Any]]) -> tuple[str, list[Image.Image]]:
        images: list[Image.Image] = []
        parts: list[str] = []

        for msg in messages:
            role = msg.get("role", "user").upper()
            content = msg.get("content", "")
            if isinstance(content, str):
                parts.append(f"{role}: {content}")
                continue

            if isinstance(content, list):
                text_chunks: list[str] = []
                for part in content:
                    if part.get("type") == "text":
                        text_chunks.append(part.get("text", ""))
                    elif part.get("type") == "image_url":
                        img = self._load_image_from_url(part.get("image_url", {}).get("url", ""))
                        if img is not None:
                            images.append(img)
                            text_chunks.append("<|image|>")
                parts.append(f"{role}: {' '.join(text_chunks).strip()}")

        parts.append("ASSISTANT:")
        return "\n".join(parts), images

    def _load_image_from_url(self, url: str) -> Image.Image | None:
        if not url:
            return None
        if url.startswith("data:image"):
            try:
                b64 = url.split(",", 1)[1]
                raw = base64.b64decode(b64)
                return Image.open(io.BytesIO(raw)).convert("RGB")
            except Exception:
                return None

        try:
            r = requests.get(url, timeout=20)
            r.raise_for_status()
            return Image.open(io.BytesIO(r.content)).convert("RGB")
        except Exception:
            return None

    def generate_chat(self, messages: list[dict[str, Any]], model_id: str | None, max_tokens: int, temperature: float, top_p: float):
        self.load_model_if_needed(model_id)
        prompt, images = self._flatten_messages_to_prompt(messages)

        if images and hasattr(self.model, "get_processor"):
            processor = self.model.get_processor()
            inputs = processor(text=prompt, images=images, return_tensors="pt")
            output_ids = self.model.generate(
                input_ids=inputs["input_ids"].to(self.settings.device),
                attention_mask=inputs.get("attention_mask", None).to(self.settings.device)
                if inputs.get("attention_mask", None) is not None
                else None,
                pixel_values=inputs.get("pixel_values", None),
                aspect_ratio_ids=inputs.get("aspect_ratio_ids", None),
                aspect_ratio_mask=inputs.get("aspect_ratio_mask", None),
                cross_attention_mask=inputs.get("cross_attention_mask", None),
                max_new_tokens=max_tokens,
                use_cache=False,
            )
            text = processor.batch_decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            completion = text[len(prompt) :].strip() if text.startswith(prompt) else text.strip()
            prompt_tokens = int(inputs["input_ids"].shape[-1])
            completion_tokens = max(0, int(output_ids.shape[-1]) - prompt_tokens)
        else:
            toks = self.tokenizer([prompt], return_tensors="pt", truncation=True, max_length=self.settings.max_seq_len)
            input_ids = toks["input_ids"].to(self.settings.device)
            kwargs = {
                "max_new_tokens": max_tokens,
                "use_cache": False,
            }
            if temperature > 0:
                kwargs["do_sample"] = True
                kwargs["temperature"] = temperature
                kwargs["top_p"] = top_p
            output_ids = self.model.generate(input_ids, **kwargs)
            text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            completion = text[len(prompt) :].strip() if text.startswith(prompt) else text.strip()
            prompt_tokens = int(input_ids.shape[-1])
            completion_tokens = max(0, int(output_ids.shape[-1]) - prompt_tokens)

        return {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.loaded_model_id,
            "completion_text": completion,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }

    def generate_completion(self, prompt: str, model_id: str | None, max_tokens: int, temperature: float, top_p: float):
        self.load_model_if_needed(model_id)
        toks = self.tokenizer([prompt], return_tensors="pt", truncation=True, max_length=self.settings.max_seq_len)
        input_ids = toks["input_ids"].to(self.settings.device)

        kwargs = {
            "max_new_tokens": max_tokens,
            "use_cache": False,
        }
        if temperature > 0:
            kwargs["do_sample"] = True
            kwargs["temperature"] = temperature
            kwargs["top_p"] = top_p

        output_ids = self.model.generate(input_ids, **kwargs)
        text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        completion = text[len(prompt) :].strip() if text.startswith(prompt) else text.strip()

        prompt_tokens = int(input_ids.shape[-1])
        completion_tokens = max(0, int(output_ids.shape[-1]) - prompt_tokens)

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
        self.load_model_if_needed(model_id)
        if not hasattr(self.model, "tts"):
            raise RuntimeError("Current model backend does not support text-to-speech")

        wav_tensor = self.model.tts(text)
        wav_np = wav_tensor.detach().cpu().numpy()

        bio = io.BytesIO()
        wav_write(bio, 16000, wav_np)
        return bio.getvalue()
