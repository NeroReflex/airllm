from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, Field


class ImageURL(BaseModel):
    url: str
    detail: Optional[str] = None


class ChatContentPart(BaseModel):
    type: Literal["text", "image_url"]
    text: Optional[str] = None
    image_url: Optional[ImageURL] = None


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    # content may be None for tool-result messages or structured assistant turns
    content: Optional[Union[str, list[ChatContentPart], list[dict[str, Any]]]] = None
    name: Optional[str] = None
    tool_call_id: Optional[str] = None
    # Preserve assistant tool calls across turns (required by some templates,
    # including MiniMax-M2.5).
    tool_calls: Optional[list[dict[str, Any]]] = None
    reasoning_content: Optional[str] = None
    current_date: Optional[str] = None
    current_location: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = None
    messages: list[ChatMessage]
    max_tokens: Optional[int] = Field(default=None, ge=1)
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    stream: bool = False
    # Tool / function-calling fields (passed through to apply_chat_template for
    # models whose Jinja2 template supports them, e.g. Llama 3.1+, Qwen 2.5+).
    tools: Optional[list[dict[str, Any]]] = None
    tool_choice: Optional[Union[str, dict[str, Any]]] = None
    # Generation controls forwarded by most OpenAI-compatible clients.
    stop: Optional[Union[str, list[str]]] = None
    seed: Optional[int] = None


class CompletionRequest(BaseModel):
    model: Optional[str] = None
    prompt: str
    max_tokens: Optional[int] = Field(default=None, ge=1)
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    stream: bool = False


class AudioSpeechRequest(BaseModel):
    model: Optional[str] = None
    input: str
    voice: Optional[str] = "alloy"
    response_format: Optional[Literal["wav"]] = "wav"


class AudioTranscriptionRequest(BaseModel):
    model: Optional[str] = None


class ModelInfo(BaseModel):
    id: str
    object: Literal["model"] = "model"
    owned_by: str = "airllm"


class ModelListResponse(BaseModel):
    object: Literal["list"] = "list"
    data: list[ModelInfo]


class PullRequest(BaseModel):
    model: str


class DeleteRequest(BaseModel):
    model: str
