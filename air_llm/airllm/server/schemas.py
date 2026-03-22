from typing import Literal, Optional, Union

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
    content: Union[str, list[ChatContentPart]]


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = None
    messages: list[ChatMessage]
    max_tokens: Optional[int] = Field(default=None, ge=1)
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    stream: bool = False


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
