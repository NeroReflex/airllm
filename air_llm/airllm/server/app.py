import json
from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, status
from fastapi.responses import JSONResponse, Response, StreamingResponse

from .config import Settings
from .model_store import ModelStore
from .runner import ServerRunner
from .schemas import (
    AudioSpeechRequest,
    AudioTranscriptionRequest,
    ChatCompletionRequest,
    CompletionRequest,
    DeleteRequest,
    ModelInfo,
    ModelListResponse,
    PullRequest,
)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    runner = ServerRunner(settings)
    store = ModelStore(settings.cache_dir, hf_token=settings.hf_token)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncGenerator[None, None]:
        if not settings.lazy_load_model:
            runner.load_model_if_needed(settings.model_id)
        yield

    app = FastAPI(
        title="AirLLM OpenAI-Compatible API",
        version="0.1.0",
        description="OpenAI-compatible API server embedded in AirLLM",
        lifespan=lifespan,
    )

    def check_auth(authorization: str | None = Header(default=None)) -> None:
        if not settings.enforce_auth:
            return

        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing bearer token")

        token: str = authorization.replace("Bearer ", "", 1).strip()
        if token != settings.api_key:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/models", dependencies=[Depends(check_auth)], response_model=ModelListResponse)
    async def list_models() -> ModelListResponse:
        local: list[str] = store.list_local_models()
        if settings.model_id not in local:
            local.insert(0, settings.model_id)
        data: list[ModelInfo] = [ModelInfo(id=m) for m in dict.fromkeys(local)]
        return ModelListResponse(data=data)

    async def stream_chat(data: dict[str, Any]) -> AsyncGenerator[bytes, None]:
        delta: dict[str, Any] = {"role": "assistant", "content": data["completion_text"]}
        if data.get("tool_calls"):
            delta["tool_calls"] = data["tool_calls"]
        chunk = {
            "id": data["id"],
            "object": "chat.completion.chunk",
            "created": data["created"],
            "model": data["model"],
            "choices": [
                {
                    "index": 0,
                    "delta": delta,
                    "finish_reason": data.get("finish_reason", "stop"),
                }
            ],
        }
        payload: str = f"data: {json.dumps(chunk)}\\n\\n"
        yield payload.encode("utf-8")
        yield b"data: [DONE]\\n\\n"

    async def stream_completion(data: dict[str, Any]) -> AsyncGenerator[bytes, None]:
        chunk = {
            "id": data["id"],
            "object": "text_completion",
            "created": data["created"],
            "model": data["model"],
            "choices": [
                {
                    "index": 0,
                    "text": data["completion_text"],
                    "finish_reason": "stop",
                }
            ],
        }
        payload: str = f"data: {json.dumps(chunk)}\\n\\n"
        yield payload.encode("utf-8")
        yield b"data: [DONE]\\n\\n"

    @app.post("/v1/chat/completions", dependencies=[Depends(check_auth)], response_model=None)
    async def chat_completions(req: ChatCompletionRequest) -> Response:
        response = runner.generate_chat(
            messages=[m.model_dump(mode="json") for m in req.messages],
            model_id=req.model,
            temperature=req.temperature if req.temperature is not None else settings.temperature,
            top_p=req.top_p if req.top_p is not None else settings.top_p,
            max_tokens=req.max_tokens if req.max_tokens is not None else settings.max_new_tokens,
            tools=req.tools or None,
            tool_choice=req.tool_choice,
        )

        body = {
            "id": response["id"],
            "object": "chat.completion",
            "created": response["created"],
            "model": response["model"],
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response["completion_text"] or None,
                        "tool_calls": response.get("tool_calls") or None,
                    },
                    "finish_reason": response.get("finish_reason", "stop"),
                }
            ],
            "usage": response["usage"],
        }

        if req.stream:
            return StreamingResponse(stream_chat(response), media_type="text/event-stream")

        return JSONResponse(body)

    @app.post("/v1/completions", dependencies=[Depends(check_auth)], response_model=None)
    async def completions(req: CompletionRequest) -> Response:
        response = runner.generate_completion(
            prompt=req.prompt,
            model_id=req.model,
            temperature=req.temperature if req.temperature is not None else settings.temperature,
            top_p=req.top_p if req.top_p is not None else settings.top_p,
            max_tokens=req.max_tokens if req.max_tokens is not None else settings.max_new_tokens,
        )

        body = {
            "id": response["id"],
            "object": "text_completion",
            "created": response["created"],
            "model": response["model"],
            "choices": [
                {
                    "index": 0,
                    "text": response["completion_text"],
                    "finish_reason": "stop",
                }
            ],
            "usage": response["usage"],
        }

        if req.stream:
            return StreamingResponse(stream_completion(response), media_type="text/event-stream")

        return JSONResponse(body)

    @app.post("/v1/audio/speech", dependencies=[Depends(check_auth)])
    async def audio_speech(req: AudioSpeechRequest) -> Response:
        try:
            wav_bytes: bytes = runner.synthesize_speech(req.input, req.model)
        except RuntimeError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        return Response(content=wav_bytes, media_type="audio/wav")

    @app.post("/v1/audio/transcriptions", dependencies=[Depends(check_auth)], response_model=None)
    async def audio_transcriptions(req: AudioTranscriptionRequest) -> Response:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Speech-to-text is not implemented in AirLLM server yet.",
        )

    # Ollama-like utility endpoints
    @app.get("/api/tags", dependencies=[Depends(check_auth)])
    async def api_tags() -> dict[str, list[dict[str, Any]]]:
        return {
            "models": [
                {"name": m, "model": m, "modified_at": None, "size": None}
                for m in store.list_local_models()
            ]
        }

    @app.post("/api/pull", dependencies=[Depends(check_auth)])
    async def api_pull(req: PullRequest) -> dict[str, str]:
        local_path: str = store.pull(req.model)
        return {"status": "success", "model": req.model, "local_path": local_path}

    @app.delete("/api/delete", dependencies=[Depends(check_auth)])
    async def api_delete(req: DeleteRequest) -> dict[str, str]:
        deleted: bool = store.remove(req.model)
        if not deleted:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Model not found: {req.model}")
        return {"status": "deleted", "model": req.model}

    return app


app: FastAPI = create_app()
