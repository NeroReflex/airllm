import argparse
import json
import sys
import uvicorn

from .config import Settings
from .model_store import ModelStore


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="airllm", description="AirLLM server and model utilities")
    sub: argparse._SubParsersAction[argparse.ArgumentParser] = parser.add_subparsers(dest="command", required=True)

    serve: argparse.ArgumentParser = sub.add_parser("serve", help="Run OpenAI-compatible AirLLM server")
    serve.add_argument("--model", default=None, help="Default model id")
    serve.add_argument("--host", default=None, help="Bind host")
    serve.add_argument("--port", type=int, default=None, help="Bind port")
    serve.add_argument("--api-key", default=None, help="Bearer token required for auth")
    serve.add_argument(
        "--chat-template",
        default=None,
        metavar="PATH_OR_MODE",
        help=(
            "Jinja2 chat-template control (same as vLLM --chat-template / "
            "llama.cpp --jinja). Accepted values: a path to a .jinja file; "
            "'none' to disable templating and use legacy ROLE: content format; "
            "empty / omitted to use the model's built-in tokenizer template."
        ),
    )

    pull: argparse.ArgumentParser = sub.add_parser("pull", help="Download a model to local cache")
    pull.add_argument("model", help="Model id to pull (e.g. meta-llama/Llama-3.1-8B-Instruct)")
    pull.add_argument(
        "--hf-token",
        default=None,
        help="Hugging Face access token (defaults to HF_TOKEN env var)",
    )

    models: argparse.ArgumentParser = sub.add_parser("models", help="List locally available models")
    models.add_argument("--json", action="store_true", help="Output JSON")

    rm: argparse.ArgumentParser = sub.add_parser("rm", help="Remove model from local cache")
    rm.add_argument("model", help="Model id to remove")

    return parser


def _settings_with_overrides(args: argparse.Namespace) -> Settings:
    settings = Settings()
    if getattr(args, "model", None):
        settings.model_id = args.model
    if getattr(args, "host", None):
        settings.host = args.host
    if getattr(args, "port", None):
        settings.port = args.port
    if getattr(args, "api_key", None):
        settings.api_key = args.api_key
        settings.enforce_auth = True
    if getattr(args, "chat_template", None) is not None:
        settings.chat_template = args.chat_template
    return settings


def _cmd_serve(args: argparse.Namespace) -> int:
    from .app import create_app

    settings: Settings = _settings_with_overrides(args)
    app = create_app(settings)
    uvicorn.run(app, host=settings.host, port=settings.port)
    return 0


def _cmd_pull(args: argparse.Namespace) -> int:
    settings = Settings()
    hf_token: str = args.hf_token if args.hf_token is not None else settings.hf_token
    store = ModelStore(settings.cache_dir, hf_token=hf_token)
    path: str = store.pull(args.model)
    print(f"Pulled {args.model} to {path}")
    return 0


def _cmd_models(args: argparse.Namespace) -> int:
    settings = Settings()
    store = ModelStore(settings.cache_dir, hf_token=settings.hf_token)
    models: list[str] = store.list_local_models()

    if args.json:
        print(json.dumps({"models": models}, indent=2))
    else:
        if not models:
            print("No local models found.")
            return 0
        for model in models:
            print(model)
    return 0


def _cmd_rm(args: argparse.Namespace) -> int:
    settings = Settings()
    store = ModelStore(settings.cache_dir, hf_token=settings.hf_token)
    removed: bool = store.remove(args.model)
    if removed:
        print(f"Removed {args.model}")
        return 0
    print(f"Model not found: {args.model}", file=sys.stderr)
    return 1


def main(argv: list[str] | None = None) -> int:
    parser: argparse.ArgumentParser = _build_parser()
    args: argparse.Namespace = parser.parse_args(argv)

    if args.command == "serve":
        return _cmd_serve(args)
    if args.command == "pull":
        return _cmd_pull(args)
    if args.command == "models":
        return _cmd_models(args)
    if args.command == "rm":
        return _cmd_rm(args)

    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
