import argparse
import json
import os
import sys
import uvicorn

from .config import Settings
from .model_store import ModelStore

# ANSI colour codes for the run command output.
_ANSI_SYSTEM = "\033[2;33m"  # dim yellow
_ANSI_USER = "\033[1;32m"    # bright green
_ANSI_AI = "\033[1;34m"      # bright blue
_ANSI_THINK = "\033[2;36m"   # dim cyan  – thinking / <think> content
_ANSI_ANSWER = _ANSI_AI
_ANSI_RESET = "\033[0m"


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

    ls: argparse.ArgumentParser = sub.add_parser("ls", help="List locally available models")
    ls.add_argument("--json", action="store_true", help="Output JSON")

    # Keep 'models' as a hidden backward-compatible alias.
    models: argparse.ArgumentParser = sub.add_parser("models", help=argparse.SUPPRESS)
    models.add_argument("--json", action="store_true", help=argparse.SUPPRESS)

    rm: argparse.ArgumentParser = sub.add_parser("rm", help="Remove model from local cache")
    rm.add_argument("model", help="Model id to remove")

    run: argparse.ArgumentParser = sub.add_parser(
        "run",
        help="Load a model and start an interactive chat (Ollama-style)",
    )
    run.add_argument("model", help="Model id to run")
    run.add_argument(
        "--hf-token",
        default=None,
        help="Hugging Face access token (defaults to HF_TOKEN env var)",
    )
    run.add_argument("--max-tokens", type=int, default=None, help="Max new tokens per reply")
    run.add_argument("--temperature", type=float, default=None, help="Sampling temperature")
    run.add_argument("--top-p", type=float, default=None, help="Top-p sampling")
    run.add_argument(
        "--system",
        default=None,
        help="Override system prompt for this run session",
    )
    run.add_argument(
        "--no-color",
        action="store_true",
        help="Disable colored thinking/answer output",
    )
    run.add_argument(
        "--chat-template",
        default=None,
        metavar="PATH_OR_MODE",
        help=(
            "Jinja2 chat-template control. Accepted values: a path to a .jinja "
            "file; 'none' to disable templating; empty to use the model's built-in template."
        ),
    )

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


def _cmd_run(args: argparse.Namespace) -> int:  # noqa: C901
    """Interactive chat loop, printing <think> blocks and answers in different colors."""
    import re
    import threading

    def role_label(role: str, use_color_local: bool) -> str:
        role_l = role.lower()
        if not use_color_local:
            return f"{role_l}> "
        if role_l == "system":
            return f"{_ANSI_SYSTEM}system>{_ANSI_RESET} "
        if role_l == "assistant":
            return f"{_ANSI_AI}assistant>{_ANSI_RESET} "
        return f"{_ANSI_USER}user>{_ANSI_RESET} "

    def user_input_prompt(use_color_local: bool) -> str:
        if not use_color_local:
            return "user> "
        # Keep input text green while the user types, then reset afterwards.
        return f"{_ANSI_USER}user> "

    def maybe_strip_prefill(text: str) -> str:
        """Drop model-echoed chat scaffold prefixes often seen in some templates."""
        lowered = text.lower()
        markers = ["\nassistant\n", "\nassistant:\n", "\nassistant:", "assistant\n", "assistant:"]
        cut_idx = -1
        cut_len = 0
        for marker in markers:
            i = lowered.rfind(marker)
            if i > cut_idx:
                cut_idx = i
                cut_len = len(marker)
        if cut_idx >= 0:
            return text[cut_idx + cut_len :]

        # If explicit role headings are present but assistant start is not yet
        # visible, hold a short buffer window to avoid showing template prefill.
        role_tokens = (
            "\nsystem\n",
            "\nuser\n",
            "\nassistant\n",
            "system\n",
            "user\n",
            "system:",
            "user:",
            "assistant:",
        )
        if any(tok in lowered for tok in role_tokens) and len(text) < 400:
            return ""
        return text

    settings = Settings()
    if getattr(args, "hf_token", None):
        settings.hf_token = args.hf_token
    if getattr(args, "chat_template", None) is not None:
        settings.chat_template = args.chat_template

    settings.model_id = args.model

    max_tokens_arg = getattr(args, "max_tokens", None)
    temperature_arg = getattr(args, "temperature", None)
    top_p_arg = getattr(args, "top_p", None)

    max_tokens: int = max_tokens_arg if max_tokens_arg is not None else settings.max_new_tokens
    temperature: float = temperature_arg if temperature_arg is not None else settings.temperature
    top_p: float = top_p_arg if top_p_arg is not None else settings.top_p
    use_color: bool = not getattr(args, "no_color", False)

    from .runner import ServerRunner
    from transformers import TextIteratorStreamer

    runner = ServerRunner(settings)

    think_start = _ANSI_THINK if use_color else ""
    think_end = _ANSI_RESET if use_color else ""
    answer_start = _ANSI_ANSWER if use_color else ""
    answer_end = _ANSI_RESET if use_color else ""

    def _print_stream(streamer: TextIteratorStreamer, thread: threading.Thread) -> str:
        """Print tokens as they arrive, switching color between <think> and answer."""
        # States: "answer" | "think"
        state = "answer"
        buffer = ""   # partial tag accumulator
        full_text = ""
        prefill_buffer = ""
        prefill_done = False
        printed_assistant_label = False

        in_think_re = re.compile(r"<think>", re.IGNORECASE)
        out_think_re = re.compile(r"</think>", re.IGNORECASE)

        try:
            for token in streamer:
                full_text += token
                buffer += token
                output = ""

                # Process the buffer looking for <think> / </think> transitions.
                while buffer:
                    if state == "answer":
                        m = in_think_re.search(buffer)
                        if m:
                            pre = buffer[: m.start()]
                            if pre:
                                output += pre
                            output += answer_end + think_start
                            state = "think"
                            buffer = buffer[m.end() :]
                        else:
                            # Might be the start of a partial tag at the tail.
                            if "<" in buffer:
                                last_lt = buffer.rfind("<")
                                output += buffer[:last_lt]
                                buffer = buffer[last_lt:]
                                break
                            else:
                                output += buffer
                                buffer = ""
                    else:  # state == "think"
                        m = out_think_re.search(buffer)
                        if m:
                            pre = buffer[: m.start()]
                            if pre:
                                output += pre
                            output += think_end + answer_start
                            state = "answer"
                            buffer = buffer[m.end() :]
                        else:
                            if "<" in buffer:
                                last_lt = buffer.rfind("<")
                                output += buffer[:last_lt]
                                buffer = buffer[last_lt:]
                                break
                            else:
                                output += buffer
                                buffer = ""

                if output:
                    if not prefill_done:
                        prefill_buffer += output
                        cleaned = maybe_strip_prefill(prefill_buffer)
                        if cleaned:
                            output = cleaned
                            prefill_done = True
                            prefill_buffer = ""
                        else:
                            output = ""

                if output:
                    if not printed_assistant_label:
                        sys.stdout.write(role_label("assistant", use_color))
                        sys.stdout.write(answer_start)
                        printed_assistant_label = True
                    sys.stdout.write(output)
                    sys.stdout.flush()
        finally:
            thread.join()

        # Flush remaining buffer (incomplete tag at EOF).
        if buffer:
            if not printed_assistant_label:
                sys.stdout.write(role_label("assistant", use_color))
                sys.stdout.write(answer_start)
                printed_assistant_label = True
            sys.stdout.write(buffer)
        elif prefill_buffer and not prefill_done:
            cleaned = maybe_strip_prefill(prefill_buffer)
            if cleaned:
                if not printed_assistant_label:
                    sys.stdout.write(role_label("assistant", use_color))
                    sys.stdout.write(answer_start)
                    printed_assistant_label = True
                sys.stdout.write(cleaned)
        sys.stdout.write(answer_end + "\n")
        sys.stdout.flush()
        cleaned_full = maybe_strip_prefill(full_text)
        return cleaned_full if cleaned_full else full_text

    print(f"Loading {args.model} …", flush=True)
    runner.load_model_if_needed(args.model)
    print(f"Model ready. Type your message and press Enter. Use /bye or Ctrl-D to quit.\n", flush=True)

    conversation: list[dict] = []
    if getattr(args, "system", None):
        system_prompt = args.system.strip()
        if system_prompt:
            conversation.append({"role": "system", "content": system_prompt})
            print(f"{role_label('system', use_color)}{system_prompt}")

    while True:
        try:
            user_input = input(user_input_prompt(use_color)).strip()
            if use_color:
                sys.stdout.write(_ANSI_RESET)
                sys.stdout.flush()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_input:
            continue
        if user_input.lower() in ("/bye", "/exit", "/quit"):
            break

        conversation.append({"role": "user", "content": user_input})

        try:
            meta, streamer, thread = runner.generate_chat_streaming(
                messages=conversation,
                model_id=args.model,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                suppress_output=True,
            )
        except Exception as exc:
            print(f"\nStreaming unavailable ({exc}), falling back to single-shot…", flush=True)
            response = runner.generate_chat(
                messages=conversation,
                model_id=args.model,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                suppress_output=True,
            )
            full_text = maybe_strip_prefill(response["completion_text"])
            # Colour the output manually for single-shot too.
            _print_static(full_text, use_color, role_label("assistant", use_color))
        else:
            try:
                full_text = _print_stream(streamer, thread)
            except Exception as exc:
                print(f"\nStreaming interrupted ({exc}), falling back to single-shot…", flush=True)
                response = runner.generate_chat(
                    messages=conversation,
                    model_id=args.model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    suppress_output=True,
                )
                full_text = maybe_strip_prefill(response["completion_text"])
                _print_static(full_text, use_color, role_label("assistant", use_color))

        conversation.append({"role": "assistant", "content": full_text})

    return 0


def _print_static(text: str, use_color: bool, assistant_label: str = "") -> None:
    """Print a complete response string with think/answer colouring."""
    import re

    if assistant_label:
        sys.stdout.write(assistant_label)

    if not use_color:
        print(text)
        return

    think_re = re.compile(r"(<think>)(.*?)(</think>)", re.IGNORECASE | re.DOTALL)
    pos = 0
    for m in think_re.finditer(text):
        if m.start() > pos:
            sys.stdout.write(_ANSI_ANSWER + text[pos : m.start()] + _ANSI_RESET)
        sys.stdout.write(_ANSI_THINK + m.group(2) + _ANSI_RESET)
        pos = m.end()
    if pos < len(text):
        sys.stdout.write(_ANSI_ANSWER + text[pos:] + _ANSI_RESET)
    sys.stdout.write("\n")
    sys.stdout.flush()


def main(argv: list[str] | None = None) -> int:
    original_argv = list(argv) if argv is not None else sys.argv[1:]
    parser: argparse.ArgumentParser = _build_parser()
    args: argparse.Namespace = parser.parse_args(original_argv)

    if args.command == "serve":
        return _cmd_serve(args)
    if args.command == "pull":
        return _cmd_pull(args)
    if args.command in ("models", "ls"):
        return _cmd_models(args)
    if args.command == "rm":
        return _cmd_rm(args)
    if args.command == "run":
        return _cmd_run(args)

    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
