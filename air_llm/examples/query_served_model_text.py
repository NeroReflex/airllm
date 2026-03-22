import argparse
import json
import sys

import requests


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Query an AirLLM served model over HTTP.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="Server base URL")
    parser.add_argument("--model", required=True, help="Model id exposed by the server")
    parser.add_argument("--prompt", required=True, help="User prompt to send")
    parser.add_argument("--api-key", default=None, help="Optional bearer token")
    parser.add_argument("--max-tokens", type=int, default=128, help="Maximum new tokens")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature")
    parser.add_argument("--timeout", type=float, default=300.0, help="HTTP timeout in seconds")
    return parser


def main() -> int:
    args = build_parser().parse_args()

    headers = {"Content-Type": "application/json"}
    if args.api_key:
        headers["Authorization"] = f"Bearer {args.api_key}"

    payload = {
        "model": args.model,
        "messages": [
            {"role": "user", "content": args.prompt},
        ],
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
    }

    try:
        response = requests.post(
            f"{args.base_url.rstrip('/')}/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=args.timeout,
        )
    except requests.RequestException as exc:
        print(f"Request failed: {exc}", file=sys.stderr)
        print("Tip: ensure the server is running and model loading has completed.", file=sys.stderr)
        return 3

    print(f"status_code={response.status_code}")
    if not response.ok:
        print(response.text)
        return 1

    body = response.json()
    print(json.dumps(body, indent=2, ensure_ascii=True))

    try:
        content = body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        print("Response did not contain choices[0].message.content", file=sys.stderr)
        return 2

    print("\ncompletion:\n")
    print(content)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())