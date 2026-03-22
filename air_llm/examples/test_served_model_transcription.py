import argparse
import json
import sys

import requests


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Exercise the AirLLM transcription HTTP endpoint.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="Server base URL")
    parser.add_argument("--model", required=True, help="Speech-to-text model id to request")
    parser.add_argument("--api-key", default=None, help="Optional bearer token")
    parser.add_argument("--timeout", type=float, default=30.0, help="HTTP timeout in seconds")
    return parser


def main() -> int:
    args = build_parser().parse_args()

    headers = {}
    if args.api_key:
        headers["Authorization"] = f"Bearer {args.api_key}"

    payload = {
        "model": args.model,
    }

    try:
        response = requests.post(
            f"{args.base_url.rstrip('/')}/v1/audio/transcriptions",
            headers=headers,
            json=payload,
            timeout=args.timeout,
        )
    except requests.RequestException as exc:
        print(f"Request failed: {exc}", file=sys.stderr)
        print("Tip: the server may still be loading a large model.", file=sys.stderr)
        return 3

    print(f"status_code={response.status_code}")
    content_type = response.headers.get("content-type", "")
    if "application/json" in content_type:
        print(json.dumps(response.json(), indent=2, ensure_ascii=True))
    else:
        print(response.text)

    # Current server behavior is 501 until an STT backend is implemented.
    if response.status_code == 501:
        return 0

    return 0 if response.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())