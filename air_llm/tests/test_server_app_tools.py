import json
import unittest
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from ..airllm.server import app as app_module
from ..airllm.server.config import Settings


class TestChatCompletionsToolSupport(unittest.TestCase):
    def setUp(self):
        self.runner_patcher = patch.object(app_module, "ServerRunner")
        self.store_patcher = patch.object(app_module, "ModelStore")

        self.mock_runner_cls = self.runner_patcher.start()
        self.mock_store_cls = self.store_patcher.start()

        self.runner = MagicMock()
        self.mock_runner_cls.return_value = self.runner
        self.mock_store_cls.return_value = MagicMock(list_local_models=MagicMock(return_value=[]))

        settings = Settings()
        settings.enforce_auth = False
        settings.lazy_load_model = True

        self.client = TestClient(app_module.create_app(settings))

    def tearDown(self):
        self.store_patcher.stop()
        self.runner_patcher.stop()

    def test_startup_loads_model_when_lazy_loading_disabled(self):
        settings = Settings()
        settings.enforce_auth = False
        settings.lazy_load_model = False
        settings.model_id = "MiniMaxAI/MiniMax-M2.5"

        with TestClient(app_module.create_app(settings)) as client:
            response = client.get("/healthz")

        self.assertEqual(response.status_code, 200)
        self.runner.load_model_if_needed.assert_called_with("MiniMaxAI/MiniMax-M2.5")

    def test_non_stream_returns_tool_calls_and_finish_reason(self):
        self.runner.generate_chat.return_value = {
            "id": "chatcmpl-test",
            "created": 123,
            "model": "MiniMaxAI/MiniMax-M2.5",
            "completion_text": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"city": "Paris"}',
                    },
                }
            ],
            "finish_reason": "tool_calls",
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        }

        req_tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                },
            }
        ]
        req_tool_choice = {"type": "function", "function": {"name": "get_weather"}}

        response = self.client.post(
            "/v1/chat/completions",
            json={
                "model": "MiniMaxAI/MiniMax-M2.5",
                "messages": [{"role": "user", "content": "weather in paris"}],
                "stream": False,
                "tools": req_tools,
                "tool_choice": req_tool_choice,
            },
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()

        self.assertEqual(payload["choices"][0]["finish_reason"], "tool_calls")
        self.assertIsNone(payload["choices"][0]["message"]["content"])
        self.assertEqual(payload["choices"][0]["message"]["tool_calls"][0]["function"]["name"], "get_weather")

        self.runner.generate_chat.assert_called_once()
        kwargs = self.runner.generate_chat.call_args.kwargs
        self.assertEqual(kwargs["tools"], req_tools)
        self.assertEqual(kwargs["tool_choice"], req_tool_choice)

    def test_stream_returns_tool_calls_delta_and_finish_reason(self):
        self.runner.generate_chat.return_value = {
            "id": "chatcmpl-test",
            "created": 123,
            "model": "MiniMaxAI/MiniMax-M2.5",
            "completion_text": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "search",
                        "arguments": '{"query": "news"}',
                    },
                }
            ],
            "finish_reason": "tool_calls",
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        }

        response = self.client.post(
            "/v1/chat/completions",
            json={
                "model": "MiniMaxAI/MiniMax-M2.5",
                "messages": [{"role": "user", "content": "latest news"}],
                "stream": True,
            },
        )

        self.assertEqual(response.status_code, 200)
        lines = [line for line in response.text.splitlines() if line.startswith("data: ")]

        self.assertGreaterEqual(len(lines), 1)

        first_payload = lines[0].replace("data: ", "", 1)
        first_payload = first_payload.split("\n\n", 1)[0]
        first_payload = first_payload.split("\\n\\ndata: [DONE]", 1)[0]
        first = json.loads(first_payload)
        choice = first["choices"][0]
        self.assertEqual(choice["finish_reason"], "tool_calls")
        self.assertIn("tool_calls", choice["delta"])
        self.assertEqual(choice["delta"]["tool_calls"][0]["function"]["name"], "search")

    def test_non_stream_stop_omits_tool_calls(self):
        self.runner.generate_chat.return_value = {
            "id": "chatcmpl-test",
            "created": 123,
            "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "completion_text": "hello",
            "tool_calls": [],
            "finish_reason": "stop",
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 2,
                "total_tokens": 12,
            },
        }

        response = self.client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "hi"}],
                "stream": False,
            },
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()

        self.assertEqual(payload["choices"][0]["finish_reason"], "stop")
        self.assertEqual(payload["choices"][0]["message"]["content"], "hello")
        self.assertIsNone(payload["choices"][0]["message"].get("tool_calls"))
