import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from ..airllm.server.config import Settings
from ..airllm.server.runner import ServerRunner, _HarmonyFinalChannelStreamer


def _runner(max_seq_len=None, hf_token=""):
    settings = Settings()
    settings.max_seq_len = max_seq_len
    settings.hf_token = hf_token
    return ServerRunner(settings)


def _fake_config(**kwargs) -> MagicMock:
    cfg = MagicMock(spec=[])
    for key, value in kwargs.items():
        setattr(cfg, key, value)
    return cfg


class TestInferMaxSeqLen(unittest.TestCase):
    """Unit tests for ServerRunner._infer_max_seq_len_from_model."""

    def _infer(self, config_obj):
        runner = _runner()
        with patch(
            "airllm.server.runner.AutoConfig.from_pretrained",
            return_value=config_obj,
        ):
            return runner._infer_max_seq_len_from_model("some-model")

    # --- standard single-field cases ------------------------------------------

    def test_max_position_embeddings(self):
        cfg = _fake_config(max_position_embeddings=8192)
        self.assertEqual(self._infer(cfg), 8192)

    def test_n_positions(self):
        cfg = _fake_config(n_positions=2048)
        self.assertEqual(self._infer(cfg), 2048)

    def test_seq_length(self):
        cfg = _fake_config(seq_length=4096)
        self.assertEqual(self._infer(cfg), 4096)

    def test_max_sequence_length(self):
        cfg = _fake_config(max_sequence_length=16384)
        self.assertEqual(self._infer(cfg), 16384)

    def test_max_seq_len_key(self):
        cfg = _fake_config(max_seq_len=32768)
        self.assertEqual(self._infer(cfg), 32768)

    def test_max_seq_length_key(self):
        cfg = _fake_config(max_seq_length=1024)
        self.assertEqual(self._infer(cfg), 1024)

    def test_context_length(self):
        cfg = _fake_config(context_length=131072)
        self.assertEqual(self._infer(cfg), 131072)

    def test_n_ctx(self):
        cfg = _fake_config(n_ctx=4096)
        self.assertEqual(self._infer(cfg), 4096)

    # --- picks the largest when multiple keys are present ----------------------

    def test_picks_largest_among_multiple_keys(self):
        cfg = _fake_config(max_position_embeddings=2048, context_length=8192, n_ctx=512)
        self.assertEqual(self._infer(cfg), 8192)

    # --- rope_scaling cases ----------------------------------------------------

    def test_rope_scaling_max_position_embeddings(self):
        cfg = _fake_config(
            max_position_embeddings=4096,
            rope_scaling={"max_position_embeddings": 131072},
        )
        self.assertEqual(self._infer(cfg), 131072)

    def test_rope_scaling_factor_derived(self):
        cfg = _fake_config(
            max_position_embeddings=4096,
            rope_scaling={
                "original_max_position_embeddings": 4096,
                "factor": 8.0,
            },
        )
        # 4096 * 8 = 32768 which is larger than max_position_embeddings=4096
        self.assertEqual(self._infer(cfg), 32768)

    def test_rope_scaling_factor_int(self):
        cfg = _fake_config(
            rope_scaling={
                "original_max_position_embeddings": 2048,
                "factor": 4,
            }
        )
        self.assertEqual(self._infer(cfg), 8192)

    def test_rope_scaling_factor_missing_skipped(self):
        # If only original_max_position_embeddings but no factor, don't compute.
        cfg = _fake_config(
            max_position_embeddings=2048,
            rope_scaling={"original_max_position_embeddings": 2048},
        )
        self.assertEqual(self._infer(cfg), 2048)

    # --- sanity cap / edge cases -----------------------------------------------

    def test_giant_value_is_filtered_out(self):
        # Values >= 10_000_000 (placeholder infinities) should be ignored.
        cfg = _fake_config(max_position_embeddings=10_000_000, context_length=4096)
        self.assertEqual(self._infer(cfg), 4096)

    def test_only_giant_value_falls_back(self):
        cfg = _fake_config(max_position_embeddings=10_000_000)
        self.assertEqual(self._infer(cfg), 1024)

    def test_no_relevant_keys_falls_back_to_1024(self):
        cfg = _fake_config(vocab_size=32000, hidden_size=4096)
        self.assertEqual(self._infer(cfg), 1024)

    def test_config_load_failure_falls_back_to_1024(self):
        runner = _runner()
        with patch(
            "airllm.server.runner.AutoConfig.from_pretrained",
            side_effect=OSError("not found"),
        ):
            self.assertEqual(runner._infer_max_seq_len_from_model("missing-model"), 1024)

    def test_hf_token_forwarded(self):
        runner = _runner(hf_token="my-token")
        cfg = _fake_config(max_position_embeddings=8192)
        with patch(
            "airllm.server.runner.AutoConfig.from_pretrained",
            return_value=cfg,
        ) as mock_fp:
            result = runner._infer_max_seq_len_from_model("gated-model")
        mock_fp.assert_called_once_with("gated-model", token="my-token", trust_remote_code=True)
        self.assertEqual(result, 8192)

    def test_no_hf_token_not_forwarded(self):
        runner = _runner(hf_token="")
        cfg = _fake_config(max_position_embeddings=4096)
        with patch(
            "airllm.server.runner.AutoConfig.from_pretrained",
            return_value=cfg,
        ) as mock_fp:
            result = runner._infer_max_seq_len_from_model("open-model")
        mock_fp.assert_called_once_with("open-model", trust_remote_code=True)
        self.assertEqual(result, 4096)

    # --- Settings integration --------------------------------------------------

    def test_explicit_max_seq_len_skips_inference(self):
        """When Settings.max_seq_len is set, inference should not be called."""
        runner = _runner(max_seq_len=512)
        runner.model = MagicMock()
        runner.loaded_model_id = "some-model"
        runner.effective_max_seq_len = 512

        with patch.object(runner, "_infer_max_seq_len_from_model") as mock_infer:
            # load_model_if_needed returns early because model is already loaded.
            runner.load_model_if_needed("some-model")
            mock_infer.assert_not_called()

        self.assertEqual(runner.effective_max_seq_len, 512)


def _runner_with_tokenizer(chat_template_setting="", tokenizer=None):
    """Build a pre-loaded ServerRunner with a stub tokenizer."""
    settings = Settings()
    settings.chat_template = chat_template_setting
    runner = ServerRunner(settings)
    runner.tokenizer = tokenizer or MagicMock()
    runner.model = MagicMock()
    runner.loaded_model_id = "stub-model"
    runner.effective_max_seq_len = 1024
    return runner


MESSAGES = [
    {"role": "user", "content": "Hello, world!"},
]

MULTI_MESSAGES = [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "What is the weather?"},
]


class TestApplyChatTemplate(unittest.TestCase):
    """Unit tests for ServerRunner._apply_chat_template."""

    # ------------------------------------------------------------------
    # Built-in tokenizer template (default / auto)
    # ------------------------------------------------------------------

    def test_uses_tokenizer_apply_chat_template(self):
        tok = MagicMock()
        tok.apply_chat_template.return_value = "<|user|>Hello, world!</s>"
        runner = _runner_with_tokenizer(chat_template_setting="", tokenizer=tok)

        prompt, used = runner._apply_chat_template(MESSAGES)

        self.assertTrue(used)
        self.assertEqual(prompt, "<|user|>Hello, world!</s>")
        tok.apply_chat_template.assert_called_once_with(
            MESSAGES, tokenize=False, add_generation_prompt=True
        )

    def test_tools_forwarded_to_apply_chat_template(self):
        tok = MagicMock()
        tok.apply_chat_template.return_value = "<tool_call>..."
        tools = [{"type": "function", "function": {"name": "get_weather"}}]
        runner = _runner_with_tokenizer(tokenizer=tok)

        prompt, used = runner._apply_chat_template(MESSAGES, tools=tools)

        self.assertTrue(used)
        tok.apply_chat_template.assert_called_once_with(
            MESSAGES, tokenize=False, add_generation_prompt=True, tools=tools
        )

    def test_tool_choice_forwarded_to_apply_chat_template(self):
        tok = MagicMock()
        tok.apply_chat_template.return_value = "<tool_call>..."
        runner = _runner_with_tokenizer(tokenizer=tok)

        prompt, used = runner._apply_chat_template(
            MESSAGES,
            tools=[{"type": "function", "function": {"name": "get_weather"}}],
            tool_choice={"type": "function", "function": {"name": "get_weather"}},
        )

        self.assertTrue(used)
        tok.apply_chat_template.assert_called_once_with(
            MESSAGES,
            tokenize=False,
            add_generation_prompt=True,
            tools=[{"type": "function", "function": {"name": "get_weather"}}],
            tool_choice={"type": "function", "function": {"name": "get_weather"}},
        )

    def test_fallback_to_naive_when_tokenizer_raises(self):
        tok = MagicMock()
        tok.apply_chat_template.side_effect = Exception("no template")
        runner = _runner_with_tokenizer(tokenizer=tok)

        prompt, used = runner._apply_chat_template(MESSAGES)

        self.assertFalse(used)
        self.assertIn("USER:", prompt)
        self.assertIn("ASSISTANT:", prompt)

    def test_fallback_to_naive_when_tokenizer_is_none(self):
        runner = _runner_with_tokenizer()
        runner.tokenizer = None  # tokenizer unavailable

        prompt, used = runner._apply_chat_template(MESSAGES)

        self.assertFalse(used)
        self.assertIn("USER:", prompt)

    # ------------------------------------------------------------------
    # Explicit opt-out
    # ------------------------------------------------------------------

    def test_none_setting_skips_template(self):
        tok = MagicMock()
        runner = _runner_with_tokenizer(chat_template_setting="none", tokenizer=tok)

        prompt, used = runner._apply_chat_template(MESSAGES)

        self.assertFalse(used)
        tok.apply_chat_template.assert_not_called()
        self.assertIn("USER:", prompt)

    def test_false_setting_skips_template(self):
        runner = _runner_with_tokenizer(chat_template_setting="false")
        _, used = runner._apply_chat_template(MESSAGES)
        self.assertFalse(used)

    def test_off_setting_skips_template(self):
        runner = _runner_with_tokenizer(chat_template_setting="OFF")
        _, used = runner._apply_chat_template(MESSAGES)
        self.assertFalse(used)

    # ------------------------------------------------------------------
    # Custom template file
    # ------------------------------------------------------------------

    def test_custom_template_file_is_loaded_and_passed(self):
        tok = MagicMock()
        tok.apply_chat_template.return_value = "<custom>Hello</custom>"
        template_content = "{% for m in messages %}{{ m.content }}{% endfor %}"

        with patch("builtins.open", unittest.mock.mock_open(read_data=template_content)):
            runner = _runner_with_tokenizer(
                chat_template_setting="/some/path.jinja", tokenizer=tok
            )
            prompt, used = runner._apply_chat_template(MESSAGES)

        self.assertTrue(used)
        call_kwargs = tok.apply_chat_template.call_args.kwargs
        self.assertEqual(call_kwargs["chat_template"], template_content)

    def test_custom_template_file_not_found_raises(self):
        runner = _runner_with_tokenizer(chat_template_setting="/nonexistent/template.jinja")

        with self.assertRaises(ValueError, msg="Should raise ValueError for missing file"):
            runner._apply_chat_template(MESSAGES)

    def test_harmony_plain_chat_forces_final_channel_prompt(self):
        tok = MagicMock()
        tok.apply_chat_template.return_value = (
            "<|start|>system<|message|>sys<|end|>"
            "<|start|>user<|message|>hi<|end|><|start|>assistant"
        )
        tok.chat_template = "<|channel|>"
        runner = _runner_with_tokenizer(tokenizer=tok)
        runner.loaded_model_id = "unsloth/gpt-oss-20b"

        prompt, used = runner._apply_chat_template(MESSAGES)

        self.assertTrue(used)
        self.assertTrue(prompt.endswith("<|start|>assistant<|channel|>final<|message|>"))

    def test_harmony_with_tools_does_not_force_final_channel_prompt(self):
        tok = MagicMock()
        tok.apply_chat_template.return_value = "<|start|>assistant"
        tok.chat_template = "<|channel|>"
        runner = _runner_with_tokenizer(tokenizer=tok)
        runner.loaded_model_id = "unsloth/gpt-oss-20b"

        prompt, _ = runner._apply_chat_template(
            MESSAGES,
            tools=[{"type": "function", "function": {"name": "tool_a"}}],
        )

        self.assertEqual(prompt, "<|start|>assistant")

    # ------------------------------------------------------------------
    # _flatten_messages_to_prompt integration
    # ------------------------------------------------------------------

    def test_flatten_returns_three_tuple(self):
        tok = MagicMock()
        tok.apply_chat_template.return_value = "templated"
        runner = _runner_with_tokenizer(tokenizer=tok)

        result = runner._flatten_messages_to_prompt(MESSAGES)

        self.assertEqual(len(result), 3)
        prompt, images, used = result
        self.assertIsInstance(prompt, str)
        self.assertIsInstance(images, list)
        self.assertIsInstance(used, bool)

    def test_flatten_extracts_text_from_list_content(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this:"},
                ],
            }
        ]
        runner = _runner_with_tokenizer(chat_template_setting="none")

        prompt, images, used = runner._flatten_messages_to_prompt(messages)

        self.assertFalse(used)
        self.assertEqual(images, [])
        self.assertIn("Describe this:", prompt)

    def test_flatten_forces_harmony_final_channel_for_plain_chat(self):
        tok = MagicMock()
        tok.chat_template = "<|start|>assistant<|channel|>"
        tok.apply_chat_template.return_value = "...<|start|>assistant"
        runner = _runner_with_tokenizer(tokenizer=tok)
        runner.loaded_model_id = "unsloth/gpt-oss-20b"

        prompt, _, used = runner._flatten_messages_to_prompt(MESSAGES)

        self.assertTrue(used)
        self.assertTrue(prompt.endswith("<|channel|>final<|message|>"))

    def test_flatten_does_not_double_append_harmony_final_channel(self):
        tok = MagicMock()
        tok.chat_template = "<|channel|>"
        tok.apply_chat_template.return_value = "...<|start|>assistant"
        runner = _runner_with_tokenizer(tokenizer=tok)
        runner.loaded_model_id = "unsloth/gpt-oss-20b"

        prompt, _, _ = runner._flatten_messages_to_prompt(MESSAGES)

        self.assertEqual(prompt.count("<|channel|>final<|message|>"), 1)


class TestNaiveFormat(unittest.TestCase):
    """Unit tests for ServerRunner._naive_format."""

    def _naive(self, messages, chat_template_setting="none"):
        runner = _runner_with_tokenizer(chat_template_setting=chat_template_setting)
        return runner._naive_format(messages)

    def test_single_user_message(self):
        prompt = self._naive([{"role": "user", "content": "Hello"}])
        self.assertEqual(prompt, "USER: Hello\nASSISTANT:")

    def test_system_and_user(self):
        prompt = self._naive(MULTI_MESSAGES)
        self.assertIn("SYSTEM: You are helpful.", prompt)
        self.assertIn("USER: What is the weather?", prompt)
        self.assertTrue(prompt.endswith("ASSISTANT:"))

    def test_none_content_treated_as_empty(self):
        prompt = self._naive([{"role": "user", "content": None}])
        self.assertEqual(prompt, "USER: \nASSISTANT:")

    def test_missing_role_defaults_to_user(self):
        prompt = self._naive([{"content": "hi"}])
        self.assertIn("USER:", prompt)

    def test_empty_messages_list(self):
        prompt = self._naive([])
        self.assertEqual(prompt, "ASSISTANT:")


class TestExtractToolCalls(unittest.TestCase):
    """Unit tests for ServerRunner._extract_tool_calls_from_completion."""

    def _parse(self, text):
        runner = _runner_with_tokenizer()
        return runner._extract_tool_calls_from_completion(text)

    def test_no_tool_call_block(self):
        clean, tool_calls = self._parse("hello world")
        self.assertEqual(clean, "hello world")
        self.assertEqual(tool_calls, [])

    def test_single_tool_call_block(self):
        text = (
            "<minimax:tool_call>\n"
            '<invoke name="get_weather">\n'
            '<parameter name="city">"Paris"</parameter>\n'
            '<parameter name="days">3</parameter>\n'
            "</invoke>\n"
            "</minimax:tool_call>"
        )
        clean, tool_calls = self._parse(text)
        self.assertEqual(clean, "")
        self.assertEqual(len(tool_calls), 1)
        self.assertEqual(tool_calls[0]["type"], "function")
        self.assertEqual(tool_calls[0]["function"]["name"], "get_weather")
        self.assertEqual(
            tool_calls[0]["function"]["arguments"],
            '{"city": "Paris", "days": 3}',
        )

    def test_multiple_invokes_are_parsed(self):
        text = (
            "prefix\n"
            "<minimax:tool_call>\n"
            '<invoke name="a"><parameter name="x">1</parameter></invoke>\n'
            '<invoke name="b"><parameter name="y">2</parameter></invoke>\n'
            "</minimax:tool_call>\n"
            "suffix"
        )
        clean, tool_calls = self._parse(text)
        self.assertEqual(clean, "prefix\n\nsuffix")
        self.assertEqual([tc["function"]["name"] for tc in tool_calls], ["a", "b"])

    def test_non_json_parameter_stays_string(self):
        text = (
            "<minimax:tool_call>\n"
            '<invoke name="search"><parameter name="query">foo bar</parameter></invoke>\n'
            "</minimax:tool_call>"
        )
        _, tool_calls = self._parse(text)
        self.assertEqual(tool_calls[0]["function"]["arguments"], '{"query": "foo bar"}')


class TestExtractReasoning(unittest.TestCase):
    def _parse(self, text):
        runner = _runner_with_tokenizer()
        return runner._extract_reasoning_from_completion(text)

    def test_no_think_block(self):
        clean, reasoning = self._parse("hello world")
        self.assertEqual(clean, "hello world")
        self.assertIsNone(reasoning)

    def test_single_think_block_is_split_out(self):
        clean, reasoning = self._parse("<think>plan first</think>final answer")
        self.assertEqual(clean, "final answer")
        self.assertEqual(reasoning, "plan first")

    def test_multiple_think_blocks_are_joined(self):
        clean, reasoning = self._parse("<think>a</think>visible<think>b</think>")
        self.assertEqual(clean, "visible")
        self.assertEqual(reasoning, "a\n\nb")


class TestGenerateChatStructuredOutput(unittest.TestCase):
    def test_generate_chat_extracts_reasoning_and_tool_calls(self):
        runner = _runner_with_tokenizer(tokenizer=MagicMock())
        runner.settings.device = "cpu"
        runner.load_model_if_needed = MagicMock(side_effect=lambda model_id=None: None)
        runner.loaded_model_id = "MiniMaxAI/MiniMax-M2.5"
        runner.model = MagicMock()
        runner.effective_max_seq_len = 1024
        runner._flatten_messages_to_prompt = MagicMock(return_value=("prompt", [], True))

        toks = {
            "input_ids": MagicMock(),
            "attention_mask": MagicMock(),
        }
        toks["input_ids"].to.return_value = toks["input_ids"]
        toks["attention_mask"].to.return_value = toks["attention_mask"]
        toks["input_ids"].shape = (1, 3)
        runner.tokenizer.return_value = toks

        output_ids = MagicMock()
        completion_ids = MagicMock()
        completion_ids.shape = (5,)
        output_ids.__getitem__.return_value = completion_ids
        runner.model.generate.return_value = output_ids
        runner.tokenizer.decode.return_value = (
            "<think>reason privately</think>"
            "visible answer\n"
            "<minimax:tool_call><invoke name=\"search\">"
            "<parameter name=\"query\">\"weather\"</parameter>"
            "</invoke></minimax:tool_call>"
        )

        response = runner.generate_chat(
            messages=[{"role": "user", "content": "hi"}],
            model_id="MiniMaxAI/MiniMax-M2.5",
            max_tokens=32,
            temperature=0.0,
            top_p=1.0,
        )

        self.assertEqual(response["completion_text"], "visible answer")
        self.assertEqual(response["reasoning_content"], "reason privately")
        self.assertEqual(response["finish_reason"], "tool_calls")
        self.assertEqual(response["tool_calls"][0]["function"]["name"], "search")


class TestGenerateChatStreaming(unittest.TestCase):
    def test_streamer_is_configured_to_skip_prompt(self):
        tok = MagicMock()
        ids = MagicMock()
        ids.to.return_value = ids
        ids.shape = (1, 5)
        attn = MagicMock()
        attn.to.return_value = attn
        tok.return_value = {
            "input_ids": ids,
            "attention_mask": attn,
        }

        runner = _runner_with_tokenizer(tokenizer=tok)
        runner.load_model_if_needed = MagicMock(side_effect=lambda model_id=None: None)
        runner._flatten_messages_to_prompt = MagicMock(return_value=("prompt", [], True))
        runner.settings.device = "cpu"

        with patch("transformers.TextIteratorStreamer") as mock_streamer:
            mock_streamer.return_value = MagicMock()
            _, _, thread = runner.generate_chat_streaming(
                messages=[{"role": "user", "content": "hi"}],
                model_id="stub-model",
                max_tokens=8,
                temperature=0.0,
                top_p=1.0,
            )

        thread.join(timeout=1.0)

        mock_streamer.assert_called_once_with(
            tok,
            skip_special_tokens=True,
            skip_prompt=True,
            timeout=None,
        )

    def test_generate_chat_streaming_uses_harmony_streamer_for_plain_gpt_oss_chat(self):
        import torch
        from openai_harmony import HarmonyEncodingName, load_harmony_encoding

        tok = MagicMock()
        input_ids = torch.tensor([[1, 2, 3]])
        attention_mask = torch.ones_like(input_ids)
        tok.return_value = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        runner = _runner_with_tokenizer(tokenizer=tok)
        runner.loaded_model_id = "unsloth/gpt-oss-20b"
        runner.load_model_if_needed = MagicMock(side_effect=lambda model_id=None: None)
        runner._flatten_messages_to_prompt = MagicMock(return_value=("prompt", [], True))
        runner.settings.device = "cpu"
        encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)

        with patch.object(runner, "_get_harmony_encoding", return_value=encoding):
            _, streamer, thread = runner.generate_chat_streaming(
                messages=[{"role": "user", "content": "hi"}],
                model_id="unsloth/gpt-oss-20b",
                max_tokens=8,
                temperature=0.0,
                top_p=1.0,
            )

        thread.join(timeout=1.0)
        self.assertIsInstance(streamer, _HarmonyFinalChannelStreamer)

    def test_generate_chat_streaming_uses_harmony_streamer_for_tool_turns(self):
        import torch
        from openai_harmony import HarmonyEncodingName, load_harmony_encoding

        tok = MagicMock()
        input_ids = torch.tensor([[1, 2, 3]])
        attention_mask = torch.ones_like(input_ids)
        tok.return_value = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        runner = _runner_with_tokenizer(tokenizer=tok)
        runner.loaded_model_id = "unsloth/gpt-oss-20b"
        runner.load_model_if_needed = MagicMock(side_effect=lambda model_id=None: None)
        runner._flatten_messages_to_prompt = MagicMock(return_value=("prompt", [], True))
        runner.settings.device = "cpu"
        encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)

        with patch.object(runner, "_get_harmony_encoding", return_value=encoding):
            _, streamer, thread = runner.generate_chat_streaming(
                messages=[{"role": "user", "content": "hi"}],
                model_id="unsloth/gpt-oss-20b",
                max_tokens=8,
                temperature=0.0,
                top_p=1.0,
                tools=[{"type": "function", "function": {"name": "echo"}}],
            )

        thread.join(timeout=1.0)
        self.assertIsInstance(streamer, _HarmonyFinalChannelStreamer)


class TestHarmonyCompletionParsing(unittest.TestCase):
    def test_parse_harmony_completion_tokens_extracts_final_and_reasoning(self):
        from openai_harmony import HarmonyEncodingName, load_harmony_encoding

        runner = _runner_with_tokenizer()
        encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
        completion_tokens = encoding.encode(
            "<|start|>assistant<|channel|>analysis<|message|>think first<|end|>"
            "<|start|>assistant<|channel|>final<|message|>hello there<|return|>",
            allowed_special="all",
        )

        with patch.object(runner, "_get_harmony_encoding", return_value=encoding):
            completion_text, reasoning_content, tool_calls, finish_reason = (
                runner._parse_harmony_completion_tokens(completion_tokens)
            )

        self.assertEqual(completion_text, "hello there")
        self.assertEqual(reasoning_content, "think first")
        self.assertEqual(tool_calls, [])
        self.assertEqual(finish_reason, "stop")

    def test_parse_harmony_completion_tokens_falls_back_to_reasoning_when_no_final(self):
        from openai_harmony import HarmonyEncodingName, load_harmony_encoding

        runner = _runner_with_tokenizer()
        encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
        completion_tokens = encoding.encode(
            "<|start|>assistant<|channel|>analysis<|message|>draft reply<|return|>",
            allowed_special="all",
        )

        with patch.object(runner, "_get_harmony_encoding", return_value=encoding):
            completion_text, reasoning_content, tool_calls, finish_reason = (
                runner._parse_harmony_completion_tokens(completion_tokens)
            )

        self.assertEqual(completion_text, "draft reply")
        self.assertEqual(reasoning_content, "draft reply")
        self.assertEqual(tool_calls, [])
        self.assertEqual(finish_reason, "stop")

    def test_harmony_streamer_emits_only_final_channel(self):
        from openai_harmony import HarmonyEncodingName, load_harmony_encoding

        encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
        completion_tokens = encoding.encode(
            "<|start|>assistant<|channel|>analysis<|message|>private reasoning<|end|>"
            "<|start|>assistant<|channel|>final<|message|>visible answer<|return|>",
            allowed_special="all",
        )
        streamer = _HarmonyFinalChannelStreamer(
            encoding,
            prompt_token_count=2,
            timeout=1.0,
        )

        streamer.put([123, 456] + completion_tokens[:5])
        streamer.put(completion_tokens[5:])
        streamer.end()

        self.assertEqual("".join(list(streamer)), "visible answer")

    def test_harmony_streamer_waits_without_timeout_error(self):
        import threading
        import time
        from openai_harmony import HarmonyEncodingName, load_harmony_encoding

        encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
        completion_tokens = encoding.encode(
            "<|start|>assistant<|channel|>final<|message|>ok<|return|>",
            allowed_special="all",
        )
        streamer = _HarmonyFinalChannelStreamer(
            encoding,
            prompt_token_count=0,
            timeout=0.01,
        )

        def delayed_feed():
            time.sleep(0.05)
            streamer.put(completion_tokens)
            streamer.end()

        thread = threading.Thread(target=delayed_feed, daemon=True)
        thread.start()
        values = list(streamer)
        thread.join(timeout=1.0)

        self.assertEqual("".join(values), "ok")

    def test_generate_chat_uses_harmony_parser_for_gpt_oss(self):
        import torch
        from openai_harmony import HarmonyEncodingName, load_harmony_encoding

        tok = MagicMock()
        input_ids = torch.tensor([[11, 22]])
        attention_mask = torch.ones_like(input_ids)
        tok.return_value = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        runner = _runner_with_tokenizer(tokenizer=tok)
        runner.settings.device = "cpu"
        runner.loaded_model_id = "unsloth/gpt-oss-20b"
        runner.load_model_if_needed = MagicMock(side_effect=lambda model_id=None: None)
        runner._flatten_messages_to_prompt = MagicMock(return_value=("prompt", [], True))

        encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
        completion_tokens = encoding.encode(
            "<|start|>assistant<|channel|>analysis<|message|>plan<|end|>"
            "<|start|>assistant<|channel|>final<|message|>done<|return|>",
            allowed_special="all",
        )
        runner.model.generate.return_value = torch.tensor([[11, 22] + completion_tokens])

        with patch.object(runner, "_get_harmony_encoding", return_value=encoding):
            response = runner.generate_chat(
                messages=[{"role": "user", "content": "hi"}],
                model_id="unsloth/gpt-oss-20b",
                max_tokens=32,
                temperature=0.0,
                top_p=1.0,
            )

        tok.decode.assert_called_once()
        self.assertEqual(response["completion_text"], "done")
        self.assertEqual(response["reasoning_content"], "plan")
