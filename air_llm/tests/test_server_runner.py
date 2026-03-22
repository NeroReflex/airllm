import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from ..airllm.server.config import Settings
from ..airllm.server.runner import ServerRunner


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
