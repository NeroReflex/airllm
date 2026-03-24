"""
Tests for single-file model support in split_and_save_layers.

Covers the case where a model ships as a single model.safetensors or
pytorch_model.bin file (no shard index), which is common for models <= ~7B.
"""
import json
import os
import tempfile
import shutil
import unittest
from unittest.mock import patch

import torch
from safetensors.torch import load_file, save_file


class TestSingleFileModelSplit(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="airllm_single_file_test_")

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _make_fake_model_state(self):
        """Minimal Llama-style state dict with 1 decoder layer."""
        hidden = 64
        inter = 128
        vocab = 100
        heads = 4
        return {
            "model.embed_tokens.weight":              torch.randn(vocab, hidden),
            "model.layers.0.input_layernorm.weight":  torch.randn(hidden),
            "model.layers.0.self_attn.q_proj.weight": torch.randn(hidden, hidden),
            "model.layers.0.self_attn.k_proj.weight": torch.randn(hidden // heads, hidden),
            "model.layers.0.self_attn.v_proj.weight": torch.randn(hidden // heads, hidden),
            "model.layers.0.self_attn.o_proj.weight": torch.randn(hidden, hidden),
            "model.layers.0.mlp.gate_proj.weight":    torch.randn(inter, hidden),
            "model.layers.0.mlp.up_proj.weight":      torch.randn(inter, hidden),
            "model.layers.0.mlp.down_proj.weight":    torch.randn(hidden, inter),
            "model.layers.0.post_attention_layernorm.weight": torch.randn(hidden),
            "model.norm.weight":                      torch.randn(hidden),
            "lm_head.weight":                         torch.randn(vocab, hidden),
        }

    def _make_fake_model_state_without_lm_head(self):
        state = self._make_fake_model_state()
        del state["lm_head.weight"]
        return state

    def _touch_done_marker(self, split_path, layer_prefix):
        shard_path = os.path.join(split_path, f"{layer_prefix}.safetensors")
        save_file({f"{layer_prefix}.weight": torch.randn(1)}, shard_path)
        open(shard_path + ".done", "a", encoding="utf-8").close()

    # ------------------------------------------------------------------
    # single model.safetensors (no index)
    # ------------------------------------------------------------------
    def test_split_single_safetensors_file(self):
        state = self._make_fake_model_state()
        save_file(state, os.path.join(self.tmpdir, "model.safetensors"))

        from airllm.utils import split_and_save_layers
        split_path = split_and_save_layers(self.tmpdir)

        self.assertTrue(os.path.isdir(split_path))
        expected_files = [
            "model.embed_tokens.safetensors",
            "model.layers.0.safetensors",
            "model.norm.safetensors",
            "lm_head.safetensors",
        ]
        for fname in expected_files:
            self.assertTrue(
                os.path.exists(os.path.join(split_path, fname)),
                f"Expected shard file missing: {fname}",
            )

    # ------------------------------------------------------------------
    # single pytorch_model.bin (no index)
    # ------------------------------------------------------------------
    def test_split_single_pytorch_bin_file(self):
        state = self._make_fake_model_state()
        torch.save(state, os.path.join(self.tmpdir, "pytorch_model.bin"))

        from airllm.utils import split_and_save_layers
        split_path = split_and_save_layers(self.tmpdir)

        self.assertTrue(os.path.isdir(split_path))
        self.assertTrue(
            os.path.exists(os.path.join(split_path, "model.embed_tokens.safetensors"))
        )

    # ------------------------------------------------------------------
    # single model.safetensors without lm_head (tied output embedding)
    # ------------------------------------------------------------------
    def test_split_single_safetensors_without_lm_head(self):
        state = self._make_fake_model_state_without_lm_head()
        save_file(state, os.path.join(self.tmpdir, "model.safetensors"))

        from airllm.utils import split_and_save_layers
        split_path = split_and_save_layers(self.tmpdir)

        self.assertTrue(os.path.isdir(split_path))
        self.assertTrue(
            os.path.exists(os.path.join(split_path, "lm_head.safetensors")),
            "Expected lm_head shard to be synthesized for tied-output models",
        )

    # ------------------------------------------------------------------
    # sharded model.safetensors.index.json still works (regression)
    # ------------------------------------------------------------------
    def test_split_sharded_safetensors_still_works(self):
        state = self._make_fake_model_state()
        shard_file = "model-00001-of-00001.safetensors"
        save_file(state, os.path.join(self.tmpdir, shard_file))
        index = {"metadata": {}, "weight_map": {k: shard_file for k in state}}
        with open(os.path.join(self.tmpdir, "model.safetensors.index.json"), "w") as f:
            json.dump(index, f)

        from airllm.utils import split_and_save_layers
        split_path = split_and_save_layers(self.tmpdir)

        self.assertTrue(
            os.path.exists(os.path.join(split_path, "model.embed_tokens.safetensors"))
        )

    # ------------------------------------------------------------------
    # no weights at all → FileNotFoundError
    # ------------------------------------------------------------------
    def test_raises_when_no_weights(self):
        from airllm.utils import split_and_save_layers
        with self.assertRaises(FileNotFoundError):
            split_and_save_layers(self.tmpdir)

    # ------------------------------------------------------------------
    # Non-standard shard zero-padding (alexey fix / issue #214)
    # DeepSeek and some other models use 6-digit shard numbers
    # (model-000001-of-000002) instead of the common 5-digit pattern.
    # The old hardcoded f-string would build the wrong filename and raise
    # FileNotFoundError.  With the fix, actual filenames from the index
    # are used directly.
    # ------------------------------------------------------------------
    def test_split_sharded_nonstandard_padding(self):
        state1 = {
            "model.embed_tokens.weight": torch.randn(100, 64),
            "model.layers.0.input_layernorm.weight": torch.randn(64),
        }
        state2 = {
            "model.norm.weight": torch.randn(64),
            "lm_head.weight": torch.randn(100, 64),
        }
        # 6-digit zero-padded shard filenames (non-standard)
        shard1 = "model-000001-of-000002.safetensors"
        shard2 = "model-000002-of-000002.safetensors"
        save_file(state1, os.path.join(self.tmpdir, shard1))
        save_file(state2, os.path.join(self.tmpdir, shard2))
        weight_map = {k: shard1 for k in state1}
        weight_map.update({k: shard2 for k in state2})
        index = {"metadata": {}, "weight_map": weight_map}
        with open(os.path.join(self.tmpdir, "model.safetensors.index.json"), "w") as f:
            json.dump(index, f)

        from airllm.utils import split_and_save_layers
        # Before the fix this would raise FileNotFoundError trying to open
        # model-00001-of-00002.safetensors (5-digit pattern) instead of the
        # actual 6-digit filename.
        split_path = split_and_save_layers(self.tmpdir)

        self.assertTrue(os.path.isdir(split_path))
        for fname in ("model.embed_tokens.safetensors", "model.norm.safetensors", "lm_head.safetensors"):
            self.assertTrue(
                os.path.exists(os.path.join(split_path, fname)),
                f"Expected output shard missing: {fname}",
            )

    # ------------------------------------------------------------------
    # interrupted split resumes from missing layer instead of replaying
    # earlier shards
    # ------------------------------------------------------------------
    def test_resume_partial_split_skips_completed_layers(self):
        shard1_state = {
            "model.embed_tokens.weight": torch.randn(100, 64),
            "model.layers.0.input_layernorm.weight": torch.randn(64),
        }
        shard2_state = {
            "model.layers.1.input_layernorm.weight": torch.randn(64),
            "model.norm.weight": torch.randn(64),
            "lm_head.weight": torch.randn(100, 64),
        }
        shard1 = "model-00001-of-00002.safetensors"
        shard2 = "model-00002-of-00002.safetensors"
        save_file(shard1_state, os.path.join(self.tmpdir, shard1))
        save_file(shard2_state, os.path.join(self.tmpdir, shard2))

        weight_map = {k: shard1 for k in shard1_state}
        weight_map.update({k: shard2 for k in shard2_state})
        index = {"metadata": {}, "weight_map": weight_map}
        with open(os.path.join(self.tmpdir, "model.safetensors.index.json"), "w") as f:
            json.dump(index, f)

        split_path = os.path.join(self.tmpdir, "splitted_model")
        os.makedirs(split_path, exist_ok=True)
        self._touch_done_marker(split_path, "model.embed_tokens")
        self._touch_done_marker(split_path, "model.layers.0")

        loaded_paths = []

        def recording_load_file(path, device="cpu"):
            loaded_paths.append(os.path.basename(str(path)))
            return load_file(path, device=device)

        with patch("airllm.utils.load_file", side_effect=recording_load_file):
            from airllm.utils import split_and_save_layers
            split_and_save_layers(self.tmpdir)

        self.assertEqual(loaded_paths, [shard2])
        for fname in (
            "model.layers.1.safetensors",
            "model.norm.safetensors",
            "lm_head.safetensors",
        ):
            self.assertTrue(
                os.path.exists(os.path.join(split_path, fname)),
                f"Expected resumed split output missing: {fname}",
            )

    # ------------------------------------------------------------------
    # check_space accounts for existing splits when saving_path is None
    # (muhammad fix)
    # Previously the function only counted pre-existing split files when
    # layer_shards_saving_path was explicitly given.  When it is None
    # (saving into the checkpoint dir), existing splits were ignored and
    # the free-space check would raise NotEnoughSpaceException even when
    # all shards had already been split.
    # ------------------------------------------------------------------
    def test_check_space_counts_existing_splits_when_no_saving_path(self):
        import shutil
        from unittest.mock import patch
        from pathlib import Path
        from airllm.utils import check_space, NotEnoughSpaceException

        checkpoint_path = Path(self.tmpdir)
        splitted_dir = checkpoint_path / 'splitted_model'
        splitted_dir.mkdir()

        # Write a "shard" file (1 MB) in the checkpoint dir.
        # Note: check_space glob includes the splitted_model/ directory entry
        # itself in total_shard_files_size_bytes (os.path.getsize on a dir
        # returns the directory metadata size, typically 4096 on Linux).
        # So total = 1MB + ~4KB.  The existing split file must be larger.
        shard = checkpoint_path / 'model.safetensors'
        shard.write_bytes(b'\x00' * 1_000_000)

        # Write a pre-existing split file (2 MB) — more than the total shard
        # size, so the space check should pass once we count it.
        existing_split = splitted_dir / 'model.embed_tokens.safetensors'
        existing_split.write_bytes(b'\x00' * 2_000_000)

        # Mock disk_usage to report only 1 byte free — without the fix this
        # would cause NotEnoughSpaceException even though the existing split
        # already covers the required space.
        fake_usage = shutil.disk_usage(self.tmpdir)  # get real namedtuple type
        fake_usage = fake_usage._replace(free=1)     # pretend almost no free space

        with patch('airllm.utils.shutil.disk_usage', return_value=fake_usage):
            # Should NOT raise: the 1 MB existing split covers the 1 MB needed
            try:
                check_space(checkpoint_path, layer_shards_saving_path=None)
            except NotEnoughSpaceException:
                self.fail(
                    "check_space raised NotEnoughSpaceException even though "
                    "pre-existing splits already cover the required space"
                )


if __name__ == "__main__":
    unittest.main()
