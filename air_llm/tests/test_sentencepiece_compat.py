import warnings
import unittest
from unittest.mock import patch

from ..airllm import _sentencepiece_compat as sp_compat


class _FakeSentencePieceModule:
    SentencePieceProcessor = object()


class TestSentencePieceCompat(unittest.TestCase):
    def test_suppresses_known_swig_warnings(self):
        def fake_import(name):
            warnings.warn(
                "builtin type swigvarlink has no __module__ attribute",
                DeprecationWarning,
                stacklevel=1,
            )
            return _FakeSentencePieceModule()

        with patch.object(sp_compat.importlib, "import_module", side_effect=fake_import):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                module = sp_compat.import_sentencepiece_module()

        self.assertIsInstance(module, _FakeSentencePieceModule)
        self.assertEqual(caught, [])

    def test_does_not_suppress_unrelated_deprecations(self):
        def fake_import(name):
            warnings.warn("different deprecation", DeprecationWarning, stacklevel=1)
            return _FakeSentencePieceModule()

        with patch.object(sp_compat.importlib, "import_module", side_effect=fake_import):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                module = sp_compat.import_sentencepiece_module()

        self.assertIsInstance(module, _FakeSentencePieceModule)
        self.assertEqual(len(caught), 1)
        self.assertEqual(str(caught[0].message), "different deprecation")

    def test_import_sentencepiece_processor_returns_processor_symbol(self):
        with patch.object(
            sp_compat,
            "import_sentencepiece_module",
            return_value=_FakeSentencePieceModule(),
        ):
            processor = sp_compat.import_sentencepiece_processor()

        self.assertIs(processor, _FakeSentencePieceModule.SentencePieceProcessor)