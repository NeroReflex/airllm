from __future__ import annotations

import importlib
import warnings
from typing import Any


_SENTENCEPIECE_SWIG_WARNING_PATTERNS = (
    r"builtin type SwigPyPacked has no __module__ attribute",
    r"builtin type SwigPyObject has no __module__ attribute",
    r"builtin type swigvarlink has no __module__ attribute",
)


def import_sentencepiece_module() -> Any:
    """Import sentencepiece while suppressing known upstream SWIG warnings.

    Python 3.14 emits DeprecationWarning for legacy SWIG-generated types inside
    sentencepiece. Those warnings are outside this project's control and do not
    affect runtime correctness, so we contain them at the import boundary.
    """
    with warnings.catch_warnings():
        for pattern in _SENTENCEPIECE_SWIG_WARNING_PATTERNS:
            warnings.filterwarnings(
                "ignore",
                message=pattern,
                category=DeprecationWarning,
            )
        return importlib.import_module("sentencepiece")


def import_sentencepiece_processor() -> Any:
    """Return sentencepiece.SentencePieceProcessor with warning filtering."""
    return import_sentencepiece_module().SentencePieceProcessor