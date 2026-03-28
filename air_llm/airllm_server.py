import importlib.abc
import importlib.machinery
import os
import sys


def _maybe_disable_triton_for_frozen_runtime() -> None:
    """Prevent Triton import crashes in frozen one-file runtimes.

    In one-file builds, Triton JIT may fail at import time because it cannot
    inspect Python source files from the embedded archive. When that happens,
    `torch.utils._triton.has_triton_package()` raises a non-ImportError and
    aborts startup. Blocking `triton` imports in this mode makes torch treat
    Triton as unavailable and continue normally.
    """

    # Allow opting out for debugging.
    disable = os.environ.get("AIRLLM_DISABLE_TRITON_IN_FROZEN", "1")
    if disable.strip().lower() in {"0", "false", "no", "off"}:
        return
    if not getattr(sys, "frozen", False):
        return

    # Also disable inductor-side Triton probing.
    os.environ.setdefault("TORCHINDUCTOR_TRITON_DISABLE_DEVICE_DETECTION", "1")

    class _BlockedTritonLoader(importlib.abc.Loader):
        def create_module(self, spec):
            return None

        def exec_module(self, module):
            raise ImportError("Triton disabled in frozen AirLLM runtime")

    class _BlockedTritonFinder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname == "triton" or fullname.startswith("triton."):
                return importlib.machinery.ModuleSpec(fullname, _BlockedTritonLoader())
            return None

    sys.meta_path.insert(0, _BlockedTritonFinder())


_maybe_disable_triton_for_frozen_runtime()

from airllm.server.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
