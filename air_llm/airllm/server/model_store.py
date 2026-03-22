import shutil
from pathlib import Path

from huggingface_hub import snapshot_download


def _repo_dir_name(model_id: str) -> str:
    return "models--" + model_id.replace("/", "--")


class ModelStore:
    def __init__(self, cache_dir: str, hf_token: str = ""):
        self.cache_dir = Path(cache_dir)
        self.hf_token = hf_token or None

    def pull(self, model_id: str) -> str:
        return snapshot_download(
            repo_id=model_id,
            token=self.hf_token,
            resume_download=True,
        )

    def list_local_models(self) -> list[str]:
        if not self.cache_dir.exists():
            return []

        models: list[str] = []
        for child in self.cache_dir.iterdir():
            if not child.is_dir() or not child.name.startswith("models--"):
                continue
            # models--org--name -> org/name
            model_id = child.name[len("models--") :].replace("--", "/")
            snapshots = child / "snapshots"
            if snapshots.exists() and any(snapshots.iterdir()):
                models.append(model_id)
        return sorted(models)

    def remove(self, model_id: str) -> bool:
        target = self.cache_dir / _repo_dir_name(model_id)
        if not target.exists():
            return False
        shutil.rmtree(target)
        return True

    def exists(self, model_id: str) -> bool:
        target = self.cache_dir / _repo_dir_name(model_id)
        return target.exists()
