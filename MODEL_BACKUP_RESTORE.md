# Model Backup and Restore Runbook (AirLLM)

This document explains where models are stored on this machine and how to back them up or restore them from another PC with minimal extra disk writes.

## Where models are stored

Default Hugging Face cache root:

- $HOME/.cache/huggingface

Model files used by AirLLM are under:

- $HOME/.cache/huggingface/hub

Model directory naming rule:

- model id org/name -> models--org--name
- Example: unsloth/phi-4 -> models--unsloth--phi-4

AirLLM split-layer cache location (important):

- $HOME/.cache/huggingface/hub/models--<org>--<name>/snapshots/<snapshot_hash>/splitted_model

On this machine, detected examples:

- $HOME/.cache/huggingface/hub/models--unsloth--phi-4/snapshots/c6220bde10fff762dbd72c3331894aa4cade249d/splitted_model
- $HOME/.cache/huggingface/hub/models--Jackrong--Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled/snapshots/99902bc300f37f8722e6af2c69e9c13e19d39c61/splitted_model
- $HOME/.cache/huggingface/hub/models--unsloth--Qwen3-Coder-Next-FP8-Dynamic/snapshots/35760007a480d66badd50f56a5b00501ea431e86/splitted_model

---

## Goal: minimize disk writes

To avoid heavy write bursts during inference startup:

1. Restore the full model cache directory before first run.
2. Include the splitted_model folder so AirLLM does not re-split weights.
3. Keep the same cache path on destination if possible.

If splitted_model is missing or partial, AirLLM will re-create it and write many files.

---

## Backup from source PC

### Option A (recommended): rsync model-specific directory

Use this for one model only.

```bash
# Source PC
MODEL_DIR="$HOME/.cache/huggingface/hub/models--unsloth--phi-4"
DEST="user@target-pc:/home/user/backups/hf-models/"

rsync -aH --info=progress2 "$MODEL_DIR" "$DEST"
```

Notes:
- -a preserves symlinks/permissions/timestamps.
- -H preserves hard links if present.

### Option B: tar archive for transfer

```bash
# Source PC
cd $HOME/.cache/huggingface/hub

tar --numeric-owner -cpf /tmp/models--unsloth--phi-4.tar models--unsloth--phi-4
zstd -19 -T0 /tmp/models--unsloth--phi-4.tar -o /tmp/models--unsloth--phi-4.tar.zst
```

Copy /tmp/models--unsloth--phi-4.tar.zst to target machine.

---

## Restore on destination PC

### Option A: rsync pull from source

```bash
# Destination PC
mkdir -p /home/user/.cache/huggingface/hub

rsync -aH --info=progress2 \
  user@source-pc:$HOME/.cache/huggingface/hub/models--unsloth--phi-4 \
  /home/user/.cache/huggingface/hub/
```

### Option B: restore from tar.zst

```bash
# Destination PC
mkdir -p /home/user/.cache/huggingface/hub
cd /home/user/.cache/huggingface/hub

zstd -d -T0 /path/to/models--unsloth--phi-4.tar.zst -o /tmp/models--unsloth--phi-4.tar
tar -xpf /tmp/models--unsloth--phi-4.tar
```

---

## Verify restore is complete (prevents re-download/re-split)

```bash
python - << 'PY'
from pathlib import Path
p = Path('/home/user/.cache/huggingface/hub/models--unsloth--phi-4/snapshots')
print('snapshots exists:', p.exists())
if p.exists():
    for s in sorted(p.iterdir()):
        print(s.name, 'splitted_model:', (s/'splitted_model').exists())
PY
```

If splitted_model is True for your snapshot, startup writes are minimized.

---

## Restore multiple models quickly

```bash
# Destination PC
mkdir -p /home/user/.cache/huggingface/hub

for m in \
  models--unsloth--phi-4 \
  models--Jackrong--Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled \
  models--unsloth--Qwen3-Coder-Next-FP8-Dynamic

do
  rsync -aH --info=progress2 "user@source-pc:$HOME/.cache/huggingface/hub/$m" \
    /home/user/.cache/huggingface/hub/
done
```

---

## Environment variables (optional)

If destination path differs, set cache root explicitly:

```bash
export HF_HOME=/path/to/huggingface-cache
```

Then place model folders under:

- $HF_HOME/hub/models--org--name

---

## Low-RAM and low-write tips

- Keep max_seq_len conservative for consumer hardware.
- Use layers_per_batch=1 when memory is tight.
- Reuse one cache location and avoid moving model folders between runs.
- Avoid partial copies: interrupted restore often triggers re-download or re-split writes.

---

## Quick smoke command after restore

```bash
cd $HOME/projects/neroreflex/airllm
source .venv/bin/activate
export PYTHONPATH=air_llm

python - << 'PY'
from airllm import AutoModel

model = AutoModel.from_pretrained(
    'unsloth/phi-4',
    device='cuda:0',
    max_seq_len=64,
    prefetching=False,
    layers_per_batch=1,
)

toks = model.tokenizer(['Say hello in one short sentence.'], return_tensors='pt', truncation=True, max_length=32)
out = model.generate(toks['input_ids'].to('cuda:0'), max_new_tokens=8, use_cache=False)
print(model.tokenizer.decode(out[0], skip_special_tokens=True))
PY
```
