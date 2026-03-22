![airllm_logo](https://github.com/lyogavin/airllm/blob/main/assets/airllm_logo_sm.png?v=3&raw=true)

[**Quickstart**](#quickstart) | 
[**Configurations**](#configurations) | 
[**OpenAI-Compatible Server**](#openai-compatible-server) | 
[**Docker**](#docker-openai-api--open-webui) | 
[**MacOS**](#macos) | 
[**Example notebooks**](#example-python-notebook) | 
[**FAQ**](#faq)

**AirLLM** optimizes inference memory usage, allowing 70B large language models to run inference on a single 4GB GPU card. No quantization, distillation, pruning or other model compression techniques that would result in degraded model performance are needed.

<a href="https://github.com/lyogavin/Anima/stargazers">![GitHub Repo stars](https://img.shields.io/github/stars/lyogavin/Anima?style=social)</a>
[![Downloads](https://static.pepy.tech/personalized-badge/airllm?period=total&units=international_system&left_color=grey&right_color=blue&left_text=downloads)](https://pepy.tech/project/airllm)

[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](https://github.com/LianjiaTech/BELLE/blob/main/LICENSE)
[![Generic badge](https://img.shields.io/badge/wechat-Anima-brightgreen?logo=wechat)](https://static.aicompose.cn/static/wecom_barcode.png?t=1671918938)
[![Discord](https://img.shields.io/discord/1175437549783760896?logo=discord&color=7289da
)](https://discord.gg/2xffU5sn)
[![PyPI - AirLLM](https://img.shields.io/pypi/format/airllm?logo=pypi&color=3571a3)
](https://pypi.org/project/airllm/)
[![Website](https://img.shields.io/website?up_message=blog&url=https%3A%2F%2Fmedium.com%2F%40lyo.gavin&logo=medium&color=black)](https://medium.com/@lyo.gavin)
[![Support me on Patreon](https://img.shields.io/endpoint.svg?url=https%3A%2F%2Fshieldsio-patreon.vercel.app%2Fapi%3Fusername%3Dgavinli%26type%3Dpatrons&style=flat)](https://patreon.com/gavinli)
[![GitHub Sponsors](https://img.shields.io/github/sponsors/lyogavin?logo=GitHub&color=lightgray)](https://github.com/sponsors/lyogavin)


## Updates

[2024/04/20] AirLLM supports Llama3 natively already. Run Llama3 70B on 4GB single GPU.

[2023/12/25] v2.8.2: Support MacOS running 70B large language models.

[2023/12/20] v2.7: Support AirLLMMixtral. 

[2023/12/20] v2.6: Added AutoModel, automatically detect model type, no need to provide model class to initialize model.

[2023/12/18] v2.5: added prefetching to overlap the model loading and compute. 10% speed improvement.

[2023/12/03] added support of **ChatGLM**, **QWen**, **Baichuan**, **Mistral**, **InternLM**!

[2023/12/02] added support for safetensors. Now support all top 10 models in open llm leaderboard.

[2023/12/01] airllm 2.0. Support compressions: **3x run time speed up!**

[2023/11/20] airllm Initial verion!

## Table of Contents

* [Quick start](#quickstart)
* [Model Compression](#model-compression---3x-inference-speed-up)
* [Configurations](#configurations)
* [OpenAI-Compatible Server](#openai-compatible-server)
* [Docker (OpenAI API + Open WebUI)](#docker-openai-api--open-webui)
* [Run on MacOS](#macos)
* [Example notebooks](#example-python-notebook)
* [Supported Models](#supported-models)
* [Acknowledgement](#acknowledgement)
* [FAQ](#faq)

## Quickstart

### 1. Install package

First, install the airllm pip package.

```bash
pip install airllm
```

### 2. Inference

Then, initialize AirLLMLlama2, pass in the huggingface repo ID of the model being used, or the local path, and inference can be performed similar to a regular transformer model.

(*You can also specify the path to save the splitted layered model through **layer_shards_saving_path** when init AirLLMLlama2.*

```python
from airllm import AutoModel

MAX_LENGTH = 128
# could use hugging face model repo id:
model = AutoModel.from_pretrained("garage-bAInd/Platypus2-70B-instruct")

# or use model's local path...
#model = AutoModel.from_pretrained("/home/ubuntu/.cache/huggingface/hub/models--garage-bAInd--Platypus2-70B-instruct/snapshots/b585e74bcaae02e52665d9ac6d23f4d0dbc81a0f")

input_text = [
        'What is the capital of United States?',
        #'I like',
    ]

input_tokens = model.tokenizer(input_text,
    return_tensors="pt", 
    return_attention_mask=False, 
    truncation=True, 
    max_length=MAX_LENGTH, 
    padding=False)
           
generation_output = model.generate(
    input_tokens['input_ids'].cuda(), 
    max_new_tokens=20,
    use_cache=True,
    return_dict_in_generate=True)

output = model.tokenizer.decode(generation_output.sequences[0])

print(output)

```
 
 
Note: During inference, the original model will first be decomposed and saved layer-wise. Please ensure there is sufficient disk space in the huggingface cache directory.


## OpenAI-Compatible Server

AirLLM now includes a built-in OpenAI-compatible server and Ollama-like local model utilities.

### Install runtime dependencies

```bash
pip install airllm
```

### CLI commands

```bash
# Serve API (OpenAI-compatible)
airllm serve --model meta-llama/Llama-3.1-8B-Instruct --host 0.0.0.0 --port 8000

# Optional auth key (requires Bearer token in requests)
airllm serve --model meta-llama/Llama-3.1-8B-Instruct --api-key my-secret-key

# Pull model into local Hugging Face cache
airllm pull meta-llama/Llama-3.1-8B-Instruct

# List local models
airllm models

# Remove model from local cache
airllm rm meta-llama/Llama-3.1-8B-Instruct
```

### OpenAI-style endpoints

When running `airllm serve`, the following endpoints are available:

* `GET /healthz`
* `GET /v1/models`
* `POST /v1/chat/completions` (text + image URLs/data URLs)
* `POST /v1/completions`
* `POST /v1/audio/speech` (TTS models only)
* `POST /v1/audio/transcriptions` (currently returns `501 Not Implemented`)

### Ollama-like utility endpoints

* `GET /api/tags`
* `POST /api/pull`
* `DELETE /api/delete`

### Example request (chat)

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "messages": [
            {"role": "user", "content": "Explain AirLLM in one paragraph."}
        ]
    }'
```

### Example request (speech)

```bash
curl http://127.0.0.1:8000/v1/audio/speech \
    -H "Content-Type: application/json" \
    -d '{"model":"microsoft/speecht5_tts","input":"Hello from AirLLM."}' \
    --output out.wav
```

### Python HTTP examples

Query a served text model:

```bash
python air_llm/examples/query_served_model_text.py \
    --base-url http://127.0.0.1:8000 \
    --model garage-bAInd/Platypus2-7B \
    --prompt "What is the capital of France?"
```

Exercise the transcription endpoint with an audio file:

```bash
python air_llm/examples/test_served_model_transcription.py \
    --base-url http://127.0.0.1:8000 \
    --model openai/whisper-small
```

Note: the transcription endpoint currently returns `501 Not Implemented` until a speech-to-text backend is added. The example is useful as an HTTP smoke test for the served API path.

### Native standalone executable

Build a full standalone native executable (Nuitka onefile build):

```bash
make native-standalone
```

Notes:

* Full standalone build is pinned to Python 3.13 for stability.
* Output binary path: `air_llm/build/native/airllm`.

Install or remove it system-wide:

```bash
# install to /usr/local/bin/airllm (or set PREFIX=...)
sudo make install-native

# remove /usr/local/bin/airllm
sudo make uninstall-native
```

### Release artifacts via Git tags

Pushing a tag like `v2.12.1` triggers `.github/workflows/release-native.yml`, which:

* builds the standalone Linux binary with Python 3.13,
* packages `dist/airllm-<tag>-linux-x86_64.tar.gz`,
* generates a SHA256 checksum file,
* publishes both files to the GitHub Release.

## Docker (OpenAI API + Open WebUI)

### Build and run the AirLLM OpenAI-compatible API container

```bash
docker build -t airllm-openai:local .

docker run --rm -p 8000:8000 \
    -e AIRLLM_MODEL=garage-bAInd/Platypus2-7B \
    -e AIRLLM_DEVICE=cpu \
    -e AIRLLM_MAX_SEQ_LEN=4096 \
    -e AIRLLM_MAX_NEW_TOKENS=256 \
    -e AIRLLM_TEMPERATURE=0.2 \
    -e AIRLLM_TOP_P=0.95 \
    -e AIRLLM_API_KEY=changeme \
    -e AIRLLM_ENFORCE_AUTH=true \
    airllm-openai:local
```

### Run AirLLM API + Open WebUI together (docker compose)

```bash
AIRLLM_MODEL=garage-bAInd/Platypus2-7B \
AIRLLM_MAX_NEW_TOKENS=256 \
AIRLLM_TEMPERATURE=0.2 \
AIRLLM_TOP_P=0.95 \
AIRLLM_API_KEY=changeme \
docker compose -f docker-compose.openwebui.yml up -d --build
```

`AIRLLM_MAX_SEQ_LEN` is optional. If unset or empty, AirLLM infers the largest supported context window from the model config.

### Server environment variables

The server reads these environment variables at startup:

| Variable | Default | Description |
| --- | --- | --- |
| `AIRLLM_MODEL` | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | Default model id to load when no model is provided in request. |
| `AIRLLM_DEVICE` | `cuda:0` (image default) / `cpu` (compose default) | Runtime device, for example `cuda:0`, `cpu`, `privateuseone:0`, or `xpu:0` where supported. |
| `AIRLLM_HOST` | `0.0.0.0` | Server bind host. |
| `AIRLLM_PORT` | `8000` | Server bind port (inside container). |
| `AIRLLM_MAX_SEQ_LEN` | unset (auto) | Max prompt context length used by server tokenization and model internals. If empty/unset, it is inferred from model config. |
| `AIRLLM_MAX_NEW_TOKENS` | `256` | Default generation length when `max_tokens` is not provided by request. |
| `AIRLLM_TEMPERATURE` | `0.2` | Default sampling temperature. |
| `AIRLLM_TOP_P` | `0.95` | Default nucleus sampling value. |
| `AIRLLM_CHAT_TEMPLATE` | empty (auto) | Jinja2 chat-template control. Empty/`auto` = use the model's built-in tokenizer template (equivalent to vLLM `--chat-template` / llama.cpp `--jinja`). Set to a `.jinja` file path to load a custom template. Set to `none` to disable templating and fall back to legacy `ROLE: content` formatting. |
| `AIRLLM_PREFETCHING` | `true` | Enables overlapped layer prefetching for supported backends. |
| `AIRLLM_LAYERS_PER_BATCH` | `auto` | Number of layers loaded to GPU simultaneously (`auto` or integer). |
| `AIRLLM_LAZY_LOAD_MODEL` | `true` | If `true`, loads model on first request; if `false`, loads on server startup. |
| `AIRLLM_API_KEY` | empty | Bearer token expected from clients when auth is enforced. |
| `AIRLLM_ENFORCE_AUTH` | `false` | Enables/disables auth check for incoming requests. |
| `HF_TOKEN` | empty | Hugging Face token for gated model downloads. |
| `AIRLLM_CACHE_DIR` | `~/.cache/huggingface/hub` | Cache directory used for model files and transformed shards. |

Then open Open WebUI at `http://localhost:3000`.

The compose file sets Open WebUI to call AirLLM through:

* `OPENAI_API_BASE_URL=http://airllm-api:8000/v1`
* `OPENAI_API_KEY=${AIRLLM_API_KEY}`

Optional host port overrides:

* `AIRLLM_API_PORT` (default `8000`)
* `OPEN_WEBUI_PORT` (default `3000`)

### Publish container image to GHCR

Pushing a tag like `v2.12.1` triggers `.github/workflows/publish-ghcr.yml` which builds and publishes:

* `ghcr.io/<your-org-or-user>/airllm-openai:v2.12.1`
* `ghcr.io/<your-org-or-user>/airllm-openai:latest`

You can also trigger it manually with GitHub Actions `workflow_dispatch`.
 

## Model Compression - 3x Inference Speed Up!

We just added model compression based on block-wise quantization-based model compression. Which can further **speed up the inference speed** for up to **3x** , with **almost ignorable accuracy loss!** (see more performance evaluation and why we use block-wise quantization in [this paper](https://arxiv.org/abs/2212.09720))

![speed_improvement](https://github.com/lyogavin/Anima/blob/main/assets/airllm2_time_improvement.png?v=2&raw=true)

#### How to enable model compression speed up:

* Step 1. make sure you have [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) installed by `pip install -U bitsandbytes `
* Step 2. make sure airllm verion later than 2.0.0: `pip install -U airllm` 
* Step 3. when initialize the model, passing the argument compression ('4bit' or '8bit'):

```python
model = AutoModel.from_pretrained("garage-bAInd/Platypus2-70B-instruct",
                     compression='4bit' # specify '8bit' for 8-bit block-wise quantization 
                    )
```

#### What are the differences between model compression and quantization?

Quantization normally needs to quantize both weights and activations to really speed things up. Which makes it harder to maintain accuracy and avoid the impact of outliers in all kinds of inputs.

While in our case the bottleneck is mainly at the disk loading, we only need to make the model loading size smaller. So, we get to only quantize the weights' part, which is easier to ensure the accuracy.

## Configurations
 
When initialize the model, we support the following configurations:

* **compression**: supported options: 4bit, 8bit for 4-bit or 8-bit block-wise quantization, or by default None for no compression
* **profiling_mode**: supported options: True to output time consumptions or by default False
* **layer_shards_saving_path**: optionally another path to save the splitted model
* **hf_token**: huggingface token can be provided here if downloading gated models like: *meta-llama/Llama-2-7b-hf*
* **prefetching**: prefetching to overlap the model loading and compute. By default, turned on. For now, only AirLLMLlama2 supports this.
* **delete_original**: if you don't have too much disk space, you can set delete_original to true to delete the original downloaded hugging face model, only keep the transformed one to save half of the disk space. 

## MacOS

Just install airllm and run the code the same as on linux. See more in [Quick Start](#quickstart).

* make sure you installed [mlx](https://github.com/ml-explore/mlx?tab=readme-ov-file#installation) and torch
* you probabaly need to install python native see more [here](https://stackoverflow.com/a/65432861/21230266)
* only [Apple silicon](https://support.apple.com/en-us/HT211814) is supported

Example [python notebook] (https://github.com/lyogavin/Anima/blob/main/air_llm/examples/run_on_macos.ipynb)


## Example Python Notebook

Example colabs here:

<a target="_blank" href="https://colab.research.google.com/github/lyogavin/Anima/blob/main/air_llm/examples/run_all_types_of_models.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

#### example of other models (ChatGLM, QWen, Baichuan, Mistral, etc):

<details>


* ChatGLM:

```python
from airllm import AutoModel
MAX_LENGTH = 128
model = AutoModel.from_pretrained("THUDM/chatglm3-6b-base")
input_text = ['What is the capital of China?',]
input_tokens = model.tokenizer(input_text,
    return_tensors="pt", 
    return_attention_mask=False, 
    truncation=True, 
    max_length=MAX_LENGTH, 
    padding=True)
generation_output = model.generate(
    input_tokens['input_ids'].cuda(), 
    max_new_tokens=5,
    use_cache= True,
    return_dict_in_generate=True)
model.tokenizer.decode(generation_output.sequences[0])
```

* QWen:

```python
from airllm import AutoModel
MAX_LENGTH = 128
model = AutoModel.from_pretrained("Qwen/Qwen-7B")
input_text = ['What is the capital of China?',]
input_tokens = model.tokenizer(input_text,
    return_tensors="pt", 
    return_attention_mask=False, 
    truncation=True, 
    max_length=MAX_LENGTH)
generation_output = model.generate(
    input_tokens['input_ids'].cuda(), 
    max_new_tokens=5,
    use_cache=True,
    return_dict_in_generate=True)
model.tokenizer.decode(generation_output.sequences[0])
```


* Baichuan, InternLM, Mistral, etc:

```python
from airllm import AutoModel
MAX_LENGTH = 128
model = AutoModel.from_pretrained("baichuan-inc/Baichuan2-7B-Base")
#model = AutoModel.from_pretrained("internlm/internlm-20b")
#model = AutoModel.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
input_text = ['What is the capital of China?',]
input_tokens = model.tokenizer(input_text,
    return_tensors="pt", 
    return_attention_mask=False, 
    truncation=True, 
    max_length=MAX_LENGTH)
generation_output = model.generate(
    input_tokens['input_ids'].cuda(), 
    max_new_tokens=5,
    use_cache=True,
    return_dict_in_generate=True)
model.tokenizer.decode(generation_output.sequences[0])
```


</details>


#### To request other model support: [here](https://docs.google.com/forms/d/e/1FAIpQLSe0Io9ANMT964Zi-OQOq1TJmnvP-G3_ZgQDhP7SatN0IEdbOg/viewform?usp=sf_link)



## Acknowledgement

A lot of the code are based on SimJeg's great work in the Kaggle exam competition. Big shoutout to SimJeg:

[GitHub account @SimJeg](https://github.com/SimJeg), 
[the code on Kaggle](https://www.kaggle.com/code/simjeg/platypus2-70b-with-wikipedia-rag), 
[the associated discussion](https://www.kaggle.com/competitions/kaggle-llm-science-exam/discussion/446414).


## FAQ

### 1. MetadataIncompleteBuffer

safetensors_rust.SafetensorError: Error while deserializing header: MetadataIncompleteBuffer

If you run into this error, most possible cause is you run out of disk space. The process of splitting model is very disk-consuming. See [this](https://huggingface.co/TheBloke/guanaco-65B-GPTQ/discussions/12). You may need to extend your disk space, clear huggingface [.cache](https://huggingface.co/docs/datasets/cache) and rerun. 

### 2. ValueError: max() arg is an empty sequence

Most likely you are loading QWen or ChatGLM model with Llama2 class. Try the following:

For QWen model: 

```python
from airllm import AutoModel #<----- instead of AirLLMLlama2
AutoModel.from_pretrained(...)
```

For ChatGLM model: 

```python
from airllm import AutoModel #<----- instead of AirLLMLlama2
AutoModel.from_pretrained(...)
```

### 3. 401 Client Error....Repo model ... is gated.

Some models are gated models, needs huggingface api token. You can provide hf_token:

```python
model = AutoModel.from_pretrained("meta-llama/Llama-2-7b-hf", #hf_token='HF_API_TOKEN')
```

### 4. ValueError: Asking to pad but the tokenizer does not have a padding token.

Some model's tokenizer doesn't have padding token, so you can set a padding token or simply turn the padding config off:

 ```python
input_tokens = model.tokenizer(input_text,
    return_tensors="pt", 
    return_attention_mask=False, 
    truncation=True, 
    max_length=MAX_LENGTH, 
    padding=False  #<-----------   turn off padding 
)
```

## Citing AirLLM

If you find
AirLLM useful in your research and wish to cite it, please use the following
BibTex entry:

```
@software{airllm2023,
  author = {Gavin Li},
  title = {AirLLM: scaling large language models on low-end commodity computers},
  url = {https://github.com/lyogavin/Anima/tree/main/air_llm},
  version = {0.0},
  year = {2023},
}
```


## Contribution 

Welcomed contributions, ideas and discussions!

If you find it useful, please ⭐ or buy me a coffee! 🙏

[!["Buy Me A Coffee"](https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png)](https://bmc.link/lyogavinQ)
