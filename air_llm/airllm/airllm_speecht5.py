import torch
from transformers import SpeechT5ForTextToSpeech, SpeechT5HifiGan, SpeechT5Processor


class AirLLMSpeechT5:
    """SpeechT5 text-to-speech backend for AutoModel.

    This backend is intentionally separate from the layer-streaming AirLLMBaseModel
    path because SpeechT5 is an encoder-decoder speech model, not a causal LM.
    """

    def __init__(
        self,
        model_local_path_or_repo_id,
        device="cpu",
        dtype=torch.float32,
        hf_token=None,
        vocoder_model_id="microsoft/speecht5_hifigan",
        **kwargs,
    ):
        self.model_local_path_or_repo_id = model_local_path_or_repo_id
        self.running_device = device
        self.device = torch.device(device)
        self.running_dtype = dtype
        self.dtype = dtype
        self.hf_token = hf_token
        self.vocoder_model_id = vocoder_model_id

        self.processor = SpeechT5Processor.from_pretrained(
            model_local_path_or_repo_id,
            token=hf_token,
            trust_remote_code=True,
        )
        self.tokenizer = self.processor.tokenizer

        self.model = SpeechT5ForTextToSpeech.from_pretrained(
            model_local_path_or_repo_id,
            token=hf_token,
            trust_remote_code=True,
        ).to(self.device)
        self.model.eval()

        self.vocoder = SpeechT5HifiGan.from_pretrained(
            vocoder_model_id,
            token=hf_token,
            trust_remote_code=True,
        ).to(self.device)
        self.vocoder.eval()

    def get_processor(self):
        return self.processor

    def tts(self, text, speaker_embeddings=None):
        if isinstance(text, str):
            text = [text]

        inputs = self.processor(text=text, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)

        batch_size = input_ids.shape[0]
        if speaker_embeddings is None:
            speaker_embeddings = torch.zeros((batch_size, 512), device=self.device)
        else:
            speaker_embeddings = speaker_embeddings.to(self.device)

        with torch.no_grad():
            wav = self.model.generate_speech(
                input_ids,
                speaker_embeddings,
                vocoder=self.vocoder,
            )
        return wav

    def generate(self, input_ids, **kwargs):
        speaker_embeddings = kwargs.pop("speaker_embeddings", None)
        if input_ids is None:
            raise ValueError("input_ids is required for SpeechT5 generation")

        if isinstance(input_ids, torch.Tensor):
            ids = input_ids.to(self.device)
        else:
            ids = torch.tensor(input_ids, device=self.device)

        if speaker_embeddings is None:
            speaker_embeddings = torch.zeros((ids.shape[0], 512), device=self.device)
        else:
            speaker_embeddings = speaker_embeddings.to(self.device)

        with torch.no_grad():
            wav = self.model.generate_speech(
                ids,
                speaker_embeddings,
                vocoder=self.vocoder,
            )
        return wav
