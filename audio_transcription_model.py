from typing import List, Optional

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedModel, WhisperConfig, WhisperModel

from .audio_transcription_config import AudioTranscriptionConfig

IGNORE_INDEX = -100
WHISPER_HOP_LENGTH = 160
WHISPER_CONV_STRIDE = 2
WHISPER_SAMPLE_RATE = 16000
WHISPER_MAX_ENCODER_FRAMES = 1500


def _rebuild_auto_config(config_dict):
    config_dict = dict(config_dict)
    model_type = config_dict.pop("model_type")
    return AutoConfig.for_model(model_type, **config_dict)


class AudioTranscriptionModel(PreTrainedModel):
    config_class = AudioTranscriptionConfig
    base_model_prefix = ""
    main_input_name = "audio_features"
    _tied_weights_keys = []

    def __init__(self, config: AudioTranscriptionConfig):
        super().__init__(config)
        self.all_tied_weights_keys = {}

        whisper_config = WhisperConfig.from_dict(config.whisper_config)
        llm_config = _rebuild_auto_config(config.llm_config)

        whisper = WhisperModel(whisper_config)
        self.whisper_encoder = whisper.encoder
        del whisper.decoder
        self.whisper_hidden = self.whisper_encoder.config.d_model

        self.pool_kernel = config.pool_kernel
        self.max_pooled_len = WHISPER_MAX_ENCODER_FRAMES // self.pool_kernel
        self.avg_pool = nn.AvgPool1d(kernel_size=self.pool_kernel, stride=self.pool_kernel)

        self.llm = AutoModelForCausalLM.from_config(llm_config, trust_remote_code=True)
        self.llm_hidden = self.llm.config.hidden_size
        self.llm.config.use_cache = False

        self.audio_projector = nn.Sequential(
            nn.Linear(self.whisper_hidden, self.llm_hidden),
            nn.GELU(),
            nn.Linear(self.llm_hidden, self.llm_hidden),
        )

        prompt_ids = torch.tensor(config.prompt_ids, dtype=torch.long)
        self.register_buffer("_prompt_ids", prompt_ids, persistent=True)

    def get_input_embeddings(self):
        return self.llm.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.llm.set_input_embeddings(value)

    def _pooled_real_len(self, audio_lengths: torch.Tensor) -> torch.Tensor:
        samples_per_frame = WHISPER_HOP_LENGTH * WHISPER_CONV_STRIDE
        encoder_frames = torch.clamp(
            (audio_lengths.float() / samples_per_frame).ceil().long(),
            max=WHISPER_MAX_ENCODER_FRAMES,
        )
        return torch.clamp(
            (encoder_frames.float() / self.pool_kernel).ceil().long(),
            max=self.max_pooled_len,
        )

    def encode_audio(self, audio_features: torch.Tensor) -> torch.Tensor:
        hidden = self.whisper_encoder(audio_features).last_hidden_state
        pooled = self.avg_pool(hidden.transpose(1, 2)).transpose(1, 2)
        return self.audio_projector(pooled)

    def forward(
        self,
        audio_features: torch.Tensor,
        audio_lengths: torch.Tensor,
        transcript_ids: Optional[torch.Tensor] = None,
        transcript_attention_mask: Optional[torch.Tensor] = None,
    ):
        device = audio_features.device
        embed = self.llm.get_input_embeddings()

        audio_embeds = self.encode_audio(audio_features)
        pooled_lens = self._pooled_real_len(audio_lengths)
        prompt_embeds = embed(self._prompt_ids.to(device))
        prompt_len = prompt_embeds.shape[0]

        if transcript_ids is None:
            seqs: List[torch.Tensor] = []
            max_len = 0
            for i in range(audio_embeds.shape[0]):
                audio_len = pooled_lens[i].item()
                seq = torch.cat([prompt_embeds, audio_embeds[i, :audio_len]])
                seqs.append(seq)
                max_len = max(max_len, seq.shape[0])

            inputs_embeds = audio_embeds.new_zeros(audio_embeds.shape[0], max_len, self.llm_hidden)
            attention_mask = torch.zeros(audio_embeds.shape[0], max_len, dtype=torch.long, device=device)
            for i, seq in enumerate(seqs):
                seq_len = seq.shape[0]
                inputs_embeds[i, :seq_len] = seq
                attention_mask[i, :seq_len] = 1
            return self.llm(inputs_embeds=inputs_embeds, attention_mask=attention_mask)

        transcript_embeds = embed(transcript_ids)
        text_lens = transcript_attention_mask.sum(dim=1)

        seqs: List[torch.Tensor] = []
        labels: List[torch.Tensor] = []
        max_len = 0
        for i in range(audio_embeds.shape[0]):
            audio_len = pooled_lens[i].item()
            text_len = text_lens[i].item()
            seq = torch.cat([
                prompt_embeds,
                audio_embeds[i, :audio_len],
                transcript_embeds[i, :text_len],
            ])
            lab = torch.cat([
                torch.full((prompt_len + audio_len,), IGNORE_INDEX, dtype=torch.long, device=device),
                transcript_ids[i, :text_len],
            ])
            seqs.append(seq)
            labels.append(lab)
            max_len = max(max_len, seq.shape[0])

        inputs_embeds = audio_embeds.new_zeros(audio_embeds.shape[0], max_len, self.llm_hidden)
        padded_labels = torch.full((audio_embeds.shape[0], max_len), IGNORE_INDEX, dtype=torch.long, device=device)
        attention_mask = torch.zeros(audio_embeds.shape[0], max_len, dtype=torch.long, device=device)
        for i, seq in enumerate(seqs):
            seq_len = seq.shape[0]
            inputs_embeds[i, :seq_len] = seq
            padded_labels[i, :seq_len] = labels[i]
            attention_mask[i, :seq_len] = 1

        return self.llm(
            inputs_embeds=inputs_embeds,
            labels=padded_labels,
            attention_mask=attention_mask,
        )

    @torch.no_grad()
    def transcribe(
        self,
        audio_features: torch.Tensor,
        audio_lengths: torch.Tensor,
        **generate_kwargs,
    ) -> torch.LongTensor:
        device = audio_features.device
        embed = self.llm.get_input_embeddings()

        audio_embeds = self.encode_audio(audio_features)
        pooled_lens = self._pooled_real_len(audio_lengths)
        prompt_embeds = embed(self._prompt_ids.to(device))

        seqs: List[torch.Tensor] = []
        max_len = 0
        for i in range(audio_embeds.shape[0]):
            audio_len = pooled_lens[i].item()
            seq = torch.cat([prompt_embeds, audio_embeds[i, :audio_len]])
            seqs.append(seq)
            max_len = max(max_len, seq.shape[0])

        inputs_embeds = audio_embeds.new_zeros(audio_embeds.shape[0], max_len, self.llm_hidden)
        attention_mask = torch.zeros(audio_embeds.shape[0], max_len, dtype=torch.long, device=device)
        for i, seq in enumerate(seqs):
            seq_len = seq.shape[0]
            inputs_embeds[i, :seq_len] = seq
            attention_mask[i, :seq_len] = 1

        self.llm.config.use_cache = True
        output = self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **generate_kwargs,
        )
        self.llm.config.use_cache = False
        return output
