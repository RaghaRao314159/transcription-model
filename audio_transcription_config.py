from transformers import PretrainedConfig


class AudioTranscriptionConfig(PretrainedConfig):
    model_type = "audio_transcription"

    def __init__(
        self,
        whisper_model_name="openai/whisper-base",
        llm_model_name="Qwen/Qwen3-1.7B",
        whisper_config=None,
        llm_config=None,
        pool_kernel=4,
        prompt_text="Transcribe:",
        prompt_ids=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.whisper_model_name = whisper_model_name
        self.llm_model_name = llm_model_name
        self.whisper_config = whisper_config or {}
        self.llm_config = llm_config or {}
        self.pool_kernel = pool_kernel
        self.prompt_text = prompt_text
        self.prompt_ids = prompt_ids or []
