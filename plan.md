# Audio Transcription Model Training Plan
**Architecture:** Whisper Base (`openai/whisper-base`) → Average Pooling → Trainable MLP Projection → Llama 3 1B (`meta-llama/Llama-3.2-1B`)
**Objective:** Pure transcription (Speech-to-Text) via autoregressive generation.
**Dataset:** 50,000 combined samples (25k `parler-tts/mls_eng` + 25k `openslr/librispeech_asr`).

---

## Phase 1: Environment & Data Preparation
**Goal:** Ingest, balance, and preprocess the audio and text data into a unified format.

1. **Environment Setup:**
   * Install necessary libraries: `transformers`, `datasets`, `torchaudio`, `accelerate`, `evaluate`, `jiwer`, and `librosa`.
2. **Dataset Loading & Sampling:**
   * Load `parler-tts/mls_eng` and `openslr/librispeech_asr`.
   * Shuffle and sample exactly 25,000 records from the `train` split of each dataset.
   * Concatenate them into a single 50,000-sample Hugging Face Dataset.
3. **Audio Preprocessing:**
   * **Resampling:** Ensure all audio arrays are strictly resampled to 16,000 Hz.
   * **Feature Extraction:** Pass the raw audio arrays through `WhisperFeatureExtractor` to compute the log-Mel spectrograms. The feature extractor pads/truncates all audio to 30 seconds (480,000 samples) before computing the spectrogram, yielding a fixed 1500-frame output from the encoder.
   * **Attention Mask / Padding Tracking:** Compute the actual (unpadded) length of each audio clip in Whisper encoder frames: `real_frames = ceil(sample_rate * duration / hop_length)`, capped at 1500. Store this per-sample so that after pooling, the padded audio tokens can be identified and excluded before feeding into the LLM. This avoids wasting LLM context window on silence and prevents the model from learning to attend to meaningless padding.
4. **Text Preprocessing:**
   * Normalize the transcriptions to lowercase.
   * Tokenize the transcripts using the `meta-llama/Llama-3.2-1B` tokenizer.
   * Append `<|end_of_text|>` (token ID 128001) as the stop token after every transcript.

---

## Phase 2: Model Architecture Initialization
**Goal:** Assemble the hybrid model, implement the pooling strategy, and define trainable components.

1. **The Audio Encoder (`openai/whisper-base`):**
   * Load the Whisper Base encoder.
   * **Action:** Freeze all parameters (`requires_grad = False`).
2. **The Sequence Reducer (Average Pooling):**
   * Utilize `nn.AvgPool1d(kernel_size=4, stride=4)`.
   * **Action:** This is a parameter-free operation. It reduces the Whisper output sequence length by a factor of 4 (e.g., 1500 frames → 375 tokens). Each output token represents ~80ms of audio.
   * After pooling, use the tracked real frame lengths from Phase 1 to compute the pooled real length: `pooled_real_len = ceil(real_frames / 4)`. Discard tokens beyond this index before passing to the LLM.
3. **The Projection Layer (2-layer MLP):**
   * Initialize a 2-layer MLP with GELU activation: `Linear(512, 2048) → GELU → Linear(2048, 2048)`.
   * **Input Dimension:** 512 (Whisper Base hidden size).
   * **Output Dimension:** 2048 (Llama 3.2 1B hidden size).
   * **Action:** Leave these weights **unfrozen** (trainable).
4. **The Text Prompt Prefix:**
   * Prepend a fixed text prompt `"Transcribe:"` before the projected audio tokens in every sample. This is embedded via Llama's own embedding layer and gives the LLM context about the expected task.
   * The full input sequence to Llama is: `[<|begin_of_text|>] [Transcribe:] [audio_token_1, ..., audio_token_N] [target_text_tokens] [<|end_of_text|>]`.
5. **The Language Model (`meta-llama/Llama-3.2-1B`):**
   * Load the 1-Billion parameter Llama 3.2 model.
   * **Stage A:** Freeze all LLM parameters.
   * **Stage B:** Unfreeze all LLM parameters (see Phase 3).

---

## Phase 3: Training
**Goal:** Train in two stages — first align audio features to the LLM embedding space, then fine-tune the full model end-to-end.

### Stage A: Projection Alignment (Projector-Only Training)

1. **Trainable Parameters:** MLP projection layer only. Whisper and Llama are frozen.
2. **Forward Pass:**
   * Encode audio: $Z_{audio} = \text{WhisperEncoder}(X_{audio})$ → shape `(B, 1500, 512)`
   * Pool: $Z_{pooled} = \text{AvgPool1d}(Z_{audio}, \text{kernel}=4)$ → shape `(B, 375, 512)`
   * Truncate padding: keep only the first `pooled_real_len` tokens per sample.
   * Project: $H_{audio} = \text{MLP}(Z_{pooled})$ → shape `(B, N, 2048)` where N ≤ 375.
   * Embed prompt: $H_{prompt} = \text{LlamaEmbed}(\text{"Transcribe:"})$
   * Embed target text: $H_{text} = \text{LlamaEmbed}(\text{transcript\_tokens})$
   * Concatenate: `inputs_embeds = [H_prompt, H_audio, H_text]`
   * Feed `inputs_embeds` into the frozen Llama and compute cross-entropy loss.
3. **Label Masking:** The loss is computed **only** over the transcript text tokens. The prompt tokens and audio tokens are masked with `IGNORE_INDEX = -100` in the labels so they do not contribute to the loss.
4. **Hyperparameters:**
   * **Optimizer:** AdamW.
   * **Learning Rate:** ~2e-3 with cosine schedule and 3% warmup.
   * **Batch Size:** Maximize based on VRAM.
   * **Epochs:** ~1 epoch over the 50k dataset.
5. **Expected Loss Curve:**
   * Steep initial drop as the MLP learns the linear mapping from 512-dim audio space to 2048-dim LLM space.
   * Followed by a plateau — the projection can align features but the frozen LLM cannot adapt its attention patterns to the new audio token distribution. This is the signal to move to Stage B.

### Stage B: End-to-End Fine-Tuning (Full LLM + Projector)

1. **Trainable Parameters:** MLP projection layer + all Llama 3.2 1B parameters. Whisper remains frozen.
2. **Initialization:** Load the Stage A checkpoint (trained projector weights).
3. **Forward Pass:** Same as Stage A, but gradients now flow through the entire Llama model.
4. **Hyperparameters:**
   * **Optimizer:** AdamW.
   * **Learning Rate:** ~2e-5 (much lower than Stage A to avoid catastrophic forgetting of Llama's language capabilities). Cosine schedule with 3% warmup.
   * **Batch Size:** Will be smaller than Stage A due to increased memory from LLM gradients/optimizer states. Use gradient accumulation to maintain effective batch size. DeepSpeed ZeRO-2 or ZeRO-3 recommended.
   * **Epochs:** 1–3 epochs depending on loss convergence.
5. **Expected Loss Curve:**
   * The loss should resume dropping from the Stage A plateau as Llama's attention layers learn to properly attend to and interpret the audio token representations.
   * Monitor for overfitting on the 50k dataset — if validation loss starts climbing, stop early or increase data.

---

## Phase 4: Evaluation (WER) & Inference
**Goal:** Test the model's ability to transcribe unseen audio and set up an interactive test.

1. **Validation Set:**
   * Reserve a holdout set from the `test` splits of LibriSpeech and MLS.
2. **WER Evaluation:**
   * Generate predictions for the validation set using beam search (beam width = 5).
   * Decode the predicted Llama 3 tokens to strings and calculate the Word Error Rate (`wer` metric via the `evaluate` library) against the ground-truth references.
   * Evaluate after both Stage A and Stage B to quantify the impact of LLM fine-tuning.
3. **Inference (Interactive Audio Testing):**
   * Write an `inference.py` script.
   * Prompt the user for an audio file path.
   * Push the audio through: Whisper Encoder → Avg Pool (x4) → Truncate padding → MLP Projection → Llama 3.2 1B.
   * Print the generated transcription to the console.

---

## Phase 5: Future Ideas
* **Learned Downsampling:** Replace `AvgPool1d` with a strided `nn.Conv1d(512, 512, kernel_size=4, stride=4)`. This adds minimal parameters but lets the model learn which features to preserve during compression rather than averaging uniformly.
* **Pooling Factor Experiments:** The current kernel size of 4 compresses 80ms of audio into a single token. If WER is poor, try kernel=2 (40ms, ~750 tokens) for finer granularity. If compute is tight, try kernel=8 (160ms, ~187 tokens) to trade accuracy for speed.
* **Scaling Data:** If results plateau, scale to 200k+ samples. LibriSpeech alone has ~960 hours of training data.
* **Mixed-Case Transcription:** Currently lowercased. Experiment with preserving original casing to match Llama's pretraining distribution.
