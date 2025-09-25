import os
import json
import time
import warnings
import io
import yaml  # For inline config load (from your txt)
import torch
import torchaudio
import librosa
import torch.nn.functional as F
from typing import Optional, List, Dict, Any, Union
from omegaconf import OmegaConf
from torch.nn.utils.rnn import pad_sequence

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Inline your config.yaml as dict (from uploaded txt) - load via yaml for flexibility
CONFIG_YAML = """
dataset:
    bpe_model: bpe.model
    sample_rate: 24000
    squeeze: false
    mel:
        sample_rate: 24000
        n_fft: 1024
        hop_length: 256
        win_length: 1024
        n_mels: 100
        mel_fmin: 0
        normalize: false

gpt:
    model_dim: 1280
    max_mel_tokens: 1815
    max_text_tokens: 600
    heads: 20
    use_mel_codes_as_input: true
    mel_length_compression: 1024
    layers: 24
    number_text_tokens: 12000
    number_mel_codes: 8194
    start_mel_token: 8192
    stop_mel_token: 8193
    start_text_token: 0
    stop_text_token: 1
    train_solo_embeddings: false
    condition_type: "conformer_perceiver"
    condition_module:
        output_size: 512
        linear_units: 2048
        attention_heads: 8
        num_blocks: 6
        input_layer: "conv2d2"
        perceiver_mult: 2
    emo_condition_module:
        output_size: 512
        linear_units: 1024
        attention_heads: 4
        num_blocks: 4
        input_layer: "conv2d2"
        perceiver_mult: 2

semantic_codec:
    codebook_size: 8192
    hidden_size: 1024
    codebook_dim: 8
    vocos_dim: 384
    vocos_intermediate_dim: 2048
    vocos_num_layers: 12

s2mel:
    preprocess_params:
        sr: 22050
        spect_params:
            n_fft: 1024
            win_length: 1024
            hop_length: 256
            n_mels: 80
            fmin: 0
            fmax: "None"

    dit_type: "DiT"
    reg_loss_type: "l1"
    style_encoder:
        dim: 192
    length_regulator:
        channels: 512
        is_discrete: false
        in_channels: 1024
        content_codebook_size: 2048
        sampling_ratios: [1, 1, 1, 1]
        vector_quantize: false
        n_codebooks: 1
        quantizer_dropout: 0.0
        f0_condition: false
        n_f0_bins: 512
    DiT:
        hidden_dim: 512
        num_heads: 8
        depth: 13
        class_dropout_prob: 0.1
        block_size: 8192
        in_channels: 80
        style_condition: true
        final_layer_type: 'wavenet'
        target: 'mel'
        content_dim: 512
        content_codebook_size: 1024
        content_type: 'discrete'
        f0_condition: false
        n_f0_bins: 512
        content_codebooks: 1
        is_causal: false
        long_skip_connection: true
        zero_prompt_speech_token: false
        time_as_token: false
        style_as_token: false
        uvit_skip_connection: true
        add_resblock_in_transformer: false
    wavenet:
        hidden_dim: 512
        num_layers: 8
        kernel_size: 5
        dilation_rate: 1
        p_dropout: 0.2
        style_condition: true

gpt_checkpoint: gpt.pth
w2v_stat: wav2vec2bert_stats.pt
s2mel_checkpoint: s2mel.pth
emo_matrix: feat2.pt 
spk_matrix: feat1.pt
emo_num: [3, 17, 2, 8, 4, 5, 10, 24]
qwen_emo_path: qwen0.6bemo4-merge/ 
vocoder:
    type: "bigvgan"
    name: "nvidia/bigvgan_v2_22khz_80band_256x"
version: 2.0
"""

# Stub imports - assume user installs/ copies from fork (e.g., indextts.gpt.model_v2.UnifiedVoice)
# For full: from your fork's indextts/ dir
try:
    from indextts.gpt.model_v2 import UnifiedVoice  # From fork
    from indextts.utils.maskgct_utils import build_semantic_model, build_semantic_codec
    from indextts.utils.checkpoint import load_checkpoint
    from indextts.utils.front import TextNormalizer, TextTokenizer
    from indextts.s2mel.modules.commons import load_checkpoint2, MyModel
    from indextts.s2mel.modules.audio import mel_spectrogram
    from transformers import AutoTokenizer, AutoModelForCausalLM, SeamlessM4TFeatureExtractor
    from modelscope import AutoModelForCausalLM as MSModelForCausalLM  # If needed
    from huggingface_hub import hf_hub_download
    import safetensors
    from indextts.s2mel.modules.bigvgan import BigVGAN  # From uploaded bigvgan.py
    # ... other stubs: CAMPPlus, etc.
except ImportError as e:
    raise ImportError(f"Missing deps from IndexTTS fork. Clone & copy indextts/ to plugins/. Error: {e}")

class IndexTTS2Core:
    def __init__(self, cfg_path: str = None, model_dir: str = "./checkpoints", use_fp16: bool = False, 
                 device: Optional[str] = None, use_cuda_kernel: bool = False, use_deepspeed: bool = False):
        """
        Streamlined init from infer_v2.py.
        cfg_path: If None, uses inline YAML.
        """
        # Device setup (from infer_v2)
        if device is not None:
            self.device = device
            self.use_fp16 = False if device == "cpu" else use_fp16
            self.use_cuda_kernel = use_cuda_kernel and device.startswith("cuda")
        elif torch.cuda.is_available():
            self.device = "cuda:0"
            self.use_fp16 = use_fp16
            self.use_cuda_kernel = use_cuda_kernel
        # ... (MPS/CPU fallback as in code)
        else:
            self.device = "cpu"
            self.use_fp16 = False
            self.use_cuda_kernel = False

        self.model_dir = model_dir
        self.dtype = torch.float16 if self.use_fp16 else torch.float32
        self.stop_mel_token = 8193  # From config

        # Load config (inline or file)
        if cfg_path:
            self.cfg = OmegaConf.load(cfg_path)
        else:
            self.cfg = OmegaConf.create(yaml.safe_load(CONFIG_YAML))

        # QwenEmotion (from infer_v2)
        self.qwen_emo = QwenEmotion(os.path.join(self.model_dir, self.cfg.qwen_emo_path))

        # GPT Model (UnifiedVoice)
        self.gpt = UnifiedVoice(**self.cfg.gpt)
        gpt_path = os.path.join(model_dir, self.cfg.gpt_checkpoint)
        load_checkpoint(self.gpt, gpt_path)
        self.gpt = self.gpt.to(self.device)
        if self.use_fp16:
            self.gpt = self.gpt.half()
        self.gpt.eval()
        print(f">> GPT loaded from {gpt_path}")

        # Post-init (kv_cache, etc.)
        self.gpt.post_init_gpt2_config(use_deepspeed=use_deepspeed, kv_cache=True, half=self.use_fp16)

        # Semantic Model (w2v-bert)
        self.extract_features = SeamlessM4TFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")
        w2v_stat_path = os.path.join(model_dir, self.cfg.w2v_stat)
        self.semantic_model, self.semantic_mean, self.semantic_std = build_semantic_model(w2v_stat_path)
        self.semantic_model = self.semantic_model.to(self.device).eval()

        # S2Mel (stub - load from s2mel.pth)
        s2mel_path = os.path.join(model_dir, self.cfg.s2mel_checkpoint)
        self.s2mel = MyModel.from_pretrained(s2mel_path)  # Adapt load_checkpoint2
        self.s2mel = self.s2mel.to(self.device).eval()
        if self.use_fp16:
            self.s2mel = self.s2mel.half()

        # Vocoder: BigVGAN (from uploaded bigvgan.py)
        voc_cfg = self.cfg.vocoder
        self.vocoder = BigVGAN.from_pretrained(
            model_id=voc_cfg.name, 
            use_cuda_kernel=self.use_cuda_kernel,
            device=self.device,
            dtype=self.dtype
        )
        self.vocoder.eval()
        if self.use_fp16:
            self.vocoder = self.vocoder.half()

        # Other stubs (e.g., text normalizer)
        self.text_normalizer = TextNormalizer()  # From utils.front
        self.text_tokenizer = TextTokenizer(self.cfg.dataset.bpe_model)

    def infer(self, spk_audio_prompt: str, text: str, output_path: str,
              emo_vector: Optional[List[float]] = None,
              emo_alpha: float = 1.0,
              use_emo_text: bool = False,
              emo_text: Optional[str] = None,
              emo_audio_prompt: Optional[str] = None,
              use_random: bool = False,
              verbose: bool = False) -> None:
        """
        Core infer from infer_v2.py (reconstructed from partial code).
        Generates WAV, converts to MP3 for ADRama.
        """
        if os.environ.get("ADRAMA_DRY_RUN"):
            with open(output_path, "w") as f:
                f.write(f"[DRY RUN] IndexTTS: text='{text}', spk='{spk_audio_prompt}', "
                        f"emo_vector={emo_vector or 'None'}, emo_text='{emo_text or ''}'\n")
            if verbose:
                print(">> [DRY RUN] Audio generation skipped.")
            return

        start_time = time.time()
        if verbose:
            print(f">> Generating: '{text}'")

        # 1. Text Processing
        normalized_text = self.text_normalizer(text)
        text_tokens = self.text_tokenizer(normalized_text)

        # 2. Speaker/Emo Features (semantic from w2v)
        waveform, sr = torchaudio.load(spk_audio_prompt)
        waveform = waveform.to(self.device)
        if sr != self.cfg.dataset.sample_rate:
            waveform = librosa.resample(waveform.cpu().numpy(), orig_sr=sr, target_sr=self.cfg.dataset.sample_rate)
            waveform = torch.from_numpy(waveform).to(self.device)

        # Extract features
        inputs = self.extract_features(waveform.squeeze().cpu(), sampling_rate=self.cfg.dataset.sample_rate, return_tensors="pt")
        with torch.no_grad():
            features = self.semantic_model(inputs.input_values.to(self.device)).last_hidden_state.mean(dim=1)
            features = (features - self.semantic_mean.to(self.device)) / self.semantic_std.to(self.device)

        spk_features = features  # Timbre from spk_prompt

        # Emo Fusion
        if emo_vector:
            # Encode vector (stub - from emo_matrix)
            emo_feats = self._encode_emo_vector(emo_vector, alpha=emo_alpha)
        elif use_emo_text and emo_text:
            emo_dict = self.qwen_emo.inference(emo_text)
            emo_vector = [emo_dict.get(k, 0.0) for k in ["happy", "angry", "sad", "afraid", "disgusted", "melancholic", "surprised", "calm"]]
            emo_feats = self._encode_emo_vector(emo_vector, alpha=emo_alpha)
        elif emo_audio_prompt:
            # Similar to spk, but emo-specific (stub)
            emo_wave, _ = torchaudio.load(emo_audio_prompt)
            emo_feats = self._extract_emo_features(emo_wave.to(self.device))
        else:
            emo_feats = torch.zeros_like(spk_features)

        fused_feats = spk_features + emo_feats  # Disentangle & fuse

        # 3. GPT AR Generation: Text tokens + fused feats → mel tokens
        with torch.no_grad():
            mel_tokens = self.gpt.generate(
                text_tokens.to(self.device), 
                fused_feats, 
                max_length=self.cfg.gpt.max_mel_tokens,
                use_random=use_random,
                stop_token=self.stop_mel_token
            )  # Returns mel codes

        # 4. S2Mel: Mel tokens → Mel spectrogram
        mel_spec = self.s2mel(mel_tokens)  # Stub: Adapt to your s2mel modules

        # 5. Vocode: Mel → Waveform via BigVGAN
        with torch.no_grad():
            waveform = self.vocoder(mel_spec)

        # Save as MP3 (ADRama compat)
        from pydub import AudioSegment
        wav_io = io.BytesIO()
        torchaudio.save(wav_io, waveform.cpu(), self.cfg.dataset.sample_rate)
        wav_io.seek(0)
        audio_seg = AudioSegment.from_wav(wav_io)
        audio_seg.export(output_path, format="mp3")

        if verbose:
            dur = time.time() - start_time
            print(f">> Generated '{output_path}' in {dur:.2f}s (len: {len(waveform)} samples)")

    # Stub helpers (flesh from full fork code)
    def _encode_emo_vector(self, vector: List[float], alpha: float) -> torch.Tensor:
        # Load emo_matrix from cfg.emo_matrix
        emo_path = os.path.join(self.model_dir, self.cfg.emo_matrix)
        emo_matrix = torch.load(emo_path, map_location=self.device)
        emo_feats = torch.mm(torch.tensor(vector).to(self.device).unsqueeze(0), emo_matrix) * alpha
        return emo_feats.squeeze()

    def _extract_emo_features(self, waveform: torch.Tensor) -> torch.Tensor:
        # Similar to spk, but emo-tuned (placeholder)
        return self.semantic_model(waveform).last_hidden_state.mean(dim=1)  # Stub

    def _encode_text_emo(self, text: str) -> List[float]:
        # Via Qwen
        return self.qwen_emo.inference(text)

# QwenEmotion class (direct from infer_v2.py - unchanged)
class QwenEmotion:
    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir, torch_dtype=torch.float16, device_map="auto"
        )
        self.prompt = "文本情感分类"
        self.cn_key_to_en = {
            "高兴": "happy", "愤怒": "angry", "悲伤": "sad", "恐惧": "afraid",
            "反感": "disgusted", "低落": "melancholic", "惊讶": "surprised", "自然": "calm",
        }
        self.desired_vector_order = ["高兴", "愤怒", "悲伤", "恐惧", "反感", "低落", "惊讶", "自然"]
        self.melancholic_words = {"低落", "melancholy", "melancholic", "depression", "depressed", "gloomy"}
        self.max_score = 1.2
        self.min_score = 0.0

    def clamp_score(self, value: float) -> float:
        return max(self.min_score, min(self.max_score, value))

    def convert(self, content: Dict[str, float]) -> Dict[str, float]:
        emotion_dict = {
            self.cn_key_to_en[cn_key]: self.clamp_score(content.get(cn_key, 0.0))
            for cn_key in self.desired_vector_order
        }
        if all(val <= 0.0 for val in emotion_dict.values()):
            print(">> No emotions; default calm")
            emotion_dict["calm"] = 1.0
        return emotion_dict

    def inference(self, text_input: str) -> Dict[str, float]:
        start = time.time()
        messages = [
            {"role": "system", "content": self.prompt},
            {"role": "user", "content": text_input}
        ]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        generated_ids = self.model.generate(
            **model_inputs, max_new_tokens=32768, pad_token_id=self.tokenizer.eos_token_id
        )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

        # Parse (from code)
        try:
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0
        content = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True)

        try:
            content = json.loads(content)
        except json.JSONDecodeError:
            content = {m.group(1): float(m.group(2)) for m in re.finditer(r'([^\s":.,]+?)"?\s*:\s*([\d.]+)', content)}

        # Melancholic swap
        text_lower = text_input.lower()
        if any(word in text_lower for word in self.melancholic_words):
            content["悲伤"], content["低落"] = content.get("低落", 0.0), content.get("悲伤", 0.0)

        return self.convert(content)

# BigVGAN (direct paste from uploaded bigvgan.py - full class)
# ... (Include the entire BigVGAN class code here; truncated in upload, but assume full from your fork)
# For brevity, placeholder: class BigVGAN(nn.Module): ... (from your bigvgan.py)
# Paste the full class definition from the document into this file.

if __name__ == "__main__":
    # Smoke test
    tts = IndexTTS2Core(model_dir="./checkpoints", use_fp16=False)
    tts.infer(
        spk_audio_prompt="examples/voice_01.wav",  # Your sample
        text="Welcome to ADRama with IndexTTS2.",
        output_path="test_gen.mp3",
        emo_vector=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],  # Calm
        verbose=True
    )
