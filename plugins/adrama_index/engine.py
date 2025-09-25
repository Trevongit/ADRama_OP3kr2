from engines import TTSEngine
from .core import IndexTTS2Core
import os

class IndexTTSEngine(TTSEngine):
    def __init__(self, model_dir: str = "./plugins/adrama_index/checkpoints",
                 use_fp16: bool = True, device: str = "auto", **core_kwargs):
        self.core = IndexTTS2Core(model_dir=model_dir, use_fp16=use_fp16, device=device, **core_kwargs)
        # Emo map from ADRama presets (expand with VoiceCapabilities)
        self.emo_map = {
            "audio_drama": [0.0, 0.4, 0.1, 0.0, 0.0, 0.0, 0.5, 0.0],  # Angry/surprised
            "audiobook_faithful": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],  # Calm
            "neutral": [0.0] * 8,
            "intense": [0.0, 0.7, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0],  # Angry/afraid
            # Map from "voice_emotion": e.g., "steady" → calm skew
        }

    def list_voices(self):
        return [{"type": "zero-shot", "requires": "WAV path in voice_map", "emo_control": True}]

    def synthesize(self, text: str, output_path: str, voice: Optional[str] = None, **kwargs) -> None:
        preset = kwargs.get("preset", "audio_drama")
        emotion = kwargs.get("voice_emotion", "neutral")
        emo_vector = self.emo_map.get(preset, self.emo_map.get(emotion, self.emo_map["neutral"]))
        instructions = kwargs.get("instructions", "")

        if not voice or not os.path.exists(voice):
            raise ValueError(f"Missing spk prompt for IndexTTS: {voice}")

        self.core.infer(
            spk_audio_prompt=voice,
            text=text,
            output_path=output_path,
            emo_vector=emo_vector,
            emo_alpha=kwargs.get("emo_alpha", 1.0),  # e.g., from pace
            use_emo_text=bool(instructions),
            emo_text=instructions,
            use_random=False  # Deterministic
        )
