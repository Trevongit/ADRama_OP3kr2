# engines.py
import os, json

class TTSEngine:
    def list_voices(self): return []
    def synthesize(self, text: str, output_path: str, voice: str | None = None): raise NotImplementedError

class OpenAITTSEngine(TTSEngine):
    def __init__(self, credentials_path: str | None = None, model: str = "tts-1"):
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError("openai package not installed. Run: pip install openai") from e

        api_key = None
        if credentials_path and os.path.exists(credentials_path):
            with open(credentials_path, "r", encoding="utf-8") as f:
                api_key = (json.load(f) or {}).get("api_key")
        if not api_key:
            api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("No valid OpenAI API key found")

        self.client = OpenAI(api_key=api_key)
        self.model = model
        self._voices = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]

    def list_voices(self): return self._voices

    def synthesize(self, text: str, output_path: str, voice: str | None = None):
        v = voice if voice in self._voices else "alloy"
        resp = self.client.audio.speech.create(
            model=self.model, voice=v, input=text, response_format="mp3"
        )
        with open(output_path, "wb") as f:
            f.write(resp.content)

class GoogleCloudTTSEngine(TTSEngine):
    def __init__(self, credentials_path: str | None = None, default_lang: str = "en-US"):
        try:
            from google.cloud import texttospeech
        except ImportError as e:
            raise ImportError("google-cloud-texttospeech not installed. Run: pip install google-cloud-texttospeech") from e

        if credentials_path:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path

        self.tts_mod = texttospeech
        self.client = texttospeech.TextToSpeechClient()
        self.default_lang = default_lang
        self._voices_cache = None

    def list_voices(self):
        if self._voices_cache is None:
            resp = self.client.list_voices(request={})
            self._voices_cache = sorted(v.name for v in resp.voices if getattr(v, "name", None))
        return self._voices_cache

    def synthesize(self, text: str, output_path: str, voice: str | None = None):
        lang = self.default_lang
        voices = self.list_voices()
        selected = voice if voice and voice in voices else f"{lang}-Wavenet-D"
        if selected not in voices:
            raise RuntimeError(f"Voice '{selected}' not available")
        voice_params = self.tts_mod.VoiceSelectionParams(language_code=lang, name=selected)
        audio_cfg   = self.tts_mod.AudioConfig(audio_encoding=self.tts_mod.AudioEncoding.MP3)
        input_cfg   = self.tts_mod.SynthesisInput(text=text)
        resp = self.client.synthesize_speech(input=input_cfg, voice=voice_params, audio_config=audio_cfg)
        with open(output_path, "wb") as f:
            f.write(resp.audio_content)
