#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ADRama TTS Studio v2.2.0
- Multi-engine TTS: OpenAI TTS, Google Cloud TTS
- Loads scripts from .txt ("Name: dialogue"), .jsonl, .csv
- Transform raw .txt chapter -> multi-actor script (JSONL + CSV) via OpenAI
- Per-actor voice assignment with engine selection and demo playback
- Exports per-line MP3s and a merged episode MP3 with caching
- Responsive UI with threading, progress bar, and scrollable voice assignment
- Reads config from ./config.json or ~/.adrama_config.json
- Compatible with .env loading from start_adramaz.sh
- v2.1.5: Threaded Google Voice Browser demo to prevent freezes, added timeout
- v2.2.0: Removed pyttsx3; default engine/menus now Google/OpenAI only
"""

import os
import sys
import json
import time
import re
import csv
import ast
import threading
import queue
import subprocess
import platform
import shutil
import logging
import requests
import zipfile
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
from concurrent.futures import ThreadPoolExecutor, TimeoutError, as_completed
from datetime import datetime
from pathlib import Path

from google.api_core import exceptions as google_exceptions
from pydub import AudioSegment

from adrama_core import ScriptLine, ScriptData, sanitize_filename
from transformer_service import (
    DEFAULT_PRESET_ID,
    TransformError,
    TransformRequest,
    list_presets,
    run_transform_job,
)
from build_instructions import build_instructions

# GUI
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog

# Setup logging
logging.basicConfig(
    filename="adrama_errors.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

APP_NAME = "ADRama TTS Studio"
APP_VERSION = "v2.2.0"
LOCAL_CONFIG = os.path.join(os.getcwd(), "config.json")
HOME_CONFIG = os.path.join(os.path.expanduser("~"), ".adrama_config.json")

SUPPORTED_EXTS = (".txt", ".jsonl", ".csv")
DEFAULT_OUTDIR = os.path.join(os.getcwd(), "out")
DEFAULT_LINES_DIRNAME = "lines"
DEFAULT_MERGE_FILENAME = "episode_merged.mp3"
DEFAULT_TRANSFORM_SUBDIR = "acted_scripts"
SAMPLE_TEXT = "This is a test of the voice."
DEMOS_DIR = os.path.join(os.getcwd(), "demos")
GOOGLE_PREMIUM_KINDS = {"Wavenet", "Neural2", "Studio", "Journey", "Gemini"}
MAX_TTS_WORKERS = 4
ENGINE_WORKER_LIMITS = {
    "openai": 1,
    "google": 4,
}

LANGUAGE_NAMES = {
    "af": "Afrikaans",
    "ar": "Arabic",
    "bg": "Bulgarian",
    "bn": "Bengali",
    "ca": "Catalan",
    "cs": "Czech",
    "da": "Danish",
    "de": "German",
    "el": "Greek",
    "en": "English",
    "es": "Spanish",
    "et": "Estonian",
    "fi": "Finnish",
    "fil": "Filipino",
    "fr": "French",
    "gu": "Gujarati",
    "he": "Hebrew",
    "hi": "Hindi",
    "hr": "Croatian",
    "hu": "Hungarian",
    "id": "Indonesian",
    "it": "Italian",
    "ja": "Japanese",
    "kn": "Kannada",
    "ko": "Korean",
    "lt": "Lithuanian",
    "lv": "Latvian",
    "ml": "Malayalam",
    "mr": "Marathi",
    "ms": "Malay",
    "nl": "Dutch",
    "no": "Norwegian",
    "pl": "Polish",
    "pt": "Portuguese",
    "ro": "Romanian",
    "ru": "Russian",
    "sk": "Slovak",
    "sl": "Slovenian",
    "sr": "Serbian",
    "sv": "Swedish",
    "ta": "Tamil",
    "te": "Telugu",
    "th": "Thai",
    "tr": "Turkish",
    "uk": "Ukrainian",
    "vi": "Vietnamese",
    "zh": "Chinese",
}

REGION_NAMES = {
    "AU": "Australia",
    "GB": "United Kingdom",
    "IN": "India",
    "US": "United States",
    "CA": "Canada",
    "ZA": "South Africa",
    "BR": "Brazil",
    "ES": "Spain",
    "FR": "France",
    "DE": "Germany",
    "IT": "Italy",
    "JP": "Japan",
    "KR": "South Korea",
    "MX": "Mexico",
    "SE": "Sweden",
    "NO": "Norway",
    "FI": "Finland",
    "DK": "Denmark",
    "NL": "Netherlands",
    "TW": "Taiwan",
    "HK": "Hong Kong",
    "CN": "China",
    "PH": "Philippines",
    "ID": "Indonesia",
    "RU": "Russia",
    "PL": "Poland",
    "TR": "Turkey",
    "VN": "Vietnam",
}

def format_language_label(code: str) -> str:
    if not code:
        return "Unknown"
    parts = code.split("-")
    lang_part = parts[0].lower()
    region_part = parts[1].upper() if len(parts) > 1 else ""
    lang_name = LANGUAGE_NAMES.get(lang_part, lang_part)
    region_name = REGION_NAMES.get(region_part, region_part)
    if region_part:
        return f"{lang_name} ({region_name})"
    return lang_name

class TTSEngine:
    def list_voices(self) -> List[str]:
        return []
    def synthesize(
        self,
        text: str,
        output_path: str,
        voice: Optional[str] = None,
        instructions: str = "",
    ) -> None:
        raise NotImplementedError

class OpenAITTSEngine(TTSEngine):
    def __init__(self, credentials_path: Optional[str] = None, model: str = "tts-1"):
        try:
            from openai import OpenAI
        except ImportError:
            logging.error("openai package not installed. Run: pip install openai")
            raise ImportError("openai package not installed. Run: pip install openai")
        api_key = None
        if credentials_path and os.path.exists(credentials_path):
            try:
                with open(credentials_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                    api_key = cfg.get("api_key")
                logging.info(f"Loaded OpenAI API key from {credentials_path}")
            except Exception as e:
                logging.error(f"Failed to load OpenAI credentials from {credentials_path}: {e}")
        if not api_key:
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError("No valid OpenAI API key found")
            logging.info("Using OpenAI API key from environment variable")
        self.api_key = api_key
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self._voices = self._load_voice_list()
        if not self._voices:
            # fallback to known voices if dynamic fetch fails
            self._voices = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
        logging.info(f"Initialized OpenAI TTS engine with voices: {self._voices}")

    def _load_voice_list(self) -> List[str]:
        url = "https://api.openai.com/v1/audio/voices"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        try:
            resp = requests.get(url, headers=headers, timeout=10)
            resp.raise_for_status()
            payload = resp.json()
            entries = payload.get("data") or payload.get("voices") or []
            voices: List[str] = []
            if isinstance(entries, list):
                for item in entries:
                    if isinstance(item, dict):
                        name = item.get("name") or item.get("id")
                        if name:
                            voices.append(str(name))
                    elif isinstance(item, str):
                        voices.append(item)
            voices = sorted({v for v in voices if v})
            if voices:
                logging.info(f"Dynamically loaded {len(voices)} OpenAI voices")
                return voices
        except requests.RequestException as exc:
            logging.warning(f"OpenAI voice list fetch failed: {exc}")
        except Exception as exc:
            logging.warning(f"Unexpected error loading OpenAI voices: {exc}")
        return []

    def list_voices(self) -> List[str]:
        return self._voices

    def synthesize(
        self,
        text: str,
        output_path: str,
        voice: Optional[str] = None,
        instructions: str = "",
    ) -> None:
        v = voice if voice in self._voices else "alloy"
        try:
            response = self.client.audio.speech.create(
                model=self.model,
                voice=v,
                input=text,
                instructions=instructions or "",
                response_format="mp3",
            )
            with open(output_path, "wb") as f:
                f.write(response.content)
        except Exception as e:
            logging.error(f"OpenAI TTS failed for voice {v}: {e}")
            raise RuntimeError(f"OpenAI TTS failed: {e}")

class GoogleCloudTTSEngine(TTSEngine):
    def __init__(self, credentials_path: Optional[str] = None, default_lang: str = "en-US"):
        try:
            from google.cloud import texttospeech
            self.tts_mod = texttospeech
        except ImportError:
            logging.error("google-cloud-texttospeech not installed. Run: pip install google-cloud-texttospeech")
            raise ImportError("google-cloud-texttospeech not installed. Run: pip install google-cloud-texttospeech")
        if credentials_path:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path
        try:
            self.client = self.tts_mod.TextToSpeechClient()
            self.default_lang = default_lang
            self._voices_cache: Optional[List[str]] = None
            self._voice_metadata: Optional[Dict[str, object]] = None
            self._unavailable_voices: Set[str] = set()
            logging.info("Initialized Google Cloud TTS engine")
        except Exception as e:
            logging.error(f"Failed to initialize Google Cloud TTS client: {e}")
            raise RuntimeError(f"Failed to initialize Google Cloud TTS: {e}")

    def list_voices(self) -> List[str]:
        if getattr(self, "_voices_cache", None) is not None:
            return [name for name in self._voices_cache if name not in self._unavailable_voices]
        try:
            resp = self.client.list_voices(request={})
            meta: Dict[str, object] = {}
            names: List[str] = []
            for v in resp.voices:
                name = getattr(v, "name", "")
                if not name:
                    continue
                langs = getattr(v, "language_codes", []) or []
                if not langs:
                    logging.warning(f"Skipping Google voice without language code: {name}")
                    self._unavailable_voices.add(name)
                    continue
                meta[name] = v
                if name not in self._unavailable_voices:
                    names.append(name)
            self._voice_metadata = meta
            self._voices_cache = sorted(names)
            logging.info(f"Loaded {len(self._voices_cache)} Google TTS voices (filtered {len(meta) - len(self._voices_cache)} unavailable)")
            return self._voices_cache
        except Exception as e:
            logging.error(f"Failed to list Google TTS voices: {e}")
            raise RuntimeError(f"Failed to list Google TTS voices: {e}")

    def voice_metadata(self, name: str) -> Optional[object]:
        if self._voice_metadata is None:
            self.list_voices()
        return (self._voice_metadata or {}).get(name)

    def mark_unavailable(self, name: str) -> None:
        if not name:
            return
        if name not in self._unavailable_voices:
            logging.warning(f"Marking Google voice '{name}' as unavailable")
            self._unavailable_voices.add(name)
            if self._voices_cache and name in self._voices_cache:
                self._voices_cache = [v for v in self._voices_cache if v != name]

    def synthesize(
        self,
        text: str,
        output_path: str,
        voice: Optional[str] = None,
        instructions: str = "",
    ) -> None:
        if not text.strip():
            return
        lang = self.default_lang
        valid_voices = self.list_voices()
        selected_voice = voice if voice and voice in valid_voices else f"{lang}-Wavenet-D"
        if voice and voice != selected_voice:
            logging.warning(f"Invalid Google TTS voice '{voice}', falling back to '{selected_voice}'")
        if selected_voice not in valid_voices:
            logging.error(f"Fallback voice '{selected_voice}' not available")
            raise RuntimeError(f"Voice '{selected_voice}' does not exist. Is it misspelled?")
        meta = self.voice_metadata(selected_voice)
        if meta:
            codes = getattr(meta, "language_codes", []) or []
            if codes:
                lang = codes[0]
        voice_params = self.tts_mod.VoiceSelectionParams(language_code=lang, name=selected_voice)
        audio_cfg = self.tts_mod.AudioConfig(audio_encoding=self.tts_mod.AudioEncoding.MP3)
        input_cfg = self.tts_mod.SynthesisInput(text=text)
        try:
            resp = self.client.synthesize_speech(input=input_cfg, voice=voice_params, audio_config=audio_cfg)
            with open(output_path, "wb") as f:
                f.write(resp.audio_content)
        except google_exceptions.InvalidArgument as e:
            if voice:
                self.mark_unavailable(voice)
            logging.error(f"Google TTS synthesis failed for voice {selected_voice}: {e}")
            raise RuntimeError(f"Google TTS synthesis failed: {e}")
        except Exception as e:
            if voice:
                if isinstance(e, google_exceptions.ServiceUnavailable) or isinstance(e, google_exceptions.InternalServerError) or "503" in str(e):
                    self.mark_unavailable(voice)
            logging.error(f"Google TTS synthesis failed for voice {selected_voice}: {e}")
            raise RuntimeError(f"Google TTS synthesis failed: {e}")

class ADRamaApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title(f"{APP_NAME} {APP_VERSION}")
        self.script_path: Optional[str] = None
        self.script: Optional[ScriptData] = None
        self.tts_engines: Dict[str, Optional[TTSEngine]] = {"openai": None, "google": None}
        self.openai_credentials_path: Optional[str] = None
        self.google_credentials_path: Optional[str] = None
        self.voice_map: Dict[str, Tuple[str, str]] = {}
        self.line_records: List[Dict[str, object]] = []
        self.line_metadata: Dict[str, Dict[str, object]] = {}
        self.audio_cache: Dict[str, str] = {}
        self.used_filenames: Set[str] = set()
        self.preview_item_to_index: Dict[str, int] = {}
        self.preview_index_to_item: Dict[int, str] = {}
        self.current_episode_dir: Optional[str] = None
        self.current_manifest_path: Optional[str] = None
        self.current_script_label: Optional[str] = None
        self.outdir: str = DEFAULT_OUTDIR
        self.lines_subdir: str = DEFAULT_LINES_DIRNAME
        self.transform_subdir: str = DEFAULT_TRANSFORM_SUBDIR
        self.dry_run: bool = os.environ.get("ADRAMA_DRY_RUN", "").lower() in {"1", "true", "yes"}
        self.transform_presets = list_presets()
        self.transform_style_var = tk.StringVar(value=DEFAULT_PRESET_ID)
        self.transform_style_display = tk.StringVar()
        self.task_queue = queue.Queue()
        os.makedirs(DEMOS_DIR, exist_ok=True)

        self._load_config()
        self._initialize_engines()
        self._validate_voice_map()
        self._build_menu()
        self._build_main()
        self._task_poll()

    # --- Audio helpers -------------------------------------------------
    def _play_audio(self, path: str) -> None:
        """Fire-and-forget playback using the OS default player."""
        try:
            system = platform.system()
            if system == "Windows":
                os.startfile(path)  # type: ignore[attr-defined]
            elif system == "Darwin":
                subprocess.Popen(["open", path])
            else:
                opener = shutil.which("xdg-open")
                if opener:
                    subprocess.Popen([opener, path])
                else:
                    raise RuntimeError("xdg-open not available on PATH")
        except Exception as exc:
            logging.error(f"Failed to play audio {path}: {exc}")
            messagebox.showerror("Playback Error", f"Could not play audio file:\n{path}\n{exc}", parent=self.root)

    def _set_busy_cursor(self, widget: tk.Misc, busy: bool) -> None:
        cursor = "wait" if os.name == "nt" else "watch"
        try:
            widget.config(cursor=cursor if busy else "")
            widget.update_idletasks()
        except Exception:
            pass

    def play_demo(self, actor: str) -> None:
        assignment = self.voice_map.get(actor)
        if not assignment:
            messagebox.showinfo("No Voice", f"Assign a voice to {actor} first.", parent=self.root)
            return
        eng_name, voice = assignment
        if not voice or voice.startswith("---"):
            messagebox.showinfo("No Voice", f"Assign a valid voice to {actor} first.", parent=self.root)
            return
        eng = self._ensure_engine(eng_name)
        if not eng:
            return

        demo_path = os.path.join(DEMOS_DIR, f"{eng_name}-{sanitize_filename(voice)}.mp3")
        if not os.path.exists(demo_path) or os.path.getsize(demo_path) == 0:
            self.var_status.set(f"Generating demo for {voice}...")
            self._set_busy_cursor(self.root, True)
            try:
                eng.synthesize(SAMPLE_TEXT, demo_path, voice)
                logging.info(f"Generated demo clip for {eng_name}:{voice}")
            except Exception as exc:
                logging.error(f"Demo generation failed for {eng_name}:{voice}: {exc}")
                messagebox.showerror("Demo Error", f"Failed to generate demo:\n{exc}", parent=self.root)
                return
            finally:
                self.var_status.set("Ready.")
                self._set_busy_cursor(self.root, False)
        self._play_audio(demo_path)

    def _synthesize_batch_item(
        self,
        engine_name: str,
        engine: TTSEngine,
        text: str,
        out_path: str,
        voice: str,
        instructions: str = "",
    ) -> None:
        retries = 1
        if engine_name == "google":
            retries = 3
        delay = 1.0
        for attempt in range(retries):
            try:
                engine.synthesize(text=text, output_path=out_path, voice=voice, instructions=instructions)
                return
            except Exception as exc:
                if attempt < retries - 1:
                    logging.warning(f"Retrying {engine_name} synthesis for voice {voice}: {exc}")
                    time.sleep(delay)
                    delay *= 2
                    continue
                raise

    def _load_unavailable_google_voices(self) -> Set[str]:
        try:
            meta_path = os.path.join(DEFAULT_OUTDIR, "google_unavailable.json")
            if os.path.exists(meta_path):
                with open(meta_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        return set(str(v) for v in data)
        except Exception as exc:
            logging.warning(f"Failed to load cached unavailable voices: {exc}")
        return set()

    def _save_unavailable_google_voices(self, voices: Set[str]) -> None:
        try:
            os.makedirs(DEFAULT_OUTDIR, exist_ok=True)
            meta_path = os.path.join(DEFAULT_OUTDIR, "google_unavailable.json")
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(sorted(list(voices)), f, indent=2)
        except Exception as exc:
            logging.warning(f"Failed to persist unavailable voices: {exc}")

    def _initialize_engines(self) -> None:
        unavailable_cache = self._load_unavailable_google_voices()
        if unavailable_cache:
            logging.info(f"Recovered {len(unavailable_cache)} cached unavailable Google voices")
        for eng_name in ["openai", "google"]:
            try:
                self._ensure_engine(eng_name)
                if eng_name == "google":
                    engine = self.tts_engines.get("google")
                    if isinstance(engine, GoogleCloudTTSEngine) and unavailable_cache:
                        engine._unavailable_voices.update(unavailable_cache)
                logging.info(f"Successfully initialized {eng_name} engine")
            except Exception as e:
                logging.warning(f"Failed to initialize {eng_name} engine on startup: {e}")

    def _config_paths(self) -> List[str]:
        return [LOCAL_CONFIG, HOME_CONFIG]

    def _load_config(self) -> None:
        for path in self._config_paths():
            try:
                if os.path.exists(path):
                    with open(path, "r", encoding="utf-8") as f:
                        cfg = json.load(f)
                    self.openai_credentials_path = cfg.get("openai_credentials")
                    self.google_credentials_path = cfg.get("google_credentials")
                    self.outdir = cfg.get("outdir", self.outdir)
                    self.lines_subdir = cfg.get("lines_subdir", self.lines_subdir)
                    self.transform_subdir = cfg.get("transform_subdir", self.transform_subdir)
                    preset_choice = cfg.get("transform_preset")
                    if preset_choice and any(pid == preset_choice for pid, _title, _ in self.transform_presets):
                        self.transform_style_var.set(preset_choice)
                    for actor, data in cfg.get("voice_map", {}).items():
                        if isinstance(data, list) and len(data) == 2:
                            self.voice_map[actor] = tuple(data)
                        else:
                            self.voice_map[actor] = ("google", data)
                    logging.info(f"Loaded config from {path}")
            except Exception as e:
                logging.error(f"Config load failed from {path}: {e}")
                print(f"Config load failed from {path}: {e}", file=sys.stderr)

    def _validate_voice_map(self) -> None:
        invalid = []
        for actor, (eng_name, voice) in list(self.voice_map.items()):
            try:
                eng = self._ensure_engine(eng_name)
                if eng and voice not in eng.list_voices():
                    invalid.append((actor, voice))
            except Exception:
                invalid.append((actor, voice))
        for actor, voice in invalid:
            logging.warning(f"Removing invalid voice assignment: {actor} -> {voice}")
            del self.voice_map[actor]
        if invalid:
            self._save_config()

    def _save_config(self) -> None:
        path = LOCAL_CONFIG if os.access(os.getcwd(), os.W_OK) else HOME_CONFIG
        cfg = {
            "openai_credentials": self.openai_credentials_path or "",
            "google_credentials": self.google_credentials_path or "",
            "outdir": self.outdir,
            "lines_subdir": self.lines_subdir,
            "transform_subdir": self.transform_subdir,
            "transform_preset": self.transform_style_var.get(),
            "voice_map": {k: list(v) for k, v in self.voice_map.items()}
        }
        google_engine = self.tts_engines.get("google")
        if isinstance(google_engine, GoogleCloudTTSEngine):
            cfg["google_unavailable_voices"] = sorted(list(google_engine._unavailable_voices))
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2)
            logging.info(f"Saved config to {path}")
            if "google_unavailable_voices" in cfg:
                self._save_unavailable_google_voices(set(cfg["google_unavailable_voices"]))
        except Exception as e:
            logging.error(f"Config save failed to {path}: {e}")
            print(f"Config save failed to {path}: {e}", file=sys.stderr)

    def _build_menu(self) -> None:
        menubar = tk.Menu(self.root)
        m_file = tk.Menu(menubar, tearoff=0)
        m_file.add_command(label="Open Script...", command=self.on_open_script)
        m_file.add_command(label="Save Script", command=self.on_save_script, accelerator="Ctrl+S")
        m_file.add_command(label="Save Script As...", command=self.on_save_script_as, accelerator="Ctrl+Shift+S")
        m_file.add_separator()
        m_file.add_command(label="Choose Output Folder...", command=self.on_choose_output)
        m_file.add_separator()
        m_file.add_command(label="Exit", command=self.root.quit)
        menubar.add_cascade(label="File", menu=m_file)

        m_voice = tk.Menu(menubar, tearoff=0)
        m_voice.add_command(label="Manage Voices", command=self.open_voice_actor_window, state=tk.DISABLED)
        m_voice.add_command(label="Google Voice Browser...", command=self.open_google_voice_browser)
        menubar.add_cascade(label="Voice-Actor", menu=m_voice)
        self.m_voice = m_voice

        m_line = tk.Menu(menubar, tearoff=0)
        m_line.add_command(label="Edit Selected Line...", command=self.on_edit_selected_line, accelerator="Ctrl+E")
        m_line.add_command(label="Regenerate / Play Selected Line", command=self.on_play_selected_line, accelerator="Ctrl+P")
        menubar.add_cascade(label="Line", menu=m_line)

        m_transform = tk.Menu(menubar, tearoff=0)
        m_transform.add_command(label="Transform Raw Chapter...", command=self.on_transform_chapter)
        menubar.add_cascade(label="Transform", menu=m_transform)

        m_settings = tk.Menu(menubar, tearoff=0)
        m_settings.add_command(label="Manage OpenAI Credentials", command=self._manage_openai_credentials)
        m_settings.add_command(label="Manage Google Credentials", command=self._manage_google_credentials)
        menubar.add_cascade(label="Settings", menu=m_settings)

        m_help = tk.Menu(menubar, tearoff=0)
        m_help.add_command(label="About", command=lambda: messagebox.showinfo(APP_NAME, f"{APP_NAME} {APP_VERSION}\nMulti-engine TTS with voice demos and responsive UI."))
        menubar.add_cascade(label="Help", menu=m_help)

        self.root.config(menu=menubar)
        self.root.bind_all("<Control-s>", lambda event: self.on_save_script())
        self.root.bind_all("<Control-S>", lambda event: self.on_save_script())
        self.root.bind_all("<Control-Shift-S>", lambda event: self.on_save_script_as())
        self.root.bind_all("<Control-e>", lambda event: self.on_edit_selected_line())
        self.root.bind_all("<Control-E>", lambda event: self.on_edit_selected_line())
        self.root.bind_all("<Control-p>", lambda event: self.on_play_selected_line())
        self.root.bind_all("<Control-P>", lambda event: self.on_play_selected_line())

    def _build_main(self) -> None:
        frm = ttk.Frame(self.root, padding=12)
        frm.pack(fill=tk.BOTH, expand=True)

        self.lbl_script = ttk.Label(frm, text="No script loaded.")
        self.lbl_script.pack(anchor="w")

        row = ttk.Frame(frm)
        row.pack(fill="x", pady=(6,2))
        ttk.Button(row, text="Open Script...", command=self.on_open_script).pack(side="left")
        ttk.Button(row, text="Choose Output...", command=self.on_choose_output).pack(side="left", padx=6)
        ttk.Button(row, text="Manage Voices", command=self.open_voice_actor_window).pack(side="left")
        self.btn_generate = ttk.Button(row, text="Generate MP3s + Merge", command=self.on_generate, state=tk.DISABLED)
        self.btn_generate.pack(side="right")

        self._preset_label_to_id: Dict[str, str] = {}
        labels: List[str] = []
        initial_label = ""
        for preset_id, title, _desc in self.transform_presets:
            label = f"{title} ({preset_id})"
            labels.append(label)
            self._preset_label_to_id[label] = preset_id
            if preset_id == self.transform_style_var.get():
                initial_label = label
        if not initial_label and labels:
            initial_label = labels[0]
            self.transform_style_var.set(self._preset_label_to_id.get(initial_label, DEFAULT_PRESET_ID))
        if initial_label:
            self.transform_style_display.set(initial_label)

        style_row = ttk.Frame(frm)
        style_row.pack(fill="x", pady=(4, 0))
        ttk.Label(style_row, text="Transform style:").pack(side="left")
        self.cbo_transform_style = ttk.Combobox(
            style_row,
            state="readonly",
            textvariable=self.transform_style_display,
            values=labels,
            width=36,
        )
        self.cbo_transform_style.pack(side="left", padx=6)
        self.cbo_transform_style.bind("<<ComboboxSelected>>", self._on_transform_style_changed)

        self.lbl_transform_desc = ttk.Label(
            frm,
            text=self._preset_description(self.transform_style_var.get()),
            wraplength=520,
            justify="left",
        )
        self.lbl_transform_desc.pack(fill="x", padx=2, pady=(0, 4))

        columns = ("code", "actor", "text", "voice", "engine", "emotion", "beat", "pace", "instructions")
        tree_container = ttk.Frame(frm)
        tree_container.pack(fill=tk.BOTH, expand=True, pady=(8, 0))
        self.preview_tree = ttk.Treeview(
            tree_container,
            columns=columns,
            show="headings",
            selectmode="browse",
        )
        self.preview_tree.pack(side="left", fill=tk.BOTH, expand=True)
        preview_scroll = ttk.Scrollbar(tree_container, orient="vertical", command=self.preview_tree.yview)
        preview_scroll.pack(side="right", fill="y")
        self.preview_tree.configure(yscrollcommand=preview_scroll.set)

        headings = {
            "code": "Line Code",
            "actor": "Actor",
            "text": "Text",
            "voice": "Voice",
            "engine": "Engine",
            "emotion": "Emotion",
            "beat": "Beat",
            "pace": "Pace",
            "instructions": "Instructions",
        }
        widths = {
            "code": 140,
            "actor": 120,
            "text": 420,
            "voice": 110,
            "engine": 90,
            "emotion": 110,
            "beat": 160,
            "pace": 80,
            "instructions": 300,
        }
        for col in columns:
            self.preview_tree.heading(col, text=headings[col])
            self.preview_tree.column(col, width=widths[col], anchor="w")

        self.preview_tree.bind("<Double-Button-1>", self._on_tree_double_click)
        self.preview_tree.bind("<Return>", self._on_tree_play)
        self.preview_tree.bind("<Control-p>", self._on_tree_play)
        self.preview_tree.bind("<Control-P>", self._on_tree_play)
        self.preview_tree.bind("<Control-e>", self._on_tree_edit)
        self.preview_tree.bind("<F2>", self._on_tree_edit)

        self.progress = ttk.Progressbar(frm, orient="horizontal", length=200, mode="determinate")
        self.progress.pack(fill="x", pady=4)

        status = ttk.Frame(self.root, padding=6)
        status.pack(fill="x")
        self.var_status = tk.StringVar(value="Ready.")
        ttk.Label(status, textvariable=self.var_status).pack(anchor="w")

    def _task_start(self) -> None:
        self.progress['value'] = 0
        self.var_status.set("Starting task...")
        self.btn_generate.config(state=tk.DISABLED)
        self.root.update_idletasks()

    def _task_poll(self) -> None:
        try:
            while True:
                msg = self.task_queue.get_nowait()
                if "progress" in msg:
                    self.progress['value'] = msg["progress"]
                if "status" in msg:
                    self.var_status.set(msg["status"])
                if "error" in msg:
                    messagebox.showerror("Error", msg["error"], parent=self.root)
                    self.btn_generate.config(state=tk.NORMAL)
                if "result" in msg:
                    messagebox.showinfo("Result", msg["result"], parent=self.root)
                    self.btn_generate.config(state=tk.NORMAL)
                if "load_script" in msg:
                    self._reset_generation_context()
                    self.script = ScriptData.from_jsonl(msg["load_script"])
                    self.script_path = msg["load_script"]
                    self._capture_line_records(msg["load_script"])
                    self._update_current_script_label()
                    self.lbl_script.config(text=f"Script: {os.path.basename(msg['load_script'])} ({len(self.script.lines)} lines)")
                    self._populate_preview()
                    self.btn_generate.config(state=tk.NORMAL)
                    self.m_voice.entryconfig("Manage Voices", state=tk.NORMAL)
                self.root.update_idletasks()
        except queue.Empty:
            pass
        self.root.after(100, self._task_poll)

    def _preset_description(self, preset_id: str) -> str:
        for pid, _title, desc in self.transform_presets:
            if pid == preset_id:
                return desc
        return ""

    def _on_transform_style_changed(self, _event: tk.Event) -> None:
        label = self.transform_style_display.get()
        preset_id = self._preset_label_to_id.get(label)
        if preset_id:
            self.transform_style_var.set(preset_id)
            self.lbl_transform_desc.config(text=self._preset_description(preset_id))

    def _manage_openai_credentials(self) -> None:
        path = filedialog.asksaveasfilename(
            title="Select or create OpenAI credentials JSON",
            initialfile="openai_credentials.json",
            defaultextension=".json",
            filetypes=(("JSON files", "*.json"), ("All files", "*.*"))
        )
        if path:
            key = simpledialog.askstring("OpenAI API Key", "Enter your OpenAI API key:", show="*", parent=self.root)
            if key:
                try:
                    with open(path, "w", encoding="utf-8") as f:
                        json.dump({"api_key": key.strip()}, f, indent=2)
                    self.openai_credentials_path = path
                    self.tts_engines["openai"] = None
                    try:
                        self._ensure_engine("openai")
                        logging.info("OpenAI credentials saved and engine reinitialized")
                    except Exception as e:
                        logging.error(f"Failed to reinitialize OpenAI engine: {e}")
                        messagebox.showerror("Error", f"Invalid OpenAI credentials: {e}", parent=self.root)
                    self._save_config()
                    messagebox.showinfo("Saved", f"OpenAI credentials saved to:\n{path}", parent=self.root)
                except Exception as e:
                    logging.error(f"Failed to save OpenAI credentials to {path}: {e}")
                    messagebox.showerror("Error", f"Failed to save OpenAI credentials: {e}", parent=self.root)

    def _manage_google_credentials(self) -> None:
        path = filedialog.askopenfilename(title="Select Google service account JSON", filetypes=(("JSON files", "*.json"), ("All files", "*.*")))
        if path:
            self.google_credentials_path = path
            self.tts_engines["google"] = None
            try:
                self._ensure_engine("google")
                logging.info("Google credentials updated and engine reinitialized")
            except Exception as e:
                logging.error(f"Failed to reinitialize Google engine after credentials update: {e}")
                messagebox.showerror("Error", f"Invalid Google credentials: {e}", parent=self.root)
            self._save_config()
            messagebox.showinfo("Saved", f"Google credentials set:\n{path}", parent=self.root)

    def _ensure_engine(self, name: str) -> Optional[TTSEngine]:
        if self.tts_engines.get(name):
            return self.tts_engines[name]
        try:
            if name == "openai":
                if not self.openai_credentials_path and not os.environ.get("OPENAI_API_KEY"):
                    raise RuntimeError("OpenAI credentials not set. Go to Settings -> Manage OpenAI Credentials.")
                self.tts_engines["openai"] = OpenAITTSEngine(self.openai_credentials_path)
            elif name == "google":
                if not self.google_credentials_path and not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
                    raise RuntimeError("Google credentials not set. Go to Settings -> Manage Google Credentials.")
                self.tts_engines["google"] = GoogleCloudTTSEngine(self.google_credentials_path)
            else:
                raise RuntimeError(f"Unknown engine: {name}")
            return self.tts_engines[name]
        except Exception as e:
            logging.error(f"Failed to initialize {name} engine: {e}")
            messagebox.showerror("Engine Error", f"Failed to initialize {name} engine: {e}", parent=self.root)
            return None

    def _capture_line_records(self, path: str) -> None:
        records: List[Dict[str, object]] = []
        metadata: Dict[str, Dict[str, object]] = {}
        ext = os.path.splitext(path)[1].lower()
        try:
            if ext == ".jsonl":
                with open(path, "r", encoding="utf-8") as fh:
                    for idx, raw in enumerate(fh):
                        raw = raw.strip()
                        if not raw:
                            continue
                        try:
                            obj = json.loads(raw)
                        except json.JSONDecodeError:
                            continue
                        records.append(obj)
                        line_code = str(obj.get("line_code") or f"LINE_{idx + 1:04d}")
                        metadata[line_code] = obj
            elif ext == ".csv":
                with open(path, "r", encoding="utf-8") as fh:
                    reader = csv.DictReader(fh)
                    for idx, row in enumerate(reader):
                        obj = {k: v for k, v in row.items() if v is not None}
                        records.append(obj)
                        line_code = str(obj.get("line_code") or f"LINE_{idx + 1:04d}")
                        metadata[line_code] = obj
        except Exception as exc:
            logging.warning(f"Failed to capture line metadata from {path}: {exc}")
        self.line_records = records
        self.line_metadata = metadata
        if self.script and getattr(self.script, "lines", None):
            for line in self.script.lines:
                if not isinstance(line, ScriptLine):
                    continue
                if line.line_code and line.line_code in self.line_metadata:
                    meta = self.line_metadata[line.line_code]
                    if meta:
                        line.metadata.update(meta)

    def _engine_supports(self, engine_name: str, voice: Optional[str]) -> bool:
        if not voice:
            return False
        engine = self._ensure_engine(engine_name)
        if not engine:
            return False
        try:
            voices = engine.list_voices()
        except Exception as exc:
            logging.warning(f"Failed to query voices for {engine_name}: {exc}")
            return False
        return voice in voices

    def _resolve_voice(self, actor: str, meta: Optional[Dict[str, object]] = None) -> Tuple[str, str, bool]:
        meta = meta or {}
        engine_hint = str(meta.get("voice_engine_hint") or meta.get("voice_engine") or "").lower()
        if engine_hint in {"openai_tts", "openai"}:
            engine_hint = "openai"
        elif engine_hint in {"google_tts", "google"}:
            engine_hint = "google"
        else:
            engine_hint = engine_hint or ""

        primary_voice = str(meta.get("primary_voice") or meta.get("voice_id") or "").strip()
        fallback_voice = str(meta.get("fallback_voice") or "").strip()

        if engine_hint and self._engine_supports(engine_hint, primary_voice):
            self.voice_map[actor] = (engine_hint, primary_voice)
            return engine_hint, primary_voice, False
        if engine_hint and self._engine_supports(engine_hint, fallback_voice):
            self.voice_map[actor] = (engine_hint, fallback_voice)
            return engine_hint, fallback_voice, True

        if actor in self.voice_map:
            eng_name, voice = self.voice_map[actor]
            if self._engine_supports(eng_name, voice):
                return eng_name, voice, False

        if primary_voice:
            guessed_engine = "openai" if engine_hint == "" else engine_hint
            if self._engine_supports(guessed_engine or "openai", primary_voice):
                eng = guessed_engine or "openai"
                self.voice_map[actor] = (eng, primary_voice)
                return eng, primary_voice, False

        if fallback_voice:
            guessed_engine = engine_hint or "openai"
            if self._engine_supports(guessed_engine, fallback_voice):
                self.voice_map[actor] = (guessed_engine, fallback_voice)
                return guessed_engine, fallback_voice, True

        if actor in self.voice_map:
            eng_name, voice = self.voice_map[actor]
            if eng_name and voice and not voice.startswith("---"):
                return eng_name, voice, False

        # Try OpenAI as default
        openai_engine = None
        try:
            openai_engine = self._ensure_engine("openai")
        except Exception:
            openai_engine = None
        if openai_engine:
            voices = openai_engine.list_voices()
            preferred = "alloy" if "alloy" in voices else (voices[0] if voices else "")
            if preferred:
                self.voice_map[actor] = ("openai", preferred)
                return "openai", preferred, bool(primary_voice and preferred != primary_voice)

        # Fallback to Google
        google_engine = None
        try:
            google_engine = self._ensure_engine("google")
        except Exception:
            google_engine = None
        if google_engine:
            voices = google_engine.list_voices()
            if voices:
                default_voice = voices[0]
                self.voice_map[actor] = ("google", default_voice)
                return "google", default_voice, bool(primary_voice and default_voice != primary_voice)

        raise RuntimeError(f"No voice assignment available for actor '{actor}'.")

    _slug_regexp = re.compile(r"[^A-Za-z0-9_.()-]+")

    def _slug(self, value: str, fallback: str = "item") -> str:
        if not value:
            return fallback
        slug = self._slug_regexp.sub("_", value.strip())
        return slug or fallback

    def _metadata_for_line(self, line: ScriptLine, index: int) -> Dict[str, object]:
        meta_from_line = getattr(line, "metadata", None)
        if meta_from_line:
            data = dict(meta_from_line)
        else:
            data = {}
        if self.line_metadata and line.line_code:
            meta = self.line_metadata.get(line.line_code)
            if meta:
                data.update(meta)
        if 0 <= index < len(self.line_records):
            record_meta = self.line_records[index]
            if record_meta:
                data.update(record_meta)
        return data

    def _derive_filename(
        self,
        line: ScriptLine,
        meta: Dict[str, object],
        index: int,
        used: Set[str],
    ) -> str:
        scene_raw = meta.get("scene") or meta.get("scene_label") or "scene"
        beat_raw = meta.get("beat") or meta.get("intent") or f"beat{index:03d}"
        actor_raw = line.actor or meta.get("actor") or "actor"

        scene = self._slug(str(scene_raw), fallback=f"scene{index:03d}")
        beat = self._slug(str(beat_raw), fallback=f"beat{index:03d}")
        actor = self._slug(str(actor_raw), fallback=f"actor{index:03d}")

        base = f"{scene}-{beat}-{actor}.mp3"
        name = base
        counter = 1
        while name in used:
            stem, ext = os.path.splitext(base)
            name = f"{stem}_{counter:02d}{ext}"
            counter += 1
        used.add(name)
        return name

    def _warn_schema_issues(self, records: List[Dict[str, object]]) -> None:
        if not records:
            return
        try:
            import jsonschema
        except ImportError:
            logging.info("Schema validation skipped: jsonschema not installed")
            return

        try:
            schema_path = Path(os.getcwd()) / "adrama_line.schema.json"
            schema = json.loads(schema_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logging.info(f"Schema validation skipped: {exc}")
            return

        try:
            validator = jsonschema.Draft202012Validator(schema)
        except Exception as exc:
            logging.info(f"Schema validator init failed: {exc}")
            return

        for idx, record in enumerate(records):
            try:
                errors = sorted(validator.iter_errors(record), key=lambda e: e.path)
            except Exception as exc:
                logging.info(f"Schema validation execution failed on line {idx}: {exc}")
                continue
            for err in errors:
                path = ".".join(str(p) for p in err.path) or "(root)"
                logging.warning(f"Schema warn line {idx} ({path}): {err.message}")

    def _reset_generation_context(self) -> None:
        self.audio_cache.clear()
        self.used_filenames.clear()
        self.current_episode_dir = None
        self.current_manifest_path = None

    def _update_current_script_label(self) -> None:
        if self.script_path:
            self.current_script_label = os.path.splitext(os.path.basename(self.script_path))[0]
        else:
            self.current_script_label = None

    def on_open_script(self) -> None:
        path = filedialog.askopenfilename(title="Open script", filetypes=(("Script files", "*.txt *.jsonl *.csv"), ("All files", "*.*")))
        if not path:
            return
        try:
            self._reset_generation_context()
            self.script = ScriptData.load_any(path)
            self.script_path = path
            self._capture_line_records(path)
            self._update_current_script_label()
            self.lbl_script.config(text=f"Script: {os.path.basename(path)} ({len(self.script.lines)} lines)")
            self._populate_preview()
            self.btn_generate.config(state=tk.NORMAL)
            self.m_voice.entryconfig("Manage Voices", state=tk.NORMAL)
            logging.info(f"Loaded script: {path}")
        except Exception as e:
            logging.error(f"Script load failed: {e}")
            messagebox.showerror("Load Error", str(e), parent=self.root)


    def _populate_preview(self) -> None:
        for child in self.preview_tree.get_children():
            self.preview_tree.delete(child)
        self.preview_item_to_index = {}
        self.preview_index_to_item = {}
        if not self.script:
            return

        for idx, line in enumerate(self.script.lines):
            if not isinstance(line, ScriptLine):
                continue
            meta = self._metadata_for_line(line, idx)
            code = line.line_code or f"L{idx + 1:04d}"
            voice_id = str(meta.get("voice_id") or meta.get("primary_voice") or "").strip()
            if voice_id:
                line.metadata.setdefault("voice_id", voice_id)
                line.metadata.setdefault("primary_voice", voice_id)
            voice_engine = str(meta.get("voice_engine_hint") or meta.get("voice_engine") or "").strip()
            if voice_engine:
                line.metadata.setdefault("voice_engine_hint", voice_engine)
            emotion = str(meta.get("voice_emotion") or meta.get("emotion") or "").strip()
            beat = str(meta.get("beat") or meta.get("intent") or "").strip()
            pace = str(meta.get("pace") or meta.get("speed") or "").strip()

            instructions = str(meta.get("instructions") or "").strip()
            if not instructions:
                builder_meta = {
                    "voice_style": meta.get("voice_style"),
                    "voice_emotion": emotion,
                    "micro_emotion": meta.get("micro_emotion"),
                    "pace": pace,
                    "vocal_technique": meta.get("vocal_technique"),
                    "accent_hint": meta.get("accent_hint"),
                    "rhythm_signature": meta.get("rhythm_signature"),
                }
                instructions = build_instructions(builder_meta).strip()
            if instructions:
                line.metadata["instructions"] = instructions
                if code:
                    self.line_metadata.setdefault(code, {}).update({"instructions": instructions})

            if voice_id and code:
                self.line_metadata.setdefault(code, {}).update({"voice_id": voice_id, "primary_voice": voice_id})

            values = (
                code,
                line.actor or "Narrator",
                line.text,
                voice_id,
                voice_engine,
                emotion,
                beat,
                pace,
                instructions,
            )
            iid = f"line{idx}"
            self.preview_tree.insert("", "end", iid=iid, values=values)
            self.preview_item_to_index[iid] = idx
            self.preview_index_to_item[idx] = iid

        if self.script and getattr(self.script, "lines", None):
            actors = sorted({
                (line.actor or "Narrator").strip()
                for line in self.script.lines
                if isinstance(line, ScriptLine) and (line.actor or "").strip()
            })
            logging.info("Current actors: %s", ", ".join(actors))


    def _sync_line_from_tree(self, item_id: str) -> Optional[int]:
        idx = self.preview_item_to_index.get(item_id)
        if idx is None or not self.script or idx >= len(self.script.lines):
            return None
        line = self.script.lines[idx]
        values = self.preview_tree.item(item_id, "values")
        if not values:
            return None
        (code, actor, text, voice, engine, emotion, beat, pace, instructions) = values
        old_actor = line.actor
        line.line_code = (code or line.line_code or f"L{idx + 1:04d}").strip()
        line.actor = (actor or "Narrator").strip() or "Narrator"
        line.text = text or ""
        line.metadata.setdefault("actor", line.actor)
        line.metadata.setdefault("line_code", line.line_code)
        line.metadata["text"] = line.text
        if old_actor and old_actor != line.actor:
            self.voice_map.pop(old_actor, None)
        if voice:
            line.metadata["voice_id"] = voice.strip()
            line.metadata["primary_voice"] = voice.strip()
        else:
            line.metadata.pop("voice_id", None)
            line.metadata.pop("primary_voice", None)
        if engine:
            line.metadata["voice_engine_hint"] = engine.strip()
        else:
            line.metadata.pop("voice_engine_hint", None)
        if emotion:
            line.metadata["voice_emotion"] = emotion.strip()
        else:
            line.metadata.pop("voice_emotion", None)
        if beat:
            line.metadata["beat"] = beat.strip()
        else:
            line.metadata.pop("beat", None)
        if pace:
            line.metadata["pace"] = pace.strip()
        else:
            line.metadata.pop("pace", None)
        if instructions:
            line.metadata["instructions"] = instructions.strip()
        else:
            line.metadata.pop("instructions", None)
        if line.line_code:
            stored_meta = self.line_metadata.setdefault(line.line_code, {})
            stored_meta.update(line.metadata)
            stored_meta.pop("_audio_path", None)
            self.audio_cache.pop(line.line_code, None)
        if voice and engine in {"openai", "google"}:
            self.voice_map[line.actor] = (engine, voice)
        line.metadata.pop("_audio_path", None)
        return idx

    def _refresh_tree_row(self, script_idx: int) -> None:
        if not self.script or script_idx >= len(self.script.lines):
            return
        item_id = self.preview_index_to_item.get(script_idx)
        if not item_id:
            return
        line = self.script.lines[script_idx]
        meta = self._metadata_for_line(line, script_idx)
        code = line.line_code or f"L{script_idx + 1:04d}"
        voice_id = str(meta.get("voice_id") or meta.get("primary_voice") or "").strip()
        voice_engine = str(meta.get("voice_engine_hint") or meta.get("voice_engine") or "").strip()
        emotion = str(meta.get("voice_emotion") or meta.get("emotion") or "").strip()
        beat = str(meta.get("beat") or meta.get("intent") or "").strip()
        pace = str(meta.get("pace") or meta.get("speed") or "").strip()
        instructions = str(meta.get("instructions") or "").strip()
        values = (
            code,
            line.actor or "Narrator",
            line.text,
            voice_id,
            voice_engine,
            emotion,
            beat,
            pace,
            instructions,
        )
        self.preview_tree.item(item_id, values=values)

    def _on_tree_double_click(self, event: tk.Event) -> None:  # type: ignore[override]
        item_id = self.preview_tree.identify_row(event.y)
        if not item_id:
            return
        self.preview_tree.selection_set(item_id)
        self.preview_tree.focus(item_id)
        idx = self._sync_line_from_tree(item_id)
        if idx is None:
            return
        threading.Thread(target=self._handle_line_activation, args=(idx,), daemon=True).start()

    def _on_tree_play(self, event: Optional[tk.Event] = None) -> None:
        item_id = self.preview_tree.focus()
        if not item_id:
            return
        idx = self._sync_line_from_tree(item_id)
        if idx is None:
            return
        threading.Thread(target=self._handle_line_activation, args=(idx,), daemon=True).start()

    def _on_tree_edit(self, event: Optional[tk.Event] = None) -> None:
        item_id = self.preview_tree.focus()
        if not item_id:
            return
        self._open_line_editor(item_id)

    def _open_line_editor(self, item_id: str) -> None:
        idx = self.preview_item_to_index.get(item_id)
        if idx is None or not self.script or idx >= len(self.script.lines):
            return
        self._sync_line_from_tree(item_id)
        line = self.script.lines[idx]
        meta = self._metadata_for_line(line, idx)

        editor = tk.Toplevel(self.root)
        editor.title(f"Edit Line {line.line_code or idx + 1}")
        editor.transient(self.root)
        editor.grab_set()

        fields = {
            "Line Code": line.line_code or "",
            "Actor": line.actor or "",
            "Text": line.text or "",
            "Voice": str(meta.get("voice_id") or meta.get("primary_voice") or ""),
            "Engine": str(meta.get("voice_engine_hint") or meta.get("voice_engine") or ""),
            "Emotion": str(meta.get("voice_emotion") or meta.get("emotion") or ""),
            "Beat": str(meta.get("beat") or meta.get("intent") or ""),
            "Pace": str(meta.get("pace") or meta.get("speed") or ""),
        }

        entries: dict[str, tk.Entry] = {}
        for row, (label_text, value) in enumerate(fields.items()):
            ttk.Label(editor, text=label_text).grid(row=row, column=0, sticky="e", padx=6, pady=4)
            entry = ttk.Entry(editor, width=80)
            entry.insert(0, value)
            entry.grid(row=row, column=1, sticky="we", padx=6, pady=4)
            entries[label_text] = entry

        ttk.Label(editor, text="Instructions").grid(row=len(fields), column=0, sticky="ne", padx=6, pady=4)
        instr_text = tk.Text(editor, width=80, height=4, wrap="word")
        instr_text.insert("1.0", meta.get("instructions") or "")
        instr_text.grid(row=len(fields), column=1, sticky="we", padx=6, pady=4)

        button_row = ttk.Frame(editor)
        button_row.grid(row=len(fields)+1, column=0, columnspan=2, pady=(8, 4))

        def save_and_close(play: bool) -> None:
            code = entries["Line Code"].get().strip()
            actor = entries["Actor"].get().strip()
            text = entries["Text"].get()
            voice = entries["Voice"].get().strip()
            engine = entries["Engine"].get().strip()
            emotion = entries["Emotion"].get().strip()
            beat = entries["Beat"].get().strip()
            pace = entries["Pace"].get().strip()
            instructions = instr_text.get("1.0", tk.END).strip()

            new_values = (
                code or line.line_code or f"L{idx + 1:04d}",
                actor or "Narrator",
                text,
                voice,
                engine,
                emotion,
                beat,
                pace,
                instructions,
            )
            self.preview_tree.item(item_id, values=new_values)
            updated_idx = self._sync_line_from_tree(item_id)
            editor.destroy()
            if play and updated_idx is not None:
                threading.Thread(target=self._handle_line_activation, args=(updated_idx,), daemon=True).start()
            elif updated_idx is not None:
                self._refresh_tree_row(updated_idx)

        ttk.Button(button_row, text="Save", command=lambda: save_and_close(False)).pack(side="right", padx=4)
        ttk.Button(button_row, text="Save && Regenerate", command=lambda: save_and_close(True)).pack(side="right", padx=4)
        ttk.Button(button_row, text="Cancel", command=editor.destroy).pack(side="right", padx=4)

        editor.columnconfigure(1, weight=1)
        editor.resizable(False, False)

    def _handle_line_activation(self, script_idx: int) -> None:
        if not self.script or script_idx >= len(self.script.lines):
            return
        line = self.script.lines[script_idx]
        meta = self._metadata_for_line(line, script_idx)
        line_code = line.line_code or f"L{script_idx + 1:04d}"
        audio_path = meta.get("_audio_path") or self.audio_cache.get(line_code)
        if audio_path and os.path.exists(audio_path):
            self.root.after(0, lambda: self._play_audio(audio_path))
            return
        try:
            audio_path = self._generate_single_line(script_idx, line, meta)
            if audio_path:
                if audio_path.lower().endswith(".mp3"):
                    self.root.after(0, lambda: self._play_audio(audio_path))
                else:
                    self.task_queue.put({"status": f"Dry run saved to: {audio_path}"})
        except Exception as exc:
            logging.error("Single-line generation failed", exc_info=True)
            self.task_queue.put({"error": f"Failed to generate line {line.actor}: {exc}"})

    def _generate_single_line(self, script_idx: int, line: ScriptLine, meta: Dict[str, object]) -> Optional[str]:
        text = (line.text or "").strip()
        if not text:
            raise RuntimeError("Line text is empty; edit the line before generating.")

        builder_meta = {
            "voice_style": meta.get("voice_style"),
            "voice_emotion": meta.get("voice_emotion"),
            "micro_emotion": meta.get("micro_emotion"),
            "pace": meta.get("pace") or meta.get("speed"),
            "vocal_technique": meta.get("vocal_technique"),
            "accent_hint": meta.get("accent_hint"),
            "rhythm_signature": meta.get("rhythm_signature"),
        }
        instructions = str(meta.get("instructions") or "").strip()
        if not instructions:
            instructions = build_instructions(builder_meta).strip()
        if instructions:
            line.metadata["instructions"] = instructions

        eng_name, voice, used_fallback = self._resolve_voice(line.actor or "Narrator", meta)
        engine = self._ensure_engine(eng_name)
        if not engine:
            raise RuntimeError(f"Engine '{eng_name}' is unavailable")

        base_label = self.current_script_label or os.path.splitext(os.path.basename(self.script_path or "script"))[0] or "script"
        safe_label = sanitize_filename(base_label) or f"episode_{int(time.time())}"
        base_dir = os.path.join(self.outdir, self.lines_subdir)
        os.makedirs(base_dir, exist_ok=True)
        if not self.current_episode_dir:
            self.current_episode_dir = os.path.join(base_dir, safe_label)
        target_dir = self.current_episode_dir
        os.makedirs(target_dir, exist_ok=True)

        if not self.current_manifest_path:
            self.current_manifest_path = os.path.join(target_dir, "manifest.json")

        manifest_data: Dict[str, object] = {}
        manifest_items: List[Dict[str, object]] = []
        if os.path.exists(self.current_manifest_path):
            try:
                manifest_data = json.loads(Path(self.current_manifest_path).read_text(encoding="utf-8"))
                manifest_items = list(manifest_data.get("items", []))
            except Exception as exc:
                logging.warning(f"Failed to read manifest for single-line generate: {exc}")
                manifest_data = {}
                manifest_items = []
        if not manifest_data:
            manifest_data = {
                "script_label": base_label,
                "generated_at": datetime.utcnow().isoformat() + "Z",
                "dry_run": False,
                "items": manifest_items,
            }

        line_code = line.line_code or f"L{script_idx + 1:04d}"
        manifest_entry = None
        for item in manifest_items:
            if item.get("line_code") == line_code:
                manifest_entry = item
                break
        if manifest_entry is None:
            manifest_entry = {
                "order": script_idx,
                "line_code": line_code,
                "actor": line.actor or "Narrator",
                "scene": meta.get("scene"),
                "beat": meta.get("beat"),
            }
            manifest_items.append(manifest_entry)

        filename = manifest_entry.get("file") or ""
        if not filename:
            filename = self._derive_filename(line, meta, script_idx, self.used_filenames)
        if not filename.lower().endswith(".mp3"):
            filename = os.path.splitext(filename)[0] + ".mp3"
        manifest_entry.update({
            "voice_engine": eng_name,
            "voice_id": voice,
            "fallback_used": used_fallback,
            "instructions": instructions,
            "cached": False,
        })

        if self.dry_run:
            dry_filename = os.path.splitext(filename)[0] + ".txt"
            dry_path = os.path.join(target_dir, dry_filename)
            request_preview = {
                "model": getattr(engine, "model", None),
                "voice": voice,
                "instructions": instructions,
                "text": text,
            }
            Path(dry_path).write_text(
                "[DRY RUN]\n" + json.dumps(request_preview, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            manifest_entry.update({
                "file": dry_filename,
                "status": "dry-run",
            })
            self.used_filenames.add(dry_filename)
            manifest_data["generated_at"] = datetime.utcnow().isoformat() + "Z"
            manifest_data["items"] = manifest_items
            Path(self.current_manifest_path).write_text(json.dumps(manifest_data, indent=2, ensure_ascii=False), encoding="utf-8")
            if line_code:
                entry = self.line_metadata.setdefault(line_code, {})
                entry.update({"instructions": instructions, "voice_id": voice, "primary_voice": voice, "voice_engine_hint": eng_name})
            self._refresh_tree_row(script_idx)
            return dry_path

        manifest_entry.update({
            "file": filename,
            "status": "regenerated",
        })
        self.used_filenames.add(filename)

        out_path = os.path.join(target_dir, filename)
        self.task_queue.put({"status": f"Generating {line.actor}..."})
        self._synthesize_batch_item(eng_name, engine, text, out_path, voice, instructions)

        manifest_data["generated_at"] = datetime.utcnow().isoformat() + "Z"
        manifest_data["items"] = manifest_items
        Path(self.current_manifest_path).write_text(json.dumps(manifest_data, indent=2, ensure_ascii=False), encoding="utf-8")

        line.metadata["voice_id"] = voice
        line.metadata["primary_voice"] = voice
        line.metadata["voice_engine_hint"] = eng_name
        line.metadata["_audio_path"] = out_path
        if line_code:
            entry = self.line_metadata.setdefault(line_code, {})
            entry.update({
                "voice_id": voice,
                "primary_voice": voice,
                "voice_engine_hint": eng_name,
                "instructions": instructions,
                "_audio_path": out_path,
            })
            self.audio_cache[line_code] = out_path

        self._refresh_tree_row(script_idx)
        return out_path

    def on_choose_output(self) -> None:
        d = filedialog.askdirectory(title="Choose output directory")
        if d:
            self.outdir = d
            self._save_config()
            logging.info(f"Output directory set to: {d}")

    def on_save_script(self) -> None:
        if not self.script:
            messagebox.showinfo("No Script", "Open or transform a script first.", parent=self.root)
            return
        default_path = None
        if self.script_path and self.script_path.lower().endswith(".jsonl"):
            default_path = self.script_path
        if not default_path:
            self.on_save_script_as()
            return
        try:
            self._write_script_jsonl(default_path)
            self.var_status.set(f"Saved script to {default_path}")
            logging.info(f"Saved script to {default_path}")
        except Exception as exc:
            logging.error(f"Failed to save script: {exc}")
            messagebox.showerror("Save Error", f"Failed to save script:\n{exc}", parent=self.root)

    def on_save_script_as(self) -> None:
        if not self.script:
            messagebox.showinfo("No Script", "Open or transform a script first.", parent=self.root)
            return
        path = filedialog.asksaveasfilename(
            title="Save script as",
            defaultextension=".jsonl",
            filetypes=(("JSON Lines", "*.jsonl"), ("All files", "*.*")),
            initialfile=os.path.splitext(os.path.basename(self.script_path or "script"))[0] + ".jsonl",
        )
        if not path:
            return
        if not path.lower().endswith(".jsonl"):
            path += ".jsonl"
        try:
            self._write_script_jsonl(path)
        except Exception as exc:
            logging.error(f"Failed to save script: {exc}")
            messagebox.showerror("Save Error", f"Failed to save script:\n{exc}", parent=self.root)
            return
        self.script_path = path
        self._update_current_script_label()
        self.lbl_script.config(text=f"Script: {os.path.basename(path)} ({len(self.script.lines)} lines)")
        self._save_config()
        self.var_status.set(f"Saved script to {path}")
        logging.info(f"Saved script to {path}")

    def _write_script_jsonl(self, path: str) -> None:
        if not self.script:
            raise RuntimeError("No script loaded")
        with open(path, "w", encoding="utf-8") as fh:
            for idx, line in enumerate(self.script.lines):
                payload = dict(getattr(line, "metadata", {}) or {})
                payload["actor"] = line.actor or "Narrator"
                payload["text"] = line.text or ""
                payload["line_code"] = line.line_code or f"L{idx + 1:04d}"
                fh.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def on_edit_selected_line(self) -> None:
        item_id = self.preview_tree.focus()
        if not item_id:
            messagebox.showinfo("No Selection", "Select a line in the table first.", parent=self.root)
            return
        self._open_line_editor(item_id)

    def on_play_selected_line(self) -> None:
        item_id = self.preview_tree.focus()
        if not item_id:
            messagebox.showinfo("No Selection", "Select a line in the table first.", parent=self.root)
            return
        idx = self._sync_line_from_tree(item_id)
        if idx is None:
            return
        threading.Thread(target=self._handle_line_activation, args=(idx,), daemon=True).start()

    def on_generate(self) -> None:
        if not self.script:
            messagebox.showinfo("No Script", "Open or transform a script first.", parent=self.root)
            return
        if not self.script.lines:
            messagebox.showinfo("Empty Script", "The current script has no lines to generate.", parent=self.root)
            return

        try:
            os.makedirs(self.outdir, exist_ok=True)
        except OSError as exc:
            logging.error(f"Failed to prepare output directory {self.outdir}: {exc}")
            messagebox.showerror("Output Error", f"Could not create output directory:\n{self.outdir}\n{exc}", parent=self.root)
            return

        label = os.path.splitext(os.path.basename(self.script_path or "script"))[0] or "script"

        self._reset_generation_context()
        self.current_script_label = label

        snapshot = [
            ScriptLine(
                actor=ln.actor,
                text=ln.text,
                line_code=ln.line_code,
                metadata=dict(getattr(ln, "metadata", {}) or {}),
            )
            for ln in self.script.lines
        ]

        self._task_start()
        threading.Thread(target=self._do_generate_work, args=(snapshot, label), daemon=True).start()

    def _do_generate_work(self, lines: List[ScriptLine], script_label: str) -> None:
        try:
            total = len(lines)
            safe_label = sanitize_filename(script_label) or f"episode_{int(time.time())}"
            base_dir = os.path.join(self.outdir, self.lines_subdir)
            os.makedirs(base_dir, exist_ok=True)
            target_dir = os.path.join(base_dir, safe_label)
            os.makedirs(target_dir, exist_ok=True)

            self.task_queue.put({"progress": 5, "status": f"Preparing {total} lines for generation..."})

            generated_files: List[Tuple[int, str, str, str]] = []
            tasks: List[Dict[str, object]] = []
            manifest_items: List[Dict[str, object]] = []
            schema_records: List[Dict[str, object]] = []
            used_filenames: Set[str] = set()
            completed = 0

            for order_idx, line in enumerate(lines):
                text = (line.text or "").strip()
                if not text:
                    logging.info(f"Skipping empty line {line.line_code or order_idx + 1}")
                    continue

                meta = self._metadata_for_line(line, order_idx)
                line_id = line.line_code or f"L{order_idx + 1:04d}"

                schema_record = dict(meta)
                schema_record.setdefault("actor", line.actor or "Narrator")
                schema_record.setdefault("text", text)
                schema_record.setdefault("line_code", line_id)
                schema_records.append(schema_record)

                builder_meta = {
                    "voice_style": meta.get("voice_style") or "",
                    "voice_emotion": meta.get("voice_emotion") or "",
                    "micro_emotion": meta.get("micro_emotion") or "",
                    "pace": meta.get("pace") or meta.get("speed") or "",
                    "vocal_technique": meta.get("vocal_technique") or "",
                    "accent_hint": meta.get("accent_hint") or "",
                    "rhythm_signature": meta.get("rhythm_signature") or "",
                }
                instructions = str(meta.get("instructions") or "").strip()
                if not instructions:
                    instructions = build_instructions(builder_meta)
                instructions = instructions.strip()

                try:
                    eng_name, voice, used_fallback = self._resolve_voice(line.actor or "Narrator", meta)
                    engine = self._ensure_engine(eng_name)
                    if not engine:
                        raise RuntimeError(f"Engine '{eng_name}' not available")
                except Exception as exc:
                    self.task_queue.put({"error": f"Voice setup failed for {line.actor}: {exc}"})
                    return

                filename = self._derive_filename(line, meta, order_idx, used_filenames)
                out_path = os.path.join(target_dir, filename)

                manifest_item = {
                    "order": order_idx,
                    "line_code": line_id,
                    "actor": line.actor or "Narrator",
                    "scene": meta.get("scene"),
                    "beat": meta.get("beat"),
                    "voice_engine": eng_name,
                    "voice_id": voice,
                    "fallback_used": used_fallback,
                    "instructions": instructions,
                    "cached": False,
                    "status": "pending",
                    "file": os.path.basename(out_path),
                }
                manifest_items.append(manifest_item)

                if self.dry_run:
                    dry_path = os.path.splitext(out_path)[0] + ".txt"
                    request_preview = {
                        "model": getattr(engine, "model", None),
                        "voice": voice,
                        "instructions": instructions,
                        "text": text,
                    }
                    Path(dry_path).write_text(
                        "[DRY RUN]\n" + json.dumps(request_preview, indent=2, ensure_ascii=False),
                        encoding="utf-8",
                    )
                    manifest_item["file"] = os.path.basename(dry_path)
                    manifest_item["status"] = "dry-run"
                    completed += 1
                    progress = min(90, int(5 + completed / max(total, 1) * 80))
                    self.task_queue.put({
                        "progress": progress,
                        "status": f"Dry-run {completed}/{total}: {line.actor}"
                    })
                    continue

                if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
                    logging.info(f"Skipping cached line {line_id} ({line.actor})")
                    manifest_item["cached"] = True
                    manifest_item["status"] = "cached"
                    generated_files.append((order_idx, out_path, line_id, line.actor or "Narrator"))
                    completed += 1
                    progress = min(90, int(5 + completed / max(total, 1) * 80))
                    self.task_queue.put({
                        "progress": progress,
                        "status": f"Cached {completed}/{total}: {line.actor}"
                    })
                    continue

                tasks.append({
                    "order": order_idx,
                    "engine": engine,
                    "engine_name": eng_name,
                    "voice": voice,
                    "instructions": instructions,
                    "text": text,
                    "out_path": out_path,
                    "actor": line.actor or "Narrator",
                    "line_id": line_id,
                    "manifest_item": manifest_item,
                })

            self._warn_schema_issues(schema_records)

            if self.dry_run:
                manifest = {
                    "script_label": script_label,
                    "generated_at": datetime.utcnow().isoformat() + "Z",
                    "dry_run": True,
                    "items": manifest_items,
                }
                manifest_path = os.path.join(target_dir, "manifest.json")
                Path(manifest_path).write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
                self.task_queue.put({
                    "progress": 100,
                    "status": "Dry run complete.",
                    "result": f"Dry run saved to:\n{target_dir}\nManifest:\n{manifest_path}"
                })
                return

            if tasks:
                groups: Dict[str, List[Dict[str, object]]] = {}
                for task in tasks:
                    groups.setdefault(task["engine_name"], []).append(task)

                for engine_name, group in groups.items():
                    workers = min(
                        ENGINE_WORKER_LIMITS.get(engine_name, MAX_TTS_WORKERS),
                        len(group)
                    )
                    self.task_queue.put({"status": f"Generating {engine_name} audio with {workers} worker(s)..."})
                    with ThreadPoolExecutor(max_workers=workers) as executor:
                        future_map = {
                            executor.submit(
                                self._synthesize_batch_item,
                                engine_name,
                                task["engine"],
                                task["text"],
                                task["out_path"],
                                task["voice"],
                                task["instructions"],
                            ): task
                            for task in group
                        }

                        for future in as_completed(future_map):
                            task = future_map[future]
                            order_idx = task["order"]
                            actor = task["actor"]
                            out_path = task["out_path"]
                            manifest_item = task["manifest_item"]
                            try:
                                future.result()
                                manifest_item["status"] = "generated"
                                generated_files.append((order_idx, out_path, task["line_id"], actor))
                                completed += 1
                                logging.info(f"Generated line {task['line_id']} ({actor}) -> {out_path}")
                                progress = min(90, int(5 + completed / max(total, 1) * 80))
                                self.task_queue.put({
                                    "progress": progress,
                                    "status": f"Generated {completed}/{total}: {actor}"
                                })
                            except Exception as exc:
                                logging.error(f"Synthesis failed for {task['line_id']}: {exc}")
                                try:
                                    if os.path.exists(out_path):
                                        os.remove(out_path)
                                except OSError:
                                    pass
                                manifest_item["status"] = "failed"
                                self.task_queue.put({"error": f"Failed to generate line {actor or task['line_id']}: {exc}"})
                                for fut in future_map:
                                    fut.cancel()
                                return

            if not generated_files:
                self.task_queue.put({"error": "No audio was generated."})
                return

            manifest = {
                "script_label": script_label,
                "generated_at": datetime.utcnow().isoformat() + "Z",
                "dry_run": False,
                "items": manifest_items,
            }
            manifest_path = os.path.join(target_dir, "manifest.json")
            Path(manifest_path).write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

            self.current_episode_dir = target_dir
            self.current_manifest_path = manifest_path
            self.used_filenames = {item.get("file") for item in manifest_items if item.get("file")}
            for item in manifest_items:
                line_code = item.get("line_code")
                file_name = item.get("file")
                if not line_code or not file_name:
                    continue
                audio_path = os.path.join(target_dir, file_name)
                if os.path.exists(audio_path) and file_name.lower().endswith(".mp3"):
                    self.audio_cache[line_code] = audio_path
                    self.line_metadata.setdefault(line_code, {}).update({"_audio_path": audio_path})
                    if self.script:
                        for script_line in self.script.lines:
                            if getattr(script_line, "line_code", None) == line_code:
                                script_line.metadata.setdefault("_audio_path", audio_path)
                                break

            merge_path = os.path.join(target_dir, DEFAULT_MERGE_FILENAME)
            timeline_entries: List[Dict[str, object]] = []
            try:
                merged = AudioSegment.silent(duration=0)
                current_ms = 0
                for _, path, line_id, actor in sorted(generated_files, key=lambda item: item[0]):
                    segment = AudioSegment.from_file(path)
                    duration_ms = len(segment)
                    merged += segment
                    timeline_entries.append({
                        "line_code": line_id,
                        "actor": actor,
                        "file": os.path.relpath(path, target_dir),
                        "start_ms": current_ms,
                        "duration_ms": duration_ms
                    })
                    current_ms += duration_ms
                merged.export(merge_path, format="mp3")
                logging.info(f"Merged episode written to {merge_path}")
            except Exception as exc:
                logging.error(f"Failed to merge audio: {exc}")
                self.task_queue.put({"error": f"Failed to merge audio files: {exc}"})
                return

            timeline_csv = os.path.join(target_dir, "timeline.csv")
            timeline_json = os.path.join(target_dir, "timeline.json")
            try:
                with open(timeline_csv, "w", newline="", encoding="utf-8") as f_csv:
                    writer = csv.writer(f_csv)
                    writer.writerow(["line_code", "actor", "file", "start_ms", "duration_ms"])
                    for entry in timeline_entries:
                        writer.writerow([
                            entry["line_code"],
                            entry["actor"],
                            entry["file"],
                            entry["start_ms"],
                            entry["duration_ms"]
                        ])
                with open(timeline_json, "w", encoding="utf-8") as f_json:
                    json.dump(timeline_entries, f_json, indent=2)
                logging.info(f"Wrote timeline metadata to {timeline_csv} and {timeline_json}")
            except Exception as exc:
                logging.warning(f"Failed to write timeline metadata: {exc}")

            try:
                bundle_name = f"bundle_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
                bundle_path = os.path.join(target_dir, bundle_name)
                with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                    for _, path, line_id, _ in sorted(generated_files, key=lambda item: item[0]):
                        arcname = os.path.join("lines", os.path.basename(path))
                        zf.write(path, arcname=arcname)
                    zf.write(merge_path, arcname=DEFAULT_MERGE_FILENAME)
                    zf.write(manifest_path, arcname=os.path.basename(manifest_path))
                    if os.path.exists(timeline_csv):
                        zf.write(timeline_csv, arcname=os.path.basename(timeline_csv))
                    if os.path.exists(timeline_json):
                        zf.write(timeline_json, arcname=os.path.basename(timeline_json))
                logging.info(f"Bundle archive created at {bundle_path}")
            except Exception as exc:
                logging.warning(f"Failed to create bundle archive: {exc}")

            try:
                self._save_config()
            except Exception as exc:
                logging.warning(f"Config save failed after generation: {exc}")

            self.task_queue.put({
                "progress": 100,
                "status": "Generation complete.",
                "result": (
                    f"Generated {len(generated_files)} line MP3s in:\n{target_dir}\n"
                    f"Merged episode:\n{merge_path}\nManifest:\n{manifest_path}"
                )
            })
        except Exception as exc:
            logging.error(f"Unexpected error in generation worker: {exc}")
            self.task_queue.put({"error": f"Unexpected error: {exc}"})

    def open_google_voice_browser(self) -> None:
        eng = self._ensure_engine("google")
        if not eng or not isinstance(eng, GoogleCloudTTSEngine):
            logging.error("Google Voice Browser: Failed to initialize Google TTS engine")
            messagebox.showerror("Google TTS Error", "Could not initialize Google TTS engine. Check credentials in Settings.", parent=self.root)
            return
        try:
            response = eng.client.list_voices(request={})
            voice_objs = list(response.voices)
        except Exception as exc:
            logging.error(f"Google Voice Browser failed to fetch metadata: {exc}")
            messagebox.showerror("Google TTS Error", f"Could not fetch voice metadata: {exc}", parent=self.root)
            return

        def voice_kind(voice) -> str:
            name = getattr(voice, "name", "")
            m = re.match(r"([a-z]{2}-[A-Z]{2})-(Standard|Wavenet|Neural2|Studio|Journey|Gemini)-([A-Za-z])", name)
            if m:
                return m.group(2)
            parts = name.split("-")
            return parts[1] if len(parts) > 1 else "Unknown"

        voice_map: Dict[str, object] = {voice.name: voice for voice in voice_objs if getattr(voice, "name", None)}
        unavailable = getattr(eng, "_unavailable_voices", set())

        parsed = []
        for voice in voice_objs:
            name = getattr(voice, "name", "")
            if not name:
                continue
            if name in unavailable:
                continue
            lang_code = voice.language_codes[0] if getattr(voice, "language_codes", None) else "unknown"
            lang_label = format_language_label(lang_code)
            kind = voice_kind(voice)
            parsed.append((name, lang_label, kind, lang_code))

        if not parsed:
            messagebox.showinfo("No Voices", "No Google voices returned.", parent=self.root)
            return

        top = tk.Toplevel(self.root)
        top.title("Google Voice Browser")
        frm = ttk.Frame(top, padding=8)
        frm.pack(fill="both", expand=True)

        all_langs = sorted({lang_label for (_, lang_label, _kind, _code) in parsed})
        present_kinds = sorted({kind for (_, _lang_label, kind, _code) in parsed})
        all_kinds = ["Any", *present_kinds]

        default_lang_label = format_language_label("en-US")
        lang_var = tk.StringVar(value=default_lang_label if default_lang_label in all_langs else (all_langs[0] if all_langs else ""))
        kind_var = tk.StringVar(value="Any")
        show_legacy = tk.BooleanVar(value=False)

        def filtered_names() -> List[str]:
            L, K = lang_var.get(), kind_var.get()
            return [name for (name, lang_label, kind, _code) in parsed if (lang_label == L) and (K == "Any" or kind == K)]

        row0 = ttk.Frame(frm)
        row0.pack(fill="x", pady=(0, 6))
        ttk.Label(row0, text="Language").pack(side="left")
        cb_lang = ttk.Combobox(row0, values=all_langs, textvariable=lang_var, width=16, state="readonly")
        cb_lang.pack(side="left", padx=(6, 12))
        ttk.Label(row0, text="Type").pack(side="left")
        cb_kind = ttk.Combobox(row0, values=all_kinds, textvariable=kind_var, width=12, state="readonly")
        cb_kind.pack(side="left", padx=6)

        listbox = tk.Listbox(frm, height=12)
        listbox.pack(fill="both", expand=True, pady=6)

        def refresh_list(*_) -> None:
            current_selection = None
            sel = listbox.curselection()
            if sel:
                current_selection = listbox.get(sel[0])
            listbox.delete(0, tk.END)
            target_index = None
            for name in filtered_names():
                if name in getattr(eng, "_unavailable_voices", set()):
                    continue
                kind = next((k for (n, _lang_label, k, _code) in parsed if n == name), "Unknown")
                if not show_legacy.get() and kind not in GOOGLE_PREMIUM_KINDS and kind != "Unknown":
                    continue
                idx = listbox.size()
                listbox.insert(tk.END, name)
                if current_selection and name == current_selection:
                    target_index = idx
            if target_index is not None:
                listbox.selection_set(target_index)
                listbox.see(target_index)
            else:
                listbox.selection_clear(0, tk.END)
            update_details()

        cb_lang.bind("<<ComboboxSelected>>", refresh_list)
        cb_kind.bind("<<ComboboxSelected>>", refresh_list)

        ttk.Checkbutton(frm, text="Show legacy voices", variable=show_legacy, command=refresh_list).pack(anchor="w", pady=(0, 4))

        details_var = tk.StringVar(value="Select a voice to view details.")
        ttk.Label(frm, textvariable=details_var, justify="left").pack(fill="x", pady=(0, 4))

        def describe_voice(name: str) -> str:
            voice = voice_map.get(name)
            if not voice:
                return name
            langs = ", ".join(format_language_label(code) for code in (getattr(voice, "language_codes", []) or [])) or "unknown"
            kind = voice_kind(voice)
            gender_attr = getattr(voice, "ssml_gender", None)
            if hasattr(gender_attr, "name"):
                gender = gender_attr.name.replace("_", " ").title()
            elif isinstance(gender_attr, str):
                gender = gender_attr.title()
            else:
                gender = "Unspecified"
            sample_rate = getattr(voice, "natural_sample_rate_hertz", None)
            rate_str = f"{sample_rate / 1000:.1f} kHz" if sample_rate else "n/a"
            return (f"Voice: {name}\n"
                    f"Language(s): {langs}\n"
                    f"Type: {kind}\n"
                    f"Gender: {gender}\n"
                    f"Sample rate: {rate_str}")

        def update_details(*_) -> None:
            sel = listbox.curselection()
            if not sel:
                details_var.set("Select a voice to view details.")
                return
            name = listbox.get(sel[0])
            details_var.set(describe_voice(name))

        refresh_list()

        def demo_selected_voice() -> None:
            sel = listbox.curselection()
            if not sel:
                messagebox.showinfo("Pick Voice", "Select a voice from the list.", parent=top)
                return
            vname = listbox.get(sel[0])
            demo_path = os.path.join(DEMOS_DIR, f"google-{sanitize_filename(vname)}.mp3")
            if os.path.exists(demo_path) and os.path.getsize(demo_path) > 0:
                self._play_audio(demo_path)
                logging.info(f"Played cached demo for Google voice: {vname}")
                return
            self.var_status.set(f"Generating demo for {vname}...")
            self._set_busy_cursor(top, True)
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(eng.synthesize, SAMPLE_TEXT, demo_path, vname)
                try:
                    future.result(timeout=10)
                    self._play_audio(demo_path)
                    logging.info(f"Generated demo for Google voice: {vname}")
                except TimeoutError:
                    logging.error(f"Demo timeout for Google voice {vname}")
                    messagebox.showerror("Demo Error", f"Demo timed out for voice: {vname}", parent=top)
                except Exception as e:
                    logging.error(f"Demo failed for Google voice {vname}: {e}")
                    messagebox.showerror("Demo Error", f"Failed to demo voice: {e}", parent=top)
                    refresh_list()
                finally:
                    self.var_status.set("Ready.")
                    self._set_busy_cursor(top, False)

        listbox.bind("<<ListboxSelect>>", update_details)
        ttk.Button(row0, text="Demo Selected", command=demo_selected_voice).pack(side="left", padx=6)

        actors = sorted({ln.actor for ln in self.script.lines}) if self.script else []
        grid = ttk.Frame(frm); grid.pack(fill="x", pady=(8,4))
        ttk.Label(grid, text="Assign selected voice to actor (Google engine):").grid(row=0, column=0, columnspan=2, sticky="w")
        actor_var = tk.StringVar(value=actors[0] if actors else "")
        cb_actor = ttk.Combobox(grid, values=actors, textvariable=actor_var, width=24); cb_actor.grid(row=1, column=0, sticky="w", padx=(0,8), pady=4)

        def assign_selected():
            if not actors:
                messagebox.showinfo("No Script", "Open/transform a script first.", parent=top); return
            sel = listbox.curselection()
            if not sel:
                messagebox.showinfo("Pick Voice", "Select a voice from the list.", parent=top); return
            vname = listbox.get(sel[0]); actor = actor_var.get()
            self.voice_map[actor] = ("google", vname)
            self._save_config()
            assigned_text.delete("1.0", tk.END); assigned_text.insert(tk.END, "Current Assignments:\n")
            for a in actors:
                if a in self.voice_map:
                    eng_name, voice = self.voice_map[a]
                    assigned_text.insert(tk.END, f"{a}: {eng_name} - {voice}\n")
            logging.info(f"Assigned Google voice {vname} to {actor}")
            messagebox.showinfo("Assigned", f"{actor} -> Google: {vname}", parent=top)

        ttk.Button(grid, text="Assign", command=assign_selected).grid(row=1, column=1, sticky="w", pady=4)

        assigned_text = tk.Text(frm, height=5, wrap="word"); assigned_text.pack(fill="x", pady=4)
        assigned_text.insert(tk.END, "Current Assignments:\n")
        for actor in actors:
            if actor in self.voice_map:
                eng_name, voice = self.voice_map[actor]
                assigned_text.insert(tk.END, f"{actor}: {eng_name} - {voice}\n")
        assigned_text.config(state=tk.DISABLED)

        ttk.Button(frm, text="Close", command=top.destroy).pack(anchor="e", pady=(6,4))


    def open_voice_actor_window(self) -> None:
        if not self.script:
            messagebox.showinfo("No Script", "Open or transform a script first.", parent=self.root)
            return

        actors = sorted({ln.actor for ln in self.script.lines})
        top = tk.Toplevel(self.root)
        top.title("Assign Actor Voices")
        frm = ttk.Frame(top, padding=8)
        frm.pack(fill="both", expand=True)
        ttk.Label(frm, text="Assign engine and voice to each actor (click actor name to play demo):").pack(anchor="w", pady=(0, 6))

        canvas = tk.Canvas(frm, borderwidth=0, highlightthickness=0)
        scrollbar = ttk.Scrollbar(frm, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        engine_choices = ["openai", "google"]
        combos: Dict[str, Tuple[tk.StringVar, ttk.Combobox, tk.StringVar, ttk.Combobox]] = {}

        assigned_text = tk.Text(frm, height=6, wrap="word")
        assigned_text.pack(fill="x", pady=(8, 4))

        def refresh_assignments() -> None:
            assigned_text.config(state=tk.NORMAL)
            assigned_text.delete("1.0", tk.END)
            assigned_text.insert(tk.END, "Current Assignments:\n")
            for actor_name in actors:
                if actor_name in self.voice_map:
                    eng_name, voice_name = self.voice_map[actor_name]
                    assigned_text.insert(tk.END, f"{actor_name}: {eng_name} - {voice_name}\n")
            assigned_text.config(state=tk.DISABLED)

        def get_grouped_voices(eng_name: str, voices: List[str]) -> List[str]:
            if not voices:
                return ["No voices available"]
            if eng_name == "google":
                google_engine = self._ensure_engine("google")
                if not google_engine:
                    return ["No voices available"]
                premium_groups: Dict[str, List[str]] = {}
                fallback_groups: Dict[str, List[str]] = {}
                unavailable = getattr(google_engine, "_unavailable_voices", set())
                for voice_name in voices:
                    if voice_name in unavailable:
                        continue
                    meta = google_engine.voice_metadata(voice_name)
                    kind = "Unknown"
                    lang_code = "unknown"
                    if meta:
                        lang_codes = getattr(meta, "language_codes", []) or []
                        if lang_codes:
                            lang_code = lang_codes[0]
                        name = getattr(meta, "name", voice_name)
                        m = re.match(r"([a-z]{2}-[A-Z]{2})-(Standard|Wavenet|Neural2|Studio|Journey|Gemini)-([A-Za-z])", name)
                        if m:
                            kind = m.group(2)
                        else:
                            parts = name.split("-")
                            if len(parts) > 1:
                                kind = parts[1]
                    label = format_language_label(lang_code)
                    target = premium_groups if kind in GOOGLE_PREMIUM_KINDS else fallback_groups
                    target.setdefault(label, []).append(voice_name)

                lang_groups = premium_groups if premium_groups else fallback_groups
                if not lang_groups:
                    return ["No voices available"]
                grouped: List[str] = []
                for label in sorted(lang_groups.keys()):
                    grouped.append(f"--- {label} ---")
                    grouped.extend(sorted(lang_groups[label]))
                return grouped
            return ["--- OpenAI Voices ---"] + voices

        for idx, actor in enumerate(actors):
            actor_button = ttk.Button(scrollable_frame, text=actor, command=lambda a=actor: self.play_demo(a))
            actor_button.grid(row=idx, column=0, sticky="w", padx=(0, 8), pady=4)

            default_engine, default_voice = self.voice_map.get(actor, ("google", ""))
            engine_var = tk.StringVar(value=default_engine or "google")
            cb_engine = ttk.Combobox(scrollable_frame, values=engine_choices, textvariable=engine_var, width=12, state="readonly")
            cb_engine.grid(row=idx, column=1, sticky="w", pady=4)

            voice_var = tk.StringVar(value=default_voice)
            cb_voice = ttk.Combobox(scrollable_frame, textvariable=voice_var, width=32, state="readonly")
            cb_voice.grid(row=idx, column=2, sticky="ew", pady=4)

            combos[actor] = (engine_var, cb_engine, voice_var, cb_voice)

            def update_voices(*_, actor_local=actor, eng_var_local=engine_var, voice_var_local=voice_var, cb_voice_local=cb_voice):
                eng_name_local = eng_var_local.get()
                engine_obj = self._ensure_engine(eng_name_local)
                voices_available = engine_obj.list_voices() if engine_obj else []
                cb_voice_local['values'] = get_grouped_voices(eng_name_local, voices_available)

                current_voice = voice_var_local.get()
                if current_voice not in voices_available:
                    default_choice = voices_available[0] if voices_available else ""
                    if default_choice:
                        voice_var_local.set(default_choice)
                        self.voice_map[actor_local] = (eng_name_local, default_choice)
                    else:
                        voice_var_local.set("No voices available")
                        self.voice_map.pop(actor_local, None)
                        refresh_assignments()
                        return
                else:
                    self.voice_map[actor_local] = (eng_name_local, current_voice)
                refresh_assignments()

            def demo_on_select(*_, actor_local=actor, voice_var_local=voice_var, eng_var_local=engine_var):
                selection = voice_var_local.get()
                if not selection or selection.startswith("---") or selection == "No voices available":
                    return
                eng_name_local = eng_var_local.get()
                self.voice_map[actor_local] = (eng_name_local, selection)
                refresh_assignments()
                self.play_demo(actor_local)

            cb_engine.bind("<<ComboboxSelected>>", update_voices)
            cb_voice.bind("<<ComboboxSelected>>", demo_on_select)
            update_voices()

        scrollable_frame.columnconfigure(2, weight=1)

        refresh_assignments()

        google_engine = self._ensure_engine("google")
        google_voices = google_engine.list_voices() if google_engine else []
        parsed = []
        for name in google_voices:
            m = re.match(r"([a-z]{2}-[A-Z]{2})-(Standard|Wavenet|Neural2|Studio|Journey|Gemini)-([A-Za-z])", name)
            lang = m.group(1) if m else (name.split("-")[0] if "-" in name else "unknown")
            kind = m.group(2) if m else "Unknown"
            parsed.append((name, lang, kind))

        all_langs = sorted({lang for _, lang, _ in parsed})
        all_kinds = ["Any", "Standard", "Wavenet", "Neural2", "Studio", "Journey", "Gemini"]

        lang_var = tk.StringVar(value="en-US" if "en-US" in all_langs else (all_langs[0] if all_langs else ""))
        kind_var = tk.StringVar(value="Any")

        def filtered_names() -> List[str]:
            lang_val, kind_val = lang_var.get(), kind_var.get()
            return [name for (name, lang, kind) in parsed if (lang == lang_val) and (kind_val == "Any" or kind == kind_val)]

        row0 = ttk.Frame(frm)
        row0.pack(fill="x", pady=(0, 6))
        ttk.Label(row0, text="Language").pack(side="left")
        cb_lang = ttk.Combobox(row0, values=all_langs, textvariable=lang_var, width=16, state="readonly")
        cb_lang.pack(side="left", padx=(6, 12))
        ttk.Label(row0, text="Type").pack(side="left")
        cb_kind = ttk.Combobox(row0, values=all_kinds, textvariable=kind_var, width=12, state="readonly")
        cb_kind.pack(side="left", padx=6)

        listbox = tk.Listbox(frm, height=12)
        listbox.pack(fill="both", expand=True, pady=6)

        def refresh_list(*_) -> None:
            listbox.delete(0, tk.END)
            for voice_name in filtered_names():
                listbox.insert(tk.END, voice_name)

        cb_lang.bind("<<ComboboxSelected>>", refresh_list)
        cb_kind.bind("<<ComboboxSelected>>", refresh_list)
        refresh_list()

        def demo_selected_voice() -> None:
            selection = listbox.curselection()
            if not selection:
                messagebox.showinfo("Pick Voice", "Select a voice from the list.", parent=top)
                return
            vname = listbox.get(selection[0])
            demo_path = os.path.join(DEMOS_DIR, f"google-{sanitize_filename(vname)}.mp3")
            if os.path.exists(demo_path) and os.path.getsize(demo_path) > 0:
                self._play_audio(demo_path)
                logging.info(f"Played cached demo for Google voice: {vname}")
                return
            self.var_status.set(f"Generating demo for {vname}...")
            self._set_busy_cursor(top, True)
            if not google_engine:
                messagebox.showerror("Google TTS Error", "Google engine not available.", parent=top)
            else:
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(google_engine.synthesize, SAMPLE_TEXT, demo_path, vname)
                    try:
                        future.result(timeout=10)
                        self._play_audio(demo_path)
                        logging.info(f"Generated demo for Google voice: {vname}")
                    except TimeoutError:
                        logging.error(f"Demo timeout for Google voice {vname}")
                        messagebox.showerror("Demo Error", f"Demo timed out for voice: {vname}", parent=top)
                    except Exception as exc:
                        logging.error(f"Demo failed for Google voice {vname}: {exc}")
                        messagebox.showerror("Demo Error", f"Failed to demo voice: {exc}", parent=top)
            self.var_status.set("Ready.")
            self._set_busy_cursor(top, False)

        ttk.Button(row0, text="Demo Selected", command=demo_selected_voice).pack(side="left", padx=6)

        grid = ttk.Frame(frm)
        grid.pack(fill="x", pady=(8, 4))
        ttk.Label(grid, text="Assign selected voice to actor (Google engine):").grid(row=0, column=0, columnspan=2, sticky="w")
        actor_var = tk.StringVar(value=actors[0] if actors else "")
        cb_actor = ttk.Combobox(grid, values=actors, textvariable=actor_var, width=24, state="readonly")
        cb_actor.grid(row=1, column=0, sticky="w", padx=(0, 8), pady=4)

        def assign_selected() -> None:
            if not actors:
                messagebox.showinfo("No Script", "Open/transform a script first.", parent=top)
                return
            selection = listbox.curselection()
            if not selection:
                messagebox.showinfo("Pick Voice", "Select a voice from the list.", parent=top)
                return
            vname = listbox.get(selection[0])
            actor_choice = actor_var.get()
            if not actor_choice:
                messagebox.showinfo("Pick Actor", "Choose an actor to assign this voice.", parent=top)
                return
            self.voice_map[actor_choice] = ("google", vname)
            self._save_config()
            refresh_assignments()
            logging.info(f"Assigned Google voice {vname} to {actor_choice}")
            messagebox.showinfo("Assigned", f"{actor_choice} -> Google: {vname}", parent=top)

        ttk.Button(grid, text="Assign", command=assign_selected).grid(row=1, column=1, sticky="w", pady=4)

        def save_assignments() -> None:
            missing = []
            for actor_name, (eng_var, _cb_eng, voice_var, _cb_voice) in combos.items():
                eng_val = (eng_var.get() or "").strip()
                voice_val = (voice_var.get() or "").strip()
                if not eng_val or not voice_val or voice_val.startswith("---") or voice_val == "No voices available":
                    missing.append(actor_name)
                else:
                    self.voice_map[actor_name] = (eng_val, voice_val)
            if missing:
                messagebox.showwarning("Missing Voices", "Assign voices for: " + ", ".join(missing), parent=top)
                refresh_assignments()
                return
            self._save_config()
            refresh_assignments()
            messagebox.showinfo("Saved", "Voice assignments saved.", parent=top)

        btn_row = ttk.Frame(frm)
        btn_row.pack(fill="x", pady=(6, 4))
        ttk.Button(btn_row, text="Save", command=save_assignments).pack(side="right")
        ttk.Button(btn_row, text="Close", command=top.destroy).pack(side="right", padx=6)

    def on_transform_chapter(self) -> None:
        src = filedialog.askopenfilename(title="Select raw chapter (.txt)", filetypes=[("Text files", "*.txt")])
        if not src:
            return
        try:
            with open(src, "r", encoding="utf-8") as f:
                raw_text = f.read()
            if not raw_text.strip():
                messagebox.showinfo("Empty File", "The selected file is empty.", parent=self.root); return
        except IOError as e:
            logging.error(f"Transform chapter read failed: {e}")
            messagebox.showerror("Read Error", f"Could not read file: {e}", parent=self.root); return

        actor_list = simpledialog.askstring("Actors", "Enter comma-separated actor names (or leave blank for defaults):", parent=self.root)
        actors = [a.strip() for a in actor_list.split(",") if a.strip()] if actor_list else ["Narrator", "Alice", "Bob", "Cara", "Dylan"]

        base_label = os.path.splitext(os.path.basename(src))[0]

        self._task_start()
        threading.Thread(
            target=self._do_transform_work,
            args=(raw_text, actors, base_label),
            daemon=True,
        ).start()

    def _do_transform_work(self, raw_text: str, actors: List[str], label: Optional[str]) -> None:
        clean_label = sanitize_filename(label or "") or None
        request = TransformRequest(
            raw_text=raw_text,
            actors=actors or ["Narrator"],
            outdir=self.outdir,
            transform_subdir=self.transform_subdir,
            voice_map=self.voice_map.copy(),
            openai_credentials_path=self.openai_credentials_path,
            script_label=clean_label,
            preset_id=self.transform_style_var.get() or DEFAULT_PRESET_ID,
        )

        def forward_status(msg: Dict[str, object]) -> None:
            self.task_queue.put(msg)

        try:
            result = run_transform_job(request, status_callback=forward_status)
        except TransformError as exc:
            logging.error("Transform failed: %s", exc)
            self.task_queue.put({"error": f"Transform failed: {exc}"})
            return
        except Exception as exc:  # pragma: no cover - defensive guard
            logging.error("Unexpected error in transform worker", exc_info=True)
            self.task_queue.put({"error": f"Unexpected transform error: {exc}"})
            return

        self.task_queue.put({"load_script": result.jsonl_path})
        self.task_queue.put({
            "progress": 100,
            "status": f"Transform complete ({result.preset_id}).",
            "result": f"Transformed and loaded script ({result.preset_id}):\n{result.jsonl_path}\n{result.csv_path}"
        })


def main() -> None:
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        logging.error(f"Failed to initialize Tkinter: {exc}")
        raise
    ADRamaApp(root)
    try:
        root.mainloop()
    except KeyboardInterrupt:
        logging.info("GUI interrupted by user")


if __name__ == "__main__":
    main()
