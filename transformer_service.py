"""Standalone transform service and CLI for ADRama presets."""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from adrama_core import sanitize_filename


try:  # pragma: no cover - runtime dependency
    from openai import OpenAI
except ImportError:  # pragma: no cover - handled at runtime when CLI invoked
    OpenAI = None  # type: ignore

try:  # pragma: no cover - optional validation dependency
    import jsonschema
except ImportError:  # pragma: no cover - handled gracefully
    jsonschema = None  # type: ignore


logger = logging.getLogger(__name__)

StatusMessage = Dict[str, object]
StatusCallback = Callable[[StatusMessage], None]

DEFAULT_PRESET_ID = "audio_drama"
SCHEMA_DIR = os.path.join(os.path.dirname(__file__), "schemas")


@dataclass
class TransformPreset:
    preset_id: str
    title: str
    description: str
    system_blocks: List[str]
    output_fields: List[str]
    default_field_values: Dict[str, object]
    line_budget_range: Tuple[int, int]
    schema_filename: Optional[str] = None
    preset_token: Optional[str] = None


@dataclass(frozen=True)
class VoiceCapability:
    voice_id: str
    label: str
    timbre: str
    energy: str
    best_for: str
    instruction_defaults: Dict[str, str]

    def prompt_summary(self) -> str:
        parts = [f"{self.label} – {self.timbre}"]
        if self.energy:
            parts.append(self.energy)
        if self.best_for:
            parts.append(f"Best for {self.best_for}")
        return "; ".join(parts)

    def instruction_hint(self) -> str:
        bits = []
        for key in [
            "voice_style",
            "voice_emotion",
            "micro_emotion",
            "pace",
            "vocal_technique",
            "accent_hint",
            "rhythm_signature",
        ]:
            value = self.instruction_defaults.get(key)
            if value:
                label = key.replace("_", " ")
                bits.append(f"{label}={value}")
        return ", ".join(bits)


@dataclass
class VoiceAssignment:
    engine: str
    voice_id: str
    source: str
    capability: Optional[VoiceCapability] = None

    def as_hint(self) -> str:
        if self.voice_id:
            return f"{self.engine}:{self.voice_id}"
        return self.engine


OPENAI_VOICE_CAPABILITIES: Dict[str, VoiceCapability] = {
    "alloy": VoiceCapability(
        voice_id="alloy",
        label="Alloy",
        timbre="warm baritone",
        energy="grounded, reassuring presence",
        best_for="narrators, mentors, steady leaders",
        instruction_defaults={
            "voice_style": "warm mentor",
            "voice_emotion": "steady",
            "micro_emotion": "assured",
            "pace": "measured",
            "vocal_technique": "rich low resonance, supportive breaths",
        },
    ),
    "aria": VoiceCapability(
        voice_id="aria",
        label="Aria",
        timbre="bright mezzo",
        energy="uplifted, quicksilver optimism",
        best_for="youthful heroes, excited guides",
        instruction_defaults={
            "voice_style": "bright protagonist",
            "voice_emotion": "hopeful",
            "micro_emotion": "uplifted",
            "pace": "quick",
            "vocal_technique": "open vowels, crisp articulation",
        },
    ),
    "ash": VoiceCapability(
        voice_id="ash",
        label="Ash",
        timbre="airy contralto",
        energy="soft, intimate confidant",
        best_for="tender moments, reflective allies",
        instruction_defaults={
            "voice_style": "soft confidant",
            "voice_emotion": "calm",
            "micro_emotion": "empathetic",
            "pace": "measured",
            "vocal_technique": "breathy warmth, gentle consonants",
        },
    ),
    "coral": VoiceCapability(
        voice_id="coral",
        label="Coral",
        timbre="clear soprano",
        energy="resolute, luminous conviction",
        best_for="idealists, inspiring leaders",
        instruction_defaults={
            "voice_style": "calm authority",
            "voice_emotion": "hopeful",
            "micro_emotion": "steadfast",
            "pace": "measured",
            "vocal_technique": "firm yet gentle emphasis",
        },
    ),
    "ember": VoiceCapability(
        voice_id="ember",
        label="Ember",
        timbre="smoky alto",
        energy="urgent, gritty defiance",
        best_for="rebels, tense confrontations",
        instruction_defaults={
            "voice_style": "smoldering rebel",
            "voice_emotion": "intense",
            "micro_emotion": "driven",
            "pace": "urgent",
            "vocal_technique": "tight projection, rasp edge",
        },
    ),
    "fable": VoiceCapability(
        voice_id="fable",
        label="Fable",
        timbre="storyteller tenor",
        energy="warm, imaginative narrator",
        best_for="omniscient narration, lore drops",
        instruction_defaults={
            "voice_style": "storyteller narrator",
            "voice_emotion": "reassuring",
            "micro_emotion": "wistful",
            "pace": "steady",
            "vocal_technique": "rounded diction, narrative lift",
        },
    ),
    "gilded": VoiceCapability(
        voice_id="gilded",
        label="Gilded",
        timbre="polished mezzo",
        energy="poised, articulate elegance",
        best_for="courtiers, strategists, eloquent hosts",
        instruction_defaults={
            "voice_style": "regal poise",
            "voice_emotion": "composed",
            "micro_emotion": "controlled",
            "pace": "measured",
            "vocal_technique": "crystal diction, upper presence",
            "accent_hint": "subtle transatlantic",
        },
    ),
    "onyx": VoiceCapability(
        voice_id="onyx",
        label="Onyx",
        timbre="deep bass",
        energy="stoic, commanding gravitas",
        best_for="captains, antagonists, solemn vows",
        instruction_defaults={
            "voice_style": "commanding bass",
            "voice_emotion": "determined",
            "micro_emotion": "steel",
            "pace": "slow",
            "vocal_technique": "low resonance, clipped consonants",
        },
    ),
    "sage": VoiceCapability(
        voice_id="sage",
        label="Sage",
        timbre="gentle tenor",
        energy="patient, thoughtful guidance",
        best_for="mentors, healers, exposition",
        instruction_defaults={
            "voice_style": "gentle guide",
            "voice_emotion": "calm",
            "micro_emotion": "encouraging",
            "pace": "steady",
            "vocal_technique": "clear pedagogy, patient phrasing",
        },
    ),
    "thunder": VoiceCapability(
        voice_id="thunder",
        label="Thunder",
        timbre="booming baritone",
        energy="dramatic, high-impact force",
        best_for="announcers, battle cries, SFX cues",
        instruction_defaults={
            "voice_style": "dramatic announcer",
            "voice_emotion": "commanding",
            "micro_emotion": "thundering",
            "pace": "urgent",
            "vocal_technique": "power chest, rolled consonants",
        },
    ),
    "verse": VoiceCapability(
        voice_id="verse",
        label="Verse",
        timbre="lyrical soprano",
        energy="dreamlike, creative flow",
        best_for="poets, visionaries, musical beats",
        instruction_defaults={
            "voice_style": "lyrical poet",
            "voice_emotion": "inspired",
            "micro_emotion": "dreamy",
            "pace": "measured",
            "vocal_technique": "musical cadence, flowing phrasing",
            "rhythm_signature": "6/8 lilt",
        },
    ),
}


OPENAI_TTS_VOICE_ORDER: List[str] = [
    "alloy",
    "aria",
    "ash",
    "coral",
    "ember",
    "fable",
    "gilded",
    "onyx",
    "sage",
    "thunder",
    "verse",
]


VOICE_SPECIAL_CASES: Dict[str, str] = {
    "narrator": "alloy",
    "the narrator": "alloy",
    "sfx": "thunder",
    "sound fx": "thunder",
    "music": "verse",
    "chorus": "verse",
}


def _normalise_actor_token(name: str) -> str:
    name = (name or "").strip().lower()
    return re.sub(r"\s+", " ", name)


def _lookup_voice_capability(engine: str, voice_id: str) -> Optional[VoiceCapability]:
    if not voice_id:
        return None
    if engine == "openai":
        return OPENAI_VOICE_CAPABILITIES.get(voice_id)
    return None


def _auto_select_openai_voice(actor_token: str) -> str:
    if not actor_token:
        return OPENAI_TTS_VOICE_ORDER[0]
    if actor_token in VOICE_SPECIAL_CASES:
        return VOICE_SPECIAL_CASES[actor_token]
    for key, voice in VOICE_SPECIAL_CASES.items():
        if key in actor_token:
            return voice
    digest = hashlib.sha256(actor_token.encode("utf-8")).hexdigest()
    idx = int(digest[:8], 16) % len(OPENAI_TTS_VOICE_ORDER)
    return OPENAI_TTS_VOICE_ORDER[idx]


def _normalise_voice_map(raw_map: Dict[str, Sequence[str] | Tuple[str, str]]) -> Dict[str, Tuple[str, str]]:
    result: Dict[str, Tuple[str, str]] = {}
    if not raw_map:
        return result
    for key, value in raw_map.items():
        if value is None:
            continue
        engine: str = ""
        voice: str = ""
        if isinstance(value, (list, tuple)):
            if len(value) >= 2:
                engine = str(value[0] or "").strip()
                voice = str(value[1] or "").strip()
        if not engine or not voice or voice.startswith("---"):
            continue
        token = _normalise_actor_token(key)
        if token:
            result[token] = (engine, voice)
    return result


def _coerce_str(value: object) -> str:
    if isinstance(value, str):
        return value.strip()
    if value is None:
        return ""
    return str(value).strip()


INSTRUCTION_FIELD_TEMPLATE: Tuple[Tuple[str, str], ...] = (
    ("voice_style", "{}"),
    ("voice_emotion", "emotion: {}"),
    ("micro_emotion", "micro: {}"),
    ("pace", "pace: {}"),
    ("vocal_technique", "technique: {}"),
    ("accent_hint", "accent: {}"),
    ("rhythm_signature", "rhythm: {}"),
)


def _build_instruction_string(meta: Dict[str, str]) -> str:
    parts: List[str] = []
    for field, template in INSTRUCTION_FIELD_TEMPLATE:
        value = _coerce_str(meta.get(field))
        if not value:
            continue
        if template == "{}":
            parts.append(value)
        else:
            parts.append(template.format(value))
    if not parts:
        return ""
    instruction = "; ".join(parts)
    if not instruction.endswith("."):
        instruction += "."
    return instruction


def _apply_voice_metadata(record: Dict[str, object], assignment: Optional[VoiceAssignment]) -> None:
    capability: Optional[VoiceCapability] = assignment.capability if assignment else None

    if assignment:
        record["voice_engine_hint"] = assignment.engine
        record["primary_voice"] = assignment.voice_id
        record["voice"] = assignment.voice_id
        if assignment.engine == "openai":
            record["voice_engine"] = "openai_tts"
            record["voice_id"] = assignment.voice_id
        else:
            record.pop("voice_engine", None)
            record["voice_id"] = assignment.voice_id
    else:
        record.setdefault("voice_engine_hint", "")
        record.setdefault("primary_voice", "")
        record.pop("voice_engine", None)
        record.pop("voice_id", None)

    defaults = capability.instruction_defaults if capability else {}
    for field, value in defaults.items():
        if not value:
            continue
        if field == "pace":
            if not _coerce_str(record.get("speed")):
                record["speed"] = value
            if not _coerce_str(record.get("pace")):
                record["pace"] = value
            continue
        if not _coerce_str(record.get(field)):
            record[field] = value

    pace_value = _coerce_str(record.get("speed")) or _coerce_str(record.get("pace"))
    meta = {
        "voice_style": _coerce_str(record.get("voice_style")),
        "voice_emotion": _coerce_str(record.get("voice_emotion")),
        "micro_emotion": _coerce_str(record.get("micro_emotion")),
        "pace": pace_value,
        "vocal_technique": _coerce_str(record.get("vocal_technique")),
        "accent_hint": _coerce_str(record.get("accent_hint")),
        "rhythm_signature": _coerce_str(record.get("rhythm_signature")),
    }
    instructions = _build_instruction_string(meta)
    if instructions:
        record["instructions"] = instructions


class PresetManager:
    def __init__(self, presets: Iterable[TransformPreset]):
        self._presets: Dict[str, TransformPreset] = {p.preset_id: p for p in presets}
        if DEFAULT_PRESET_ID not in self._presets:
            raise ValueError("DEFAULT_PRESET_ID must be present in presets")
        self._schema_cache: Dict[str, Dict[str, object]] = {}

    def list_presets(self) -> List[TransformPreset]:
        return list(self._presets.values())

    def get(self, preset_id: Optional[str]) -> TransformPreset:
        if not preset_id:
            return self._presets[DEFAULT_PRESET_ID]
        return self._presets.get(preset_id, self._presets[DEFAULT_PRESET_ID])

    def compute_line_budget(self, preset: TransformPreset, chunk_length: int) -> int:
        low, high = preset.line_budget_range
        heuristic = max(10, min(36, (chunk_length // 70) + 8))
        return max(low, min(high, heuristic))

    def build_system_prompt(self, preset: TransformPreset, actor_list: List[str], voice_guidance: str) -> str:
        base_prompt = BASE_PROMPT_TEMPLATE.format(
            actor_list=", ".join(actor_list),
            voice_guidance_text=voice_guidance or "No additional voice guidance.".strip(),
        )
        blocks: List[str] = []
        if preset.preset_token:
            blocks.append(f"[PRESET: {preset.preset_token}]")
        blocks.extend(preset.system_blocks)
        if blocks:
            base_prompt = base_prompt + "\n" + "\n".join(blocks)
        return base_prompt

    def load_schema(self, preset: TransformPreset) -> Optional[Dict[str, object]]:
        if not preset.schema_filename:
            return None
        if preset.schema_filename in self._schema_cache:
            return self._schema_cache[preset.schema_filename]
        schema_path = os.path.join(SCHEMA_DIR, preset.schema_filename)
        if not os.path.exists(schema_path):
            logger.warning("Schema file %s not found; skipping validation", schema_path)
            self._schema_cache[preset.schema_filename] = None  # type: ignore
            return None
        with open(schema_path, "r", encoding="utf-8") as fh:
            schema = json.load(fh)
        self._schema_cache[preset.schema_filename] = schema
        return schema

    def validate_response(self, preset: TransformPreset, data: Dict[str, object]) -> None:
        if jsonschema is None:
            return
        schema = self.load_schema(preset)
        if not schema:
            return
        try:
            jsonschema.validate(instance=data, schema=schema)
        except jsonschema.ValidationError as exc:  # pragma: no cover - runtime validation
            raise TransformError(f"Preset '{preset.preset_id}' schema validation failed: {exc.message}") from exc

    def build_record(
        self,
        preset: TransformPreset,
        base_fields: Dict[str, object],
        model_item: Dict[str, object],
        assignment: Optional[VoiceAssignment],
    ) -> Dict[str, object]:
        record = {field: preset.default_field_values.get(field, "") for field in preset.output_fields}
        for key, value in base_fields.items():
            if key in record:
                record[key] = value
        for field in preset.output_fields:
            if field in base_fields:
                continue
            if field in model_item:
                raw_value = model_item[field]
                record[field] = _normalise_model_value(raw_value)
        return record


def _normalise_model_value(value: object) -> object:
    if isinstance(value, (str, type(None))):
        return value or ""
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, (list, dict)):
        try:
            return json.dumps(value, ensure_ascii=False)
        except TypeError:
            return str(value)
    return str(value)


BASE_FIELDS = [
    "line_code",
    "scene",
    "actor",
    "voice",
    "voice_engine",
    "voice_id",
    "text",
    "actions",
    "expression",
    "gaze_to",
    "sfx",
    "notes",
    "speed",
    "voice_style",
    "voice_emotion",
    "beat",
    "instructions",
]

AUDIO_DRAMA_FIELDS = BASE_FIELDS + [
    "music_bed",
    "sfx_layer",
    "silence_beats",
    "micro_emotion",
    "vocal_technique",
    "accent_hint",
    "rhythm_signature",
    "voice_engine_hint",
    "primary_voice",
    "fallback_voice",
]

FAITHFUL_AUDIOBOOK_FIELDS = BASE_FIELDS + [
    "narration_style",
    "micro_emotion",
    "vocal_technique",
    "accent_hint",
    "voice_engine_hint",
    "primary_voice",
    "fallback_voice",
]

CINEMATIC_SCREENPLAY_FIELDS = BASE_FIELDS + [
    "scene_label",
    "camera",
    "shot_type",
    "lighting",
    "prop_notes",
    "mood_board",
    "frame_prompt",
    "camera_move",
    "color_palette",
    "voice_engine_hint",
    "primary_voice",
    "fallback_voice",
]

INTERACTIVE_SIM_FIELDS = BASE_FIELDS + [
    "npc_state",
    "player_prompt",
    "branch_token",
    "world_state",
    "micro_emotion",
    "vocal_technique",
    "voice_engine_hint",
    "primary_voice",
    "fallback_voice",
]

STORYBOARD_FIELDS = BASE_FIELDS + [
    "frame_prompt",
    "mood_board",
    "camera_move",
    "lighting",
    "color_palette",
    "voice_engine_hint",
    "primary_voice",
    "fallback_voice",
]

BASE_DEFAULTS = {field: "" for field in set(AUDIO_DRAMA_FIELDS + FAITHFUL_AUDIOBOOK_FIELDS + CINEMATIC_SCREENPLAY_FIELDS + INTERACTIVE_SIM_FIELDS + STORYBOARD_FIELDS)}

PRESET_DEFINITIONS: List[TransformPreset] = [
    TransformPreset(
        preset_id="audio_drama",
        title="Audio Drama",
        description="Cinematic pacing with rich sound design cues and strong voice direction.",
        system_blocks=[
            "Use strong SFX cues and dialogue-forward pacing.",
            "Include optional fields: music_bed, sfx_layer, silence_beats, micro_emotion, vocal_technique, accent_hint, rhythm_signature, voice_engine_hint, primary_voice, fallback_voice.",
            "If a requested field is unsupported by the target voice engine, leave it null.",
        ],
        output_fields=AUDIO_DRAMA_FIELDS,
        default_field_values=BASE_DEFAULTS,
        line_budget_range=(18, 32),
        schema_filename="base_transform.schema.json",
        preset_token="Audio Drama",
    ),
    TransformPreset(
        preset_id="audiobook_faithful",
        title="Faithful Audiobook",
        description="Narration-focused adaptation that mirrors the source closely.",
        system_blocks=[
            "Preserve authorial voice and narrative order exactly.",
            "Add subtle performance cues (micro_emotion, vocal_technique) but avoid dramatizing beyond the text.",
        ],
        output_fields=FAITHFUL_AUDIOBOOK_FIELDS,
        default_field_values=BASE_DEFAULTS,
        line_budget_range=(16, 24),
        schema_filename="base_transform.schema.json",
        preset_token="Audiobook",
    ),
    TransformPreset(
        preset_id="cinematic_screenplay",
        title="Cinematic Screenplay",
        description="Scene-first blueprint with camera, lighting, and prop cues.",
        system_blocks=[
            "Structure lines to reflect screenplay beats with implied sluglines and camera guidance.",
            "Populate camera, shot_type, lighting, prop_notes, mood_board, frame_prompt when useful.",
        ],
        output_fields=CINEMATIC_SCREENPLAY_FIELDS,
        default_field_values=BASE_DEFAULTS,
        line_budget_range=(16, 26),
        schema_filename="base_transform.schema.json",
        preset_token="Screenplay",
    ),
    TransformPreset(
        preset_id="interactive_sim",
        title="Interactive Simulation",
        description="Branch-ready output with NPC states and player hooks.",
        system_blocks=[
            "Provide npc_state, player_prompt, branch_token, and world_state cues to support branching narrative systems.",
            "Keep instructions concise for downstream game logic.",
        ],
        output_fields=INTERACTIVE_SIM_FIELDS,
        default_field_values=BASE_DEFAULTS,
        line_budget_range=(12, 20),
        schema_filename="base_transform.schema.json",
        preset_token="Interactive",
    ),
    TransformPreset(
        preset_id="storyboard",
        title="Storyboard & Mood",
        description="Visual-first beats with frame prompts and color direction.",
        system_blocks=[
            "Focus on visual descriptions and frame prompts suitable for storyboards or image generation.",
            "Fill frame_prompt, mood_board, camera_move, lighting, and color_palette where helpful.",
        ],
        output_fields=STORYBOARD_FIELDS,
        default_field_values=BASE_DEFAULTS,
        line_budget_range=(10, 18),
        schema_filename="base_transform.schema.json",
        preset_token="Storyboard",
    ),
]


PRESET_MANAGER = PresetManager(PRESET_DEFINITIONS)


BASE_PROMPT_TEMPLATE = (
    "You are ADRama's dramatization and blocking engine for a rehearsal table read.\n"
    "Actor roster (rotate naturally, keep voices balanced): {actor_list}.\n"
    "Current voice intelligence hints:\n{voice_guidance_text}\n"
    "For every chunk of narrative you receive, do ALL of the following:\n"
    "1. Cover every plot beat in the text—do not skip minor actions or dialogue.\n"
    "2. Split narration into short, speakable lines (max ~35 words) with clear performer assignments.\n"
    "3. Maintain continuity across chunks using the provided previous summary and next_scene start.\n"
    "4. For each line emit fields: actor, text, scene (int), actions, expression, gaze_to, sfx, notes, voice_style, voice_emotion, beat, pace, voice_engine_hint, primary_voice, fallback_voice, instructions.\n"
    "   - actions/expression/gaze_to/sfx should be short tokens or phrases.\n"
    "   - voice_style: 2-4 word direction that pairs the assigned voice with delivery guidance.\n"
    "   - voice_emotion: single dominant emotion keyword (e.g., 'hopeful', 'tense').\n"
    "   - beat: one short phrase describing the dramatic intention of the line.\n"
    "   - pace: describe delivery speed with one word (e.g., 'measured', 'urgent').\n"
    "   - voice_engine_hint & primary_voice must honor the provided voice assignments; leave fallback_voice blank unless you have a strong alternate.\n"
    "   - instructions (if provided) should be terse natural-language delivery cues. The renderer will augment blanks automatically.\n"
    "5. Return valid JSON with keys: lines (array) and continuity (object with summary, next_scene, pending).\n"
    "6. Honor the provided line_budget value—never return more than that number of entries in lines. Summarize overflow in continuity.pending for the next chunk.\n"
    "Keep tone PG-rated, compress exposition, and retain canonical terminology from the source.\n"
    "Do not wrap the JSON in markdown fences or add commentary—respond with JSON only."
)


@dataclass
class TransformRequest:
    raw_text: str
    actors: List[str]
    outdir: str
    transform_subdir: str
    voice_map: Dict[str, Tuple[str, str]]
    openai_credentials_path: Optional[str] = None
    script_label: Optional[str] = None
    preset_id: str = DEFAULT_PRESET_ID


@dataclass
class TransformResult:
    jsonl_path: str
    csv_path: str
    records: List[Dict[str, object]]
    preset_id: str


class TransformError(RuntimeError):
    """Raised when the transform pipeline fails irrecoverably."""


def list_presets() -> List[Tuple[str, str, str]]:
    """Return available presets as tuples (id, title, description)."""

    return [(p.preset_id, p.title, p.description) for p in PRESET_MANAGER.list_presets()]


def _default_status_callback(_: StatusMessage) -> None:
    return


def _load_openai_key(credentials_path: Optional[str]) -> str:
    if credentials_path and os.path.exists(credentials_path):
        try:
            with open(credentials_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            key = cfg.get("api_key")
            if key:
                return key
            logger.warning("No api_key present in %s", credentials_path)
        except Exception as exc:  # pragma: no cover - runtime protection
            logger.error("Failed to load OpenAI credentials from %s: %s", credentials_path, exc)
    key = os.environ.get("OPENAI_API_KEY")
    if key:
        return key
    raise TransformError("No valid OpenAI API key found")


def _chunk_text(text: str, max_chars: int = 800) -> List[str]:
    pieces: List[str] = []
    buffer: List[str] = []
    length = 0
    for block in re.split(r"\n\s*\n", text.strip()):
        segment = block.strip()
        if not segment:
            continue
        seg_len = len(segment)
        if seg_len > max_chars:
            if buffer:
                pieces.append("\n\n".join(buffer))
                buffer = []
                length = 0
            for start in range(0, seg_len, max_chars):
                chunk_part = segment[start:start + max_chars].strip()
                if chunk_part:
                    pieces.append(chunk_part)
            continue
        if buffer and (length + seg_len) > max_chars:
            pieces.append("\n\n".join(buffer))
            buffer = [segment]
            length = seg_len
        else:
            buffer.append(segment)
            length += seg_len
    if buffer:
        pieces.append("\n\n".join(buffer))
    return pieces if pieces else [text]


def _split_chunk(chunk: str) -> List[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", chunk.strip()) if p.strip()]
    if len(paragraphs) <= 1:
        midpoint = len(chunk) // 2
        return [chunk[:midpoint].strip(), chunk[midpoint:].strip()]
    target = len(chunk) // 2
    accum = 0
    first: List[str] = []
    second: List[str] = []
    for para in paragraphs:
        if accum <= target:
            first.append(para)
            accum += len(para) + 2
        else:
            second.append(para)
    if not first or not second:
        midpoint = len(chunk) // 2
        return [chunk[:midpoint].strip(), chunk[midpoint:].strip()]
    return ["\n\n".join(first), "\n\n".join(second)]


def _extract_json(payload_text: str) -> Optional[str]:
    try:
        start = payload_text.index("{")
        end = payload_text.rindex("}") + 1
        fragment = payload_text[start:end]
        json.loads(fragment)
        return fragment
    except (ValueError, json.JSONDecodeError):
        return None


def _parse_response_payload(content: str) -> Dict[str, object]:
    last_error: Optional[Exception] = None
    candidates: List[str] = []
    if content:
        candidates.append(content)
        extracted = _extract_json(content)
        if extracted and extracted != content:
            candidates.append(extracted)

    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as exc:
            last_error = exc
            try:
                parsed = ast.literal_eval(candidate)
                if isinstance(parsed, dict):
                    return json.loads(json.dumps(parsed))
            except Exception as literal_exc:
                last_error = literal_exc

    if last_error:
        raise last_error
    raise ValueError("Empty response payload")


def _voice_assignment(voice_map: Dict[str, Tuple[str, str]], actor_name: str) -> Optional[VoiceAssignment]:
    token = _normalise_actor_token(actor_name)
    if not token:
        return None
    if token in voice_map:
        eng, voice = voice_map[token]
        if eng and voice and voice.lower() != "unassigned":
            capability = _lookup_voice_capability(eng, voice)
            return VoiceAssignment(engine=eng, voice_id=voice, source="configured", capability=capability)
    voice_id = _auto_select_openai_voice(token)
    used_voices = {voice for eng, voice in voice_map.values() if eng == "openai"}
    if voice_id in used_voices:
        for candidate in OPENAI_TTS_VOICE_ORDER:
            if candidate not in used_voices:
                voice_id = candidate
                break
    capability = OPENAI_VOICE_CAPABILITIES.get(voice_id)
    assignment = VoiceAssignment(engine="openai", voice_id=voice_id, source="auto", capability=capability)
    voice_map[token] = (assignment.engine, assignment.voice_id)
    return assignment


def _default_voice_style(assignment: Optional[VoiceAssignment]) -> str:
    if assignment and assignment.capability:
        default_style = assignment.capability.instruction_defaults.get("voice_style")
        if default_style:
            return default_style
    if assignment and assignment.voice_id:
        if assignment.engine == "openai":
            return f"{assignment.voice_id} | warm empathetic"
        return f"{assignment.voice_id} | neutral"
    return "neutral | assign later"


def _voice_guidance(actors: Iterable[str], voice_map: Dict[str, Tuple[str, str]]) -> str:
    notes: List[str] = []
    for name in actors:
        assignment = _voice_assignment(voice_map, name)
        if not assignment:
            notes.append(f"{name}: no voice assigned; suggest tone + energy cues to help casting.")
            continue
        capability = assignment.capability
        if capability:
            hint = capability.instruction_hint()
            detail = capability.prompt_summary()
            if hint:
                notes.append(f"{name}: {assignment.as_hint()} – {detail}. Default cues: {hint}")
            else:
                notes.append(f"{name}: {assignment.as_hint()} – {detail}.")
        else:
            notes.append(f"{name}: {assignment.as_hint()} (no capability card yet; keep directions concise).")
    return "\n".join(notes)


def run_transform_job(
    request: TransformRequest,
    status_callback: StatusCallback | None = None,
) -> TransformResult:
    if OpenAI is None:
        raise TransformError("openai package not installed. Run: pip install openai")

    status_cb = status_callback or _default_status_callback

    preset = PRESET_MANAGER.get(request.preset_id)
    voice_map = _normalise_voice_map(request.voice_map)
    actors = request.actors or ["Narrator"]
    voice_guidance_text = _voice_guidance(actors, voice_map)

    status_cb({"status": f"Using preset: {preset.title} ({preset.preset_id})"})
    sys_prompt = PRESET_MANAGER.build_system_prompt(preset, actors, voice_guidance_text)

    key = _load_openai_key(request.openai_credentials_path)
    client = OpenAI(api_key=key)

    base_label = request.script_label or f"{preset.preset_id}_{int(time.time())}"
    safe_label = sanitize_filename(base_label) or f"{preset.preset_id}_{int(time.time())}"

    chunks = _chunk_text(request.raw_text)
    if not chunks:
        raise TransformError("No content detected in chapter.")

    preset_out_dir = os.path.join(request.outdir, request.transform_subdir, preset.preset_id)
    os.makedirs(preset_out_dir, exist_ok=True)
    raw_debug_dir = os.path.join(preset_out_dir, "_raw_responses")
    try:
        os.makedirs(raw_debug_dir, exist_ok=True)
    except OSError:
        raw_debug_dir = None

    all_records: List[Dict[str, object]] = []
    prev_summary = "No prior context."
    next_scene = 1

    idx = 0
    split_events = 0
    max_split_events = 12

    def debug_dump(content: str, chunk_idx: int, attempt: int) -> None:
        if not raw_debug_dir:
            return
        try:
            path = os.path.join(raw_debug_dir, f"{safe_label}_chunk{chunk_idx}_attempt{attempt}.txt")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(content)
        except Exception:
            logger.debug("Failed to write raw response debug file", exc_info=True)

    while idx < len(chunks):
        total_chunks = len(chunks)
        chunk = chunks[idx]
        chunk_label = f"Transforming chunk {idx + 1}/{total_chunks} (≈{len(chunk)} chars) as {preset.preset_id}..."
        progress = 40 + int((idx / max(total_chunks, 1)) * 40)
        status_cb({"progress": progress, "status": chunk_label})

        line_budget = PRESET_MANAGER.compute_line_budget(preset, len(chunk))

        recent_lines = all_records[-5:]
        context_lines = [f"{item.get('actor', 'Narrator')}: {item.get('text', '')}" for item in recent_lines]
        payload = {
            "preset": preset.preset_id,
            "chunk_index": idx + 1,
            "total_chunks": total_chunks,
            "next_scene": next_scene,
            "previous_summary": prev_summary,
            "recent_lines": context_lines,
            "chunk_text": chunk,
            "line_budget": line_budget,
            "preset_directives": preset.system_blocks,
        }

        data: Optional[Dict[str, object]] = None
        last_error: Optional[Exception] = None

        for attempt in range(3):
            try:
                status_cb({"status": f"Requesting OpenAI for chunk {idx + 1}/{total_chunks} (try {attempt + 1}/3)..."})
                payload_with_retry = dict(payload)
                if attempt > 0:
                    payload_with_retry["repair_request"] = "Previous response was invalid JSON or schema. Return clean JSON only."
                resp = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": json.dumps(payload_with_retry, ensure_ascii=False)}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.5 if attempt else 0.55,
                    max_tokens=2200,
                    timeout=90,
                )
                message_content = resp.choices[0].message.content
                if isinstance(message_content, list):
                    combined = "".join(
                        part.get("text", "") if isinstance(part, dict) else getattr(part, "text", "")
                        for part in message_content
                    )
                    content = combined.strip()
                else:
                    content = (message_content or "").strip()
                debug_dump(content, idx + 1, attempt + 1)
                data = _parse_response_payload(content)
                if not isinstance(data, dict) or "lines" not in data:
                    raise ValueError("Invalid response format: missing 'lines'")
                PRESET_MANAGER.validate_response(preset, data)
                break
            except Exception as exc:  # pragma: no cover - runtime interaction with OpenAI
                last_error = exc
                logger.warning("Chunk %s attempt %s failed: %s", idx + 1, attempt + 1, exc)
                data = None
                time.sleep(1 + attempt)

        if data is None:
            if split_events < max_split_events:
                pieces = [p for p in _split_chunk(chunk) if p]
                if len(pieces) > 1:
                    logger.warning(
                        "Splitting chunk %s into %s smaller pieces due to repeated failures",
                        idx + 1,
                        len(pieces),
                    )
                    chunks[idx:idx + 1] = pieces
                    split_events += len(pieces) - 1
                    total_chunks = len(chunks)
                    continue
            raise TransformError(f"OpenAI transform failed on chunk {idx + 1}: {last_error}")

        chunk_lines = data.get("lines", []) if isinstance(data, dict) else []
        if not chunk_lines:
            logger.warning("Chunk %s returned no lines", idx + 1)
            idx += 1
            continue

        scene_offset = max(next_scene - 1, 0)
        max_scene_in_chunk = 0
        for raw_index, item in enumerate(chunk_lines, start=1):
            if not isinstance(item, dict):
                continue
            try:
                raw_scene = int(item.get("scene", max_scene_in_chunk + 1 or 1))
            except (TypeError, ValueError):
                raw_scene = max_scene_in_chunk + 1 or 1
            raw_scene = max(raw_scene, 1)
            max_scene_in_chunk = max(max_scene_in_chunk, raw_scene)
            global_scene = raw_scene + scene_offset

            actor_name = (item.get("actor") or "Narrator").strip() or "Narrator"
            text = (item.get("text") or "").strip()
            if not text:
                continue

            record_index = len(all_records) + 1
            code = f"{safe_label}-SC{global_scene:02d}-L{record_index:04d}"
            assignment = _voice_assignment(voice_map, actor_name)
            voice_style = _coerce_str(item.get("voice_style")) or _default_voice_style(assignment)
            voice_emotion = (item.get("voice_emotion") or item.get("emotion") or "").strip()
            pace = (item.get("pace") or item.get("speed") or "").strip() or "measured"
            beat = (item.get("beat") or item.get("intent") or "").strip() or "advance plot"

            if not voice_emotion:
                if assignment and assignment.engine == "openai":
                    voice_emotion = "confident"
                elif assignment and assignment.engine == "google":
                    voice_emotion = "cinematic"
                else:
                    voice_emotion = "neutral"

            base_fields = {
                "line_code": code,
                "scene": global_scene,
                "actor": actor_name,
                "voice": assignment.voice_id if assignment else "",
                "text": text,
                "actions": item.get("actions", ""),
                "expression": item.get("expression", ""),
                "gaze_to": item.get("gaze_to", ""),
                "sfx": item.get("sfx", ""),
                "notes": item.get("notes", ""),
                "speed": pace,
                "voice_style": voice_style,
                "voice_emotion": voice_emotion,
                "beat": beat,
            }

            record = PRESET_MANAGER.build_record(preset, base_fields, item, assignment)
            _apply_voice_metadata(record, assignment)
            all_records.append(record)

        next_scene = scene_offset + max_scene_in_chunk + 1
        continuity = data.get("continuity", {}) if isinstance(data.get("continuity"), dict) else {}
        prev_summary = continuity.get("summary") or "; ".join(rec["text"] for rec in all_records[-3:]) or prev_summary
        if isinstance(continuity.get("next_scene"), int) and continuity.get("next_scene") >= next_scene:
            next_scene = continuity.get("next_scene")

        progress = 40 + int(((idx + 1) / max(len(chunks), 1)) * 40)
        status_cb({"progress": progress, "status": f"Processed chunk {idx + 1}/{len(chunks)}"})
        idx += 1

    if not all_records:
        raise TransformError("Transformer returned no usable lines.")

    jsonl_path = os.path.join(preset_out_dir, f"{safe_label}.jsonl")
    csv_path = os.path.join(preset_out_dir, f"{safe_label}.csv")

    with open(jsonl_path, "w", encoding="utf-8") as jf, open(csv_path, "w", encoding="utf-8", newline="") as cf:
        writer = csv.DictWriter(cf, fieldnames=preset.output_fields, extrasaction="ignore")
        writer.writeheader()
        for rec in all_records:
            jf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            writer.writerow(rec)

    status_cb({"progress": 100, "status": "Transform complete."})
    return TransformResult(jsonl_path=jsonl_path, csv_path=csv_path, records=all_records, preset_id=preset.preset_id)


def run_transform_from_cli() -> None:  # pragma: no cover - CLI helper
    import argparse

    parser = argparse.ArgumentParser(description="ADRama transform CLI")
    parser.add_argument("input", nargs="?", help="Path to raw chapter .txt file")
    parser.add_argument("--actors", help="Comma separated list of actors", default="")
    parser.add_argument("--outdir", default=os.path.join(os.getcwd(), "out"))
    parser.add_argument("--transform-subdir", default="acted_scripts")
    parser.add_argument("--openai-credentials", default=None)
    parser.add_argument("--label", default=None, help="Optional base label for output files")
    parser.add_argument("--preset", default=DEFAULT_PRESET_ID, choices=[p[0] for p in list_presets()], help="Transform preset to use")
    parser.add_argument("--list-presets", action="store_true", help="List available presets and exit")
    args = parser.parse_args()

    if args.list_presets:
        for pid, title, description in list_presets():
            print(f"{pid}: {title}\n    {description}\n")
        parser.exit()

    if not args.input:
        parser.error("input is required unless --list-presets is used")

    with open(args.input, "r", encoding="utf-8") as fh:
        raw_text = fh.read()

    actors = [a.strip() for a in args.actors.split(",") if a.strip()] or ["Narrator", "Alice", "Bob", "Cara", "Dylan"]

    request = TransformRequest(
        raw_text=raw_text,
        actors=actors,
        outdir=args.outdir,
        transform_subdir=args.transform_subdir,
        voice_map={},
        openai_credentials_path=args.openai_credentials,
        script_label=args.label,
        preset_id=args.preset,
    )

    def cli_status(msg: StatusMessage) -> None:
        status = msg.get("status")
        progress = msg.get("progress")
        if status and progress is not None:
            print(f"[{progress}%] {status}")
        elif status:
            print(status)

    try:
        result = run_transform_job(request, status_callback=cli_status)
    except TransformError as exc:
        parser.exit(1, f"Transform failed: {exc}\n")

    print("Preset:", result.preset_id)
    print("JSONL:", result.jsonl_path)
    print("CSV:", result.csv_path)


if __name__ == "__main__":  # pragma: no cover - CLI entry
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    run_transform_from_cli()
