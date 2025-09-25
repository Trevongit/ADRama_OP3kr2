import json
from build_instructions import build_instructions

def test_minimal():
    meta = {"voice_style": "calm authority", "voice_emotion": "hopeful", "pace": "measured"}
    out = build_instructions(meta)
    assert out == "calm authority; emotion: hopeful; pace: measured."

def test_full():
    meta = {
        "voice_style": "calm authority",
        "voice_emotion": "hopeful",
        "micro_emotion": "steadfast",
        "pace": "measured",
        "vocal_technique": "soft projection, crisp consonants",
        "accent_hint": "subtle French",
        "rhythm_signature": "4-beat, rising"
    }
    out = build_instructions(meta)
    assert out == "calm authority; emotion: hopeful; micro: steadfast; pace: measured; technique: soft projection, crisp consonants; accent: subtle French; rhythm: 4-beat, rising."
