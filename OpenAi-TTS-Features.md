# OpenAI TTS — Features, Integration & Roadmap
*Project: ADRama – Book ➜ Script ➜ Voice*

Purpose: encode emotive delivery with OpenAI voices, wire it into ADRama’s JSON line schema, and define a short‑to‑ship plan (plus a future roadmap).

---

## 0) TL;DR
- Control delivery via a concise **`instructions`** string per utterance.
- Use **Speech API** (`gpt-4o-mini-tts`) for deterministic file/stream output; **Realtime API** (`gpt-realtime`) for live agents and overrides.
- Build `instructions` from ADRama metadata: `voice_style`, `voice_emotion`, `micro_emotion`, `pace`, `vocal_technique`, `accent_hint`, `rhythm_signature`.
- Keep directives short and explicit. Audition first, then render.

---

## 1) Engines
### Speech (file/stream)
- **Model**: `gpt-4o-mini-tts`
- **Params**: `voice`, `input` (text), `instructions` (guidance string).
- **Voices**: 11 built-in voices (subset may be exposed in-app).
- **Use**: scripted playback; batch rendering; consistent outputs.

### Realtime (voice↔voice)
- **Model**: `gpt-realtime`
- **Session**: default narrator **session instructions** (tone/pace).
- **Per-utterance**: optional **override instructions** for momentary shifts (e.g., “urgent, breathless”).
- **Use**: interactive agents, barge‑in, dynamic conversation.

> There is no SSML. Style is guided by natural language in `instructions`.

---

## 2) ADRama line schema additions
Superset fields (presets may hide some columns). See the JSON Schema file `adrama_line.schema.json` for validation.

```json
{
  "voice_engine": "openai_tts | openai_realtime",
  "voice_id": "coral | ash | ...",
  "voice_style": "calm authority",
  "voice_emotion": "hopeful",
  "micro_emotion": "steadfast",
  "pace": "measured",
  "vocal_technique": "soft projection, crisp consonants",
  "accent_hint": "subtle French",
  "rhythm_signature": "4-beat, rising",
  "instructions": "computed string (see template)"
}
```

### Instruction string template
Concise, semi-structured; include fields only when present.

```
{voice_style}; emotion: {voice_emotion}{, micro: {micro_emotion}}; pace: {pace}; technique: {vocal_technique}; accent: {accent_hint}; rhythm: {rhythm_signature}.
```

---

## 3) Instruction Builder (Python)
Centralize instruction building to keep behavior consistent across renderers.

```python
def build_instructions(meta: dict) -> str:
    bits = []
    if meta.get("voice_style"):
        bits.append(meta["voice_style"])
    if meta.get("voice_emotion"):
        bits.append(f"emotion: {meta['voice_emotion']}")
    if meta.get("micro_emotion"):
        bits.append(f"micro: {meta['micro_emotion']}")
    if meta.get("pace"):
        bits.append(f"pace: {meta['pace']}")
    if meta.get("vocal_technique"):
        bits.append(f"technique: {meta['vocal_technique']}")
    if meta.get("accent_hint"):
        bits.append(f"accent: {meta['accent_hint']}")
    if meta.get("rhythm_signature"):
        bits.append(f"rhythm: {meta['rhythm_signature']}")
    return ("; ".join(bits) + ".") if bits else ""
```

---

## 4) Speech API — Deterministic Rendering (Python)
Stream each line to disk (mp3/wav).

```python
from pathlib import Path
from openai import OpenAI

client = OpenAI()

def render_tts_line(line: dict, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{line.get('scene','scene')}-{line.get('beat','beat')}-{line.get('actor','actor')}.mp3"
    out_path = out_dir / filename
    with client.audio.speech.with_streaming_response.create(
        model="gpt-4o-mini-tts",
        voice=line["voice_id"],
        input=line["text"],
        instructions=line.get("instructions","")
    ) as resp:
        resp.stream_to_file(out_path)
    return out_path
```

Notes:
- Deterministic naming helps timeline assembly.
- Batch iterate scene lines; mix SFX/music in post.

---

## 5) Realtime API — Live Session (JavaScript sketch)
Set defaults on the session; override per response when needed.

```js
const sessionRes = await fetch("https://api.openai.com/v1/realtime/sessions", {
  method: "POST",
  headers: {
    "Authorization": `Bearer ${process.env.OPENAI_API_KEY}`,
    "Content-Type": "application/json"
  },
  body: JSON.stringify({
    model: "gpt-realtime",
    voice: "marin",
    instructions: "You are a narrator. Warm, measured baseline. Avoid shouting."
  })
});
const { websocket_url } = await sessionRes.json();
// Connect WebSocket to websocket_url...

ws.send(JSON.stringify({
  type: "response.create",
  response: {
    instructions: "urgent; emotion: tense; pace: fast; technique: clipped consonants.",
    modalities: ["audio"],
    content: [{ type: "input_text", text: line.text }]
  }
}));
```

---

## 6) Example — Chapter 2 Part A line
Source line (Lira):
“They’ve put up a wall, but they haven’t won. The light isn’t in their hands, it’s in ours.”

ADRama emission:

```json
{
  "actor": "Lira",
  "text": "They’ve put up a wall, but they haven’t won. The light isn’t in their hands, it’s in ours.",
  "voice_engine": "openai_tts",
  "voice_id": "coral",
  "voice_style": "calm authority",
  "voice_emotion": "hopeful",
  "micro_emotion": "steadfast",
  "pace": "measured",
  "vocal_technique": "soft projection, crisp consonants",
  "rhythm_signature": "4-beat, rising",
  "instructions": "calm authority; emotion: hopeful; micro: steadfast; pace: measured; technique: soft projection, crisp consonants; rhythm: 4-beat, rising."
}
```

---

## 7) Preset interplay
- **Audiobook**: keep `voice_style`, `voice_emotion`, `pace` minimal; preserve author voice.
- **Audio Drama**: add `micro_emotion`, `vocal_technique`, optional `accent_hint`, `rhythm_signature`, plus SFX/music columns.
- **Screenplay**: add `camera`, `lighting`, `prop_notes` (non‑audio) for storyboarding; TTS still uses `instructions`.
- **Interactive**: add `npc_state`, `player_prompt`, `branch_hook`; pair naturally with Realtime.

---

## 8) Voice Capability Cards
Maintain small profiles per voice to auto‑tune instructions.

```json
{
  "coral": {
    "range": "mid",
    "energy": "calm → assertive",
    "defaults": {
      "voice_style": "calm authority",
      "pace": "measured"
    },
    "notes": "Good with hopeful/steady reads; keep accent hints subtle."
  }
}
```

---

## 9) Guardrails
- Be concise: one phrase per attribute (style, emotion, pace, technique, accent, rhythm).
- Audition first; paste the same `instructions` into production.
- Prefer subtle accent hints to keep intelligibility high.
- < ~35 spoken words per line for natural delivery.
- Keep style/pace stable within a scene unless a beat changes.

---

## 10) QA Harness
- Pick 5 lines per scene: calm / urgent / whisper / reflective / angry.
- Render with Speech; log `instructions`, file path, duration.
- Listen for clarity, pacing, consistency; iterate the wording.

---

## 11) Roadmap
**Short term**
- Implement schema fields + instruction builder.
- Wire Speech renderer; add audition loop in UI.
- Add capability cards for the 6 in‑app voices.

**Medium term**
- Add Realtime renderer with session defaults + per‑utterance overrides.
- Preset‑aware exports (Audiobook, Audio Drama, Screenplay, Interactive).
- JSON Schema validation per preset; auto‑repair on failure.

**Longer term**
- Per‑character delivery profiles (arc, default emotions by scene).
- Auto‑mixing hooks (music bed, SFX lanes).
- Branching hooks for interactive drama; state‑aware delivery.

---

## 12) Hand‑off Checklist
- [ ] Schema extended; `adrama_line.schema.json` added.
- [ ] `build_instructions()` implemented and unit‑tested.
- [ ] Speech renderer working (stream to file; deterministic naming).
- [ ] Audition UI path wired.
- [ ] Voice capability cards created.
- [ ] Realtime README snippet prepared.
