# ADRama Transform Prompt Blueprint

This document captures the transform prompt we currently send to OpenAI, then extends it into a modular “Book ➜ Script” framework. Use it as a launch pad for building richer presets (faithful audiobook, audio drama, cinematic screenplay, interactive sim, etc.) and for iterating on voice intelligence cues.

---

## 1. Current Prompt (v2.2.0)

```text
You are ADRama's dramatization and blocking engine for a rehearsal table read.
Actor roster (rotate naturally, keep voices balanced): {actor_list}.
Current voice intelligence hints:
{voice_guidance_text}
For every chunk of narrative you receive, do ALL of the following:
1. Cover every plot beat in the text—do not skip minor actions or dialogue.
2. Split narration into short, speakable lines (max ~35 words) with clear performer assignments.
3. Maintain continuity across chunks using the provided previous summary and next_scene start.
4. For each line emit an object with fields: actor, text, scene (int), actions, expression, gaze_to, sfx, notes, voice_style, voice_emotion, beat, pace.
   - actions/expression/gaze_to/sfx should be short tokens or phrases.
   - voice_style: 2-4 word direction that pairs the assigned voice with delivery guidance.
   - voice_emotion: single dominant emotion keyword (e.g., 'hopeful', 'tense').
   - beat: one short phrase describing the dramatic intention of the line.
   - pace: describe delivery speed with one word (e.g., 'measured', 'urgent').
5. Return valid JSON with keys: lines (array) and continuity (object with summary + next_scene).
6. Honor the provided line_budget value—never return more than that number of entries in lines. If coverage remains, summarize the remainder in continuity.pending for the next chunk.
Keep tone PG-rated, compress exposition, and retain canonical terminology from the source.
Do not wrap the JSON in markdown fences or add commentary—respond with JSON only.
```

### Payload shape (per chunk)

```json
{
  "chunk_index": 1,
  "total_chunks": 3,
  "next_scene": 1,
  "previous_summary": "Summary of prior beats",
  "recent_lines": ["Actor: last line", "Actor: penultimate line"],
  "chunk_text": "Raw prose...",
  "line_budget": 20,
  "repair_request": "Previous response was invalid JSON. Return clean JSON only." // only on retries
}
```

Keep this baseline handy—it supplies continuity metadata, voice hints, and line budgeting. Every experimental preset can be derived from this foundation.

---

## 2. Multi-Preset Strategy

### 2.1 Preset Matrix

| Preset ID | Title | Intent | Output Tone | Suggested `line_budget` | Notes |
| --- | --- | --- | --- | --- | --- |
| `audio_drama` | Audio Drama | Cinematic soundscape and performances | Dialogue-forward, strong SFX | 18–32 | Adds stage directions, stingers, crossfades; pushes `voice_style`, `micro_emotion`, `music_bed`, `sfx_layer`. |
| `audiobook_faithful` | Faithful Audiobook | Stay 100% faithful to source text | Narrative-heavy, third-person | 16–24 | Minimal dramatization; maintain author voice with subtle performance cues. |
| `cinematic_screenplay` | Cinematic Screenplay | Scene/action blueprint | Balanced action + dialogue | 16–26 | Includes sluglines (`scene_label`), camera cues, props; enforces three-act pacing markers. |
| `interactive_sim` | Interactive Simulation | Branch-ready world bible | System + actor notes | 12–20 | Expands `npc_state`, `player_prompt`, `branch_token`, `world_state` for later game logic. |
| `storyboard` | Storyboards & Mood | Visual-first planning | Mood boards, frame beats | 10–18 | Provides `frame_prompt`, `mood_board`, `camera_move`, `color_palette` for art pipelines. |

Each preset can share the same transport shape but swap prompt clauses, output columns, and validation logic.

### 2.2 Prompt Blocks

Break the system prompt into modules you can toggle:

1. **Narrative fidelity block** – dial between verbatim paraphrase vs. adaptive rewrite.
2. **Sound design block** – require `music_bed`, `sfx_layer`, `silence_beats` arrays.
3. **Cinematic block** – request `camera`, `shot_type`, `lighting`, `prop_notes`.
4. **Interactive block** – add `npc_state`, `player_prompt`, `branch_token`.
5. **Performance block** – extend `voice_style` with `micro_emotion`, `vocal_technique`, `breath_control`, `accent_hint`, `rhythm_signature`.
6. **Casting block** – include `voice_engine_hint`, `primary_voice`, `fallback_voice`, and inject capability cards per actor.

Compose the final prompt by concatenating baseline + selected blocks. This makes it easy to expose checkboxes or presets in the UI.

The current transformer service composes these blocks automatically based on the selected preset (e.g., `[PRESET: Audio Drama]`).

---

## 3. Enhanced Voice Intelligence

We already provide `voice_style` and `voice_emotion`. To enrich casting and direction:

- **Add `micro_emotion`**: single-word nuance (e.g., “wistful”, “bittersweet”).
- **Add `vocal_technique`**: mention breathiness, projection, crisp consonants.
- **Add `accent_hint`**: optional region/dialect cues (guardrails when voices support variants).
- **Add `breath_control`**: breaths, pauses, held silences to sync with SFX.
- **Emphasize `rhythm_signature`**: per-line beat pacing, e.g., “4-beat, syncopated, tension rise”.
- **Integrate `call_and_response` metadata**: helps align overlapping voices or interjections.
- **Capture `voice_engine_hint`, `primary_voice`, `fallback_voice`**: steer lines toward OpenAI or Google and keep casting resilient.

When generating prompts for voices, reference the current voice map so the model provides directions compatible with available engines (OpenAI vs Google). Consider caching per-voice “strength cards” (range, energy, default emotion). That card can be inserted into the voice guidance block for bespoke coaching.

---

## 4. Optional Output Columns by Preset

| Column | Audiobook | Audio Drama | Screenplay | Interactive | Storyboard |
| --- | --- | --- | --- | --- | --- |
| `scene` | ✓ | ✓ | ✓ (plus scene slug) | ✓ | ✓ |
| `voice_style` / `voice_emotion` | ✓ | ✓ (expanded) | ✓ | ✓ | ✓ |
| `actions`, `expression`, `gaze_to` | ✓ | ✓ | ✓ | ✓ | ✓ |
| `sfx`, `music_bed`, `silence_beats`, `sfx_layer` | optional | ✓ | optional | optional | ✓ |
| `camera`, `shot_type`, `lighting`, `prop_notes` | — | optional | ✓ | optional | ✓ |
| `npc_state`, `player_prompt`, `branch_token`, `world_state` | — | — | — | ✓ | — |
| `frame_prompt`, `mood_board`, `camera_move`, `color_palette` | — | optional | optional | — | ✓ |
| `voice_engine_hint`, `primary_voice`, `fallback_voice` | optional | ✓ | ✓ | ✓ | optional |

Choose a superset schema per preset, then include/exclude columns at export time. Scripts can still be down-sampled to the simple `actor/text` pair for legacy pipelines.

---

## 5. Future Scope Ideas

- **Character Blueprints**: parallel output file with per-character arcs, dominant emotions per scene, wardrobe, prop affinities.
- **Scene Storyboards**: request the model to emit `frame_prompt` strings ready for image-gen tools (e.g., Stable Diffusion, Sora).
- **Beat Tagging**: label beats with story structure tags (`inciting_incident`, `pinch_point`, `climax`).
- **Choosable Paths**: optional `branch_points` that describe “if/then” outcomes to build branching audio drama or game content.
- **Design Notes**: global section summarizing tone, recommended instruments, atmosphere loops.
- **Continuity Pending Queue**: persist `continuity.pending` for text not yet dramatized to drive additional passes.
- **Integration Hooks**: generate JSON-LD or GraphQL skeletons so external tools (RenPy, Ink, Unreal) can consume the data.
- **Revision Loop**: allow the transform to ingest peer feedback and request a rewrite focusing on pacing, emotional clarity, or voice balance.

---

## 6. Implementation Notes

1. **Prompt Assembly**: the standalone `transformer_service` composes system prompts from preset blocks and exposes both CLI (`python transformer_service.py --preset audio_drama`) and GUI integrations.
2. **Preset Tokens**: every preset injects `[PRESET: …]` so the model recognises the requested style and optional fields.
3. **Schema Switching**: preset metadata defines output columns and references `schemas/base_transform.schema.json` for validation (via `jsonschema` when installed).
4. **GUI Integration**: the Tkinter app delegates work to the transformer service, offering a style combobox while maintaining a single cohesive UI.
5. **Voice Capability Cards**: keep extending the voice-map metadata to feed richer guidance blocks (future work).
6. **Caching & Replay**: raw responses live under `out/<transform_subdir>/<preset_id>/_raw_responses/`, making debugging and replays straightforward.
7. **Continuity Controls**: `continuity.pending` is preserved so subsequent passes can resolve any skipped beats.

---

## 7. Next Steps

1. Flesh out per-preset JSON Schemas (beyond the base validator) to enforce optional columns and data types.
2. Build voice capability cards and feed them into the Performance/Casting blocks (engine-aware directions).
3. Experiment with additional presets (e.g., comedy timing, documentary) and surface them via the combobox.
4. Add automated regression tests for the CLI—mock OpenAI responses and assert schema compliance.
5. Gather user feedback on pacing & emotional diversity, then tune default line budgets and prompt directives.

Use this blueprint to craft richer experiences—everything from faithful audiobook reads to game-ready scene kits—while keeping the pipeline consistent and inspectable.
