# Hand-Off Guide for Next Codex Session

## Current State (Sept 2025)
- GUI (`app.py`) manages batched TTS synthesis, voice assignments, and delegates OpenAI “Transform Raw Chapter” jobs to the standalone `transformer_service` module. Preset selection (`audio_drama`, `audiobook_faithful`, `cinematic_screenplay`, `interactive_sim`, `storyboard`) persists via `config.json`.
- Preview table (`ttk.Treeview`) renders each line with metadata (engine, voice, emotion, beat, pace, instructions). Double-click a row to sync edits, regenerate the line (or emit a dry-run request), and play it instantly; Line menu or `Ctrl+E` opens a multi-field editor, and manifests/deterministic filenames stay intact.
- `transformer_service.py` composes modular prompt blocks, computes per-preset line budgets, auto-splits long paragraphs (~800 char targets), validates responses against `schemas/base_transform.schema.json`, and writes outputs to `out/<transform_subdir>/<preset_id>/`. `_raw_responses/` captures every chunk attempt for debugging.
- `scripts/run_transform_cli.sh` enables headless transforms (quote paths with spaces). `jsonschema` ships in `requirements.txt` so response validation runs by default.
- Batch generation writes `manifest.json`, `timeline.{csv,json}`, and a bundle zip alongside merged MP3s. Setting `ADRAMA_DRY_RUN=1` before launch causes both batch and single-line generation to emit `[DRY RUN]` request previews instead of hitting the APIs.
- Voice browser still surfaces OpenAI/Google lists; unavailable Google voices persist in `out/google_unavailable.json` and are skipped after failure.

## Quick Validation Checklist
1. `./start_app.sh` → load `SAMPLE_scripts/testmorescript.txt` → “Generate MP3s + Merge”; verify per-line MP3s, merged episode, bundle, and `manifest.json` under `lines/testmorescript/`.
2. Double-click a line in the preview, edit `voice=...` or the actor name (Line → Edit Selected Line or `Ctrl+E`), save with “Regenerate”, and confirm the new audio/manifest entry reflects the change.
3. Dry-run audit: `ADRAMA_DRY_RUN=1 ./start_app.sh`, load the same script, double-click a line—expect a `[DRY RUN]` request file instead of audio.
4. Transform check: select preset **Faithful Audiobook**, load `SAMPLE_scripts/chapter 2partA.txt`, run “Transform Raw Chapter…”. Outputs should land in `out/acted_scripts/audiobook_faithful/`; inspect `_raw_responses/` to confirm chunk sizes are ~800 chars.
5. Save flow: after edits/toying with metadata, use File → Save Script (or `Ctrl+S`) and confirm a JSONL with updated metadata appears at the chosen path.
6. Optional CLI smoke test (requires OpenAI creds):
   ```bash
   ./scripts/run_transform_cli.sh "SAMPLE_scripts/chapter 2partA.txt" \
     --preset audio_drama --actors "Narrator,Kai,Lira"
   ```
   Inspect `out/acted_scripts/audio_drama/` for JSONL/CSV and `_raw_responses/` for raw payloads.

## Roadmap / Open Tasks
### A. Voice Intelligence & Direction
- Build “voice capability cards” per engine voice (supported emotions, accents, recommended delivery) and funnel them into preset prompt blocks + GUI defaults.
- Integrate upcoming OpenAI natural-language voice directives so emotional cues map cleanly to TTS instructions.
- Surface alternative-take controls in the upcoming `ttk.Treeview` preview (rate, audition variants, etc.).

### B. Transform Enhancements
- Expand JSON Schemas per preset (camera metadata, branching hooks, storyboard prompts) and tighten validation.
- Expose advanced preset toggles directly in the GUI (sound design, branching, storyboard) for quick experimentation.
- Implement continuity/beat tagging helpers to support downstream reordering or visualization.

### C. Workflow Automation
- Scene-level exports for the generation pipeline (actor metadata, scene IDs) to complement transform outputs.
- Optional queue/service layer so multiple scripts can be processed headlessly without tying up the GUI.
- Manifest enhancements: add checksum/versioning so stale audio vs. edited lines are detected automatically.

## Implementation Notes for Codex
- GUI ↔ transformer integration lives near `_do_transform_work`; status messages are pushed through the Tk task queue (`forward_status`).
- Preset metadata lives at the top of `transformer_service.py`; adding a new preset requires updating that list, defining output fields, and wiring a schema.
- Schema validation is optional—if `jsonschema` is missing the service skips validation—but in dev it should be installed (`requirements.txt`).
- Config persistence includes `transform_preset`; ensure new presets are recognised when loading old configs.
- CLI depends on quoting for paths with spaces; keep documentation consistent when adding examples.
- Line chunking defaults to ~800 characters; `_chunk_text` now slices paragraphs exceeding the limit and retries add `repair_request` hints before splitting again.

Provide this file to the next Codex CLI agent so they can continue development with full awareness of the split architecture, presets, and roadmap.
