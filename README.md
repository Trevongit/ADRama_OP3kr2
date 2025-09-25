# ADRama TTS Studio — System Overview (AI-Oriented)

## 1. Purpose and High-Level Flow
- **Goal**: Turn multi-speaker scripts into fully voiced audio dramas using cloud TTS engines (OpenAI + Google) with batching, caching, line-level regeneration, and optional REST API.
- **Primary executable**: `app.py` (Tkinter GUI). Secondary: `api.py` (FastAPI service).
- **Core loop**: load or transform a script → map actors to engines/voices (auto + manual overrides) → generate line audio (batched, cached, or ad‑hoc) → merge to episode MP3 with manifest/timeline artifacts.

```
User Script ─┬─> ScriptData loader ─┬─> Voice assignment map
             │                     └─> Batch scheduler → TTS engines → MP3 cache
             └─> Transform (optional, via OpenAI) ─────────────────────────────┘
```

## 2. Key Modules
- **ADRamaApp**: GUI controller; manages config, menus (File/Line/Voice/Transform), task queue, manifest bookkeeping, and status updates.
- **Preview table** (`ttk.Treeview`):
  - Columns: line code, actor, text, voice, engine, emotion, beat, pace, instructions.
  - Double-click a row to sync edits, regenerate audio (or dry-run request), and play it immediately.
  - `Ctrl+E` / Line → Edit Selected Line opens a multi-field editor for actor/text/voice/emotion/instructions; saving optionally regenerates the take.
  - `Ctrl+P` / Line → Regenerate plays the selected line without hunting for the file.
  - Actor/voice changes update the roster; cached audio is invalidated so you always hear the freshest render.
- **File menu**: adds “Save Script”/“Save Script As…” (JSONL) alongside “Open Script…” and “Choose Output Folder…”. Scripts persist all line metadata for downstream tweaking.
- **Batching**: `_do_generate_work` launches up to `MAX_TTS_WORKERS` (default 4) using `ThreadPoolExecutor`. Cached lines are skipped, progress updates per completion, and final merge uses `AudioSegment`.
- **Batching**: `_do_generate_work` launches up to `MAX_TTS_WORKERS` (default 4) using `ThreadPoolExecutor`. Cached lines are skipped, progress updates per completion, and final merge uses `AudioSegment`.
- **Manifest**: every generation writes `manifest.json` (deterministic filenames, instructions, voice IDs, cached status) alongside merged MP3, `timeline.csv/json`, and bundle zip.
- **Dry run**: set `ADRAMA_DRY_RUN=1` before launching to emit request previews (`.txt`) instead of calling the TTS API—useful for auditing instructions/emotions quickly.
- **Voice assignment**:
  - OpenAI list fetched dynamically (`/v1/audio/voices`) with fallback to legacy six voices.
  - Google browser resolves metadata via `TextToSpeechClient.list_voices`, filters premium voices, hides failing voices, and shows details (language, gender, sample rate).
  - Selections write into `voice_map` and persist to `config.json`.
- **Transform tool**: optional script conversion using OpenAI chat completions (`gpt-4o-mini`), writing JSONL/CSV scripts to `out/acted_scripts/`.

### `api.py`
- FastAPI endpoints for `/health`, `/voices/openai`, `/voices/google`, `/tts/{engine}` using shared engine classes.
- Streams MP3 bytes back via `StreamingResponse`.

### `engines.py`
- Lightweight wrappers (`OpenAITTSEngine`, `GoogleCloudTTSEngine`) used by the API service.
- GUI has a richer fork with batching + metadata filtering; API version keeps minimal footprint.

## 3. Data & Config Layout
| Path | Purpose |
| --- | --- |
| `config.json` / `~/.adrama_config.json` | Persisted credentials paths, output folders, actor->voice map |
| `demos/` | Cached voice sample MP3s |
| `lines/<script_label>/` | Generated per-line audio, `manifest.json`, dry-run previews |
| `out/acted_scripts/` | OpenAI transform outputs (JSONL + CSV) |
| `out/<lines_subdir>/.../episode_merged.mp3` | Final merged episode |
| `adrama_errors.log` | Rotating log file for the GUI |

## 4. Batch Generation Algorithm
1. Pre-flight: resolve voice/engine per line (`_resolve_voice`), merge metadata, ensure directories exist.
2. **Caching**: if target MP3 exists the line is counted done immediately (manifest flags `cached=true`).
3. **Task pool**: schedule remaining lines on `ThreadPoolExecutor(max_workers=min(MAX_TTS_WORKERS, pending))`.
4. Each worker calls `_synthesize_batch_item` (engine.synthesize, now forwarding `instructions`) and reuses cached engine clients.
5. Failures mark Google voices unavailable (400/503) and emit a single GUI error; futures are cancelled.
6. Completed job → merge sorted line segments with `AudioSegment.silent()` + `+= AudioSegment.from_file`, update manifest/timeline, bundle artifacts.
7. Double-click regen performs the same pipeline for a single line, updating manifest entries in-place.

## 5. Credential Handling
- GUI startup (`start_app.sh`) reads JSON key files and exports env vars if `.env` present.
- OpenAI: expects JSON containing `{ "api_key": "..." }` or `OPENAI_API_KEY` env.
- Google: expects service-account JSON path assigned to `GOOGLE_APPLICATION_CREDENTIALS` (auto when chosen via GUI).

## 6. CLI / Scripts
| Script | Description |
| --- | --- |
| `start_app.sh` | Bootstrap venv, install deps, load secrets, launch GUI |
| `start_adramaz.sh` | Legacy launcher requiring pre-created `.venv` |
| `scripts/run_transform_cli.sh` | Convenience wrapper for the standalone transform service (delegates to `.venv/bin/python transformer_service.py`) |

> **Tip:** When paths include spaces, wrap them in quotes: `./scripts/run_transform_cli.sh "SAMPLE_scripts/chapter 2partA.txt" --preset audio_drama --actors "Narrator,Kai,Lira"`

## 7. Transform Service & Presets
- Core logic lives in `transformer_service.py`; the GUI only forwards jobs and streams status so both pieces behave like one app.
- Presets currently shipped: `audio_drama`, `audiobook_faithful`, `cinematic_screenplay`, `interactive_sim`, `storyboard`. Each preset assembles modular prompt blocks, sets a line-budget heuristic, and selects its own CSV/JSONL columns.
- Responses are validated against `schemas/base_transform.schema.json` (requires the optional `jsonschema` dependency, now included in `requirements.txt`).
- Output layout: `out/<transform_subdir>/<preset_id>/<label>.{jsonl,csv}` plus `_raw_responses/` for every chunk attempt.
- Chunking: long inputs are automatically split (~800 char targets). Overlong paragraphs are sliced into smaller pieces to avoid malformed JSON responses; retry payloads include `repair_request` and parsing now tolerates partial fragments before re-splitting.
- CLI quick reference:
  - List styles: `./scripts/run_transform_cli.sh --list-presets`
  - Transform: `./scripts/run_transform_cli.sh "SAMPLE_scripts/chapter 2partA.txt" --preset cinematic_screenplay --actors "Narrator,Hero,Villain"`
- GUI additions:
  - Transform style combobox persists the chosen preset via `config.json`.
  - Status bar surfaces preset-aware progress messages (e.g., “Transforming chunk 1/3 as audio_drama…”).

**Roadmap highlights**
1. Extend per-preset JSON Schemas to enforce richer optional fields (camera data, NPC metadata, storyboard prompts).
2. Add voice capability cards so prompt guidance is tailored to each engine/voice’s strengths.
3. Surface advanced preset toggles (sound design, branching, storyboard prompts) directly in the GUI for quick experimentation.

## 8. Docker Components
### Dockerfile
- Base: `python:3.12-slim` + `ffmpeg` + `build-essential` (needed for `pydub`, `grpc`).
- Installs `requirements.txt`, copies repo, then drops to non-root `appuser`.
- Default CMD runs `uvicorn api:app` on `0.0.0.0:8000` (production mode).

### docker-compose.dev.yml
- Dev-targeted service mounting the repo and running `uvicorn ... --reload` for hot reload.
- Ports: `8000:8000` for API access.
- Loads environment variables from `.env` (for API keys, etc.).

### Typical Workflow
1. `docker compose -f docker-compose.dev.yml up --build` → builds image & starts API with live code volume.
2. GUI can still run locally hitting the API (or not). API responses stream MP3 bytes using shared engine logic.

## 9. Notable Edge Handling
- Google voices requiring explicit models or lacking language codes are auto-removed after first failure.
- Busy cursor helper ensures UI feedback without invalid Tk cursors on Linux.
- Logs capture both successes and failures; repeated errors (e.g., invalid locale) quickly surface in the GUI via status updates.
- Transform retries capture raw payloads under `_raw_responses/`, making JSON issues debuggable without reruns.
- `ADRAMA_DRY_RUN` allows full pipeline rehearsal (manifest + previews) without emitting audio or spending API credits.

## 10. Extension Points
- Increase `MAX_TTS_WORKERS` cautiously (respect API rate limits).
- Add new TTS providers by implementing `TTSEngine` and extending voice assignment UI.
- REST API already shares engines; a future task queue / caching layer outside the GUI could consume the same batching helper.
