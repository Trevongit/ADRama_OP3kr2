# Potential Next Features

## 1. Advanced Batching Controls
- **Dynamic worker scaling** based on detected API rate limits or voice-engine latency.
- **Per-engine throttling** (e.g., serialize OpenAI jobs but keep Google concurrent).
- **Retry/backoff policy** for transient Google ServiceUnavailable errors before blacklisting voices.

## 2. Workflow Automation
- **Scene-based grouping** that exports chapter folders with metadata (timings, speaker order).
- **Timeline export** (CSV/JSON) for DAW import, listing clip durations and timestamps.
- **CLI generator** (`python generate.py --script script.jsonl`) for headless batch jobs.

## 3. Enhanced Voice Intelligence
- **Voice recommendations** (e.g., auto-map based on character tags or sentiment).
- **Voice quality ratings** collected from prior runs or user feedback.
- **Side-by-side audition** for multiple voices with quick switch & compare.

## 4. Monitoring & Logging
- **GUI log console** showing recent warnings/errors without opening `adrama_errors.log`.
- **Prometheus metrics** when running the API (request counts, synth durations, error types).

## 5. Collaboration & Versioning
- **Project bundle exporter** (script, config, generated audio, metadata) zipped for handoff.
- **Voice map snapshots** with timestamps, enabling rollback/compare.

## 6. Docker & Deployment Enhancements
- **Multi-stage builds** to keep images minimal.
- **Prod compose stack** with Traefik/HTTPS for hosting the API safely.
- **Kubernetes helm chart** if running at scale.

These items complement the current batching engine and provide a roadmap for AI-assisted and collaboration-centric workflows.
