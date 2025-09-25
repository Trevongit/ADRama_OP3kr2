# api.py
import io
import os
import tempfile
from typing import Optional

from fastapi import FastAPI, HTTPException, Header, Depends
from pydantic import BaseModel
from starlette.responses import StreamingResponse

from engines import OpenAITTSEngine, GoogleCloudTTSEngine

app = FastAPI(title="ADRama TTS API", version="0.1")

# ---- optional simple API key guard (enable by setting API_KEY in .env) ----
def require_api_key(x_api_key: str = Header(default="")):
    expected = os.getenv("API_KEY", "")
    if expected and x_api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid API key")
# ---------------------------------------------------------------------------

_openai_engine = None
_google_engine = None

def get_openai():
    global _openai_engine
    if _openai_engine is None:
        model = os.getenv("OPENAI_TTS_MODEL", "tts-1")
        cred_path = os.getenv("OPENAI_CREDENTIALS")  # e.g. /app/openai_credentials.json
        _openai_engine = OpenAITTSEngine(credentials_path=cred_path, model=model)
    return _openai_engine

def get_google():
    global _google_engine
    if _google_engine is None:
        cred_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")  # e.g. /app/google_credentials.json
        _google_engine = GoogleCloudTTSEngine(credentials_path=cred_path)
    return _google_engine

class TTSIn(BaseModel):
    text: str
    voice: Optional[str] = None  # e.g., "alloy" or "en-US-Wavenet-D"

def _synthesize_to_mp3_bytes(engine, text: str, voice: Optional[str]) -> bytes:
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        out_path = tmp.name
    try:
        engine.synthesize(text, out_path, voice=voice)
        with open(out_path, "rb") as f:
            return f.read()
    finally:
        try: os.remove(out_path)
        except OSError: pass

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/voices/openai")
def voices_openai(_=Depends(require_api_key)):
    try:
        return {"voices": get_openai().list_voices()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"OpenAI voices error: {e}")

@app.get("/voices/google")
def voices_google(_=Depends(require_api_key)):
    try:
        return {"voices": get_google().list_voices()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Google voices error: {e}")

@app.post("/tts/openai")
def tts_openai(payload: TTSIn, _=Depends(require_api_key)):
    try:
        data = _synthesize_to_mp3_bytes(get_openai(), payload.text, payload.voice)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"OpenAI TTS error: {e}")
    return StreamingResponse(io.BytesIO(data), media_type="audio/mpeg",
                             headers={"Content-Disposition": f'inline; filename="openai-{(payload.voice or "default")}.mp3"'})

@app.post("/tts/google")
def tts_google(payload: TTSIn, _=Depends(require_api_key)):
    try:
        data = _synthesize_to_mp3_bytes(get_google(), payload.text, payload.voice)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Google TTS error: {e}")
    return StreamingResponse(io.BytesIO(data), media_type="audio/mpeg",
                             headers={"Content-Disposition": f'inline; filename="google-{(payload.voice or "default")}.mp3"'})
