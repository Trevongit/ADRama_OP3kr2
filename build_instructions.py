def build_instructions(meta: dict) -> str:
    """Construct a concise instructions string from ADRama line metadata."""
    bits = []
    if meta.get("voice_style"): bits.append(meta["voice_style"])
    if meta.get("voice_emotion"): bits.append(f"emotion: {meta['voice_emotion']}")
    if meta.get("micro_emotion"): bits.append(f"micro: {meta['micro_emotion']}")
    if meta.get("pace"): bits.append(f"pace: {meta['pace']}")
    if meta.get("vocal_technique"): bits.append(f"technique: {meta['vocal_technique']}")
    if meta.get("accent_hint"): bits.append(f"accent: {meta['accent_hint']}")
    if meta.get("rhythm_signature"): bits.append(f"rhythm: {meta['rhythm_signature']}")
    return ("; ".join(bits) + ".") if bits else ""
