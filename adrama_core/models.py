"""Core data models shared between ADRama components."""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional


def sanitize_filename(name: str) -> str:
    """Return a filesystem-safe version of *name* preserving underscores and dashes."""
    return "".join(ch for ch in name if ch.isalnum() or ch in ("_", "-", " ")).strip().replace(" ", "_")


@dataclass
class ScriptLine:
    actor: str
    text: str
    line_code: Optional[str] = None
    metadata: Dict[str, object] = field(default_factory=dict)


@dataclass
class ScriptData:
    lines: List[ScriptLine] = field(default_factory=list)

    @staticmethod
    def from_txt(path: str) -> "ScriptData":
        lines: List[ScriptLine] = []
        with open(path, "r", encoding="utf-8") as f:
            for idx, raw in enumerate(f):
                s = raw.strip()
                if not s:
                    continue
                if ":" in s:
                    actor, dialog = s.split(":", 1)
                    actor, dialog = actor.strip(), dialog.strip()
                else:
                    actor, dialog = "Narrator", s
                code = f"EP-TXT-L{idx + 1:04d}"
                lines.append(ScriptLine(actor=actor, text=dialog, line_code=code))
        return ScriptData(lines=lines)

    @staticmethod
    def from_jsonl(path: str) -> "ScriptData":
        lines: List[ScriptLine] = []
        with open(path, "r", encoding="utf-8") as f:
            for idx, raw in enumerate(f):
                s = raw.strip()
                if not s:
                    continue
                try:
                    obj = json.loads(s)
                    actor = (obj.get("actor") or obj.get("speaker") or obj.get("character") or "Narrator").strip()
                    text = (obj.get("text") or obj.get("line") or obj.get("dialogue") or "").strip()
                    code = (obj.get("line_code") or f"EP-JSONL-L{idx + 1:04d}").strip()
                    if text:
                        meta = obj.copy()
                        lines.append(ScriptLine(actor=actor, text=text, line_code=code, metadata=meta))
                except json.JSONDecodeError:
                    continue
        return ScriptData(lines=lines)

    @staticmethod
    def from_csv(path: str) -> "ScriptData":
        lines: List[ScriptLine] = []
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader):
                actor = (row.get("actor") or row.get("speaker") or row.get("character") or "Narrator").strip()
                text = (row.get("text") or row.get("line") or row.get("dialogue") or "").strip()
                code = (row.get("line_code") or f"EP-CSV-L{idx + 1:04d}").strip()
                if text:
                    meta = {k: v for k, v in row.items() if v is not None}
                    lines.append(ScriptLine(actor=actor, text=text, line_code=code, metadata=meta))
        return ScriptData(lines=lines)

    @staticmethod
    def load_any(path: str) -> "ScriptData":
        extension = os.path.splitext(path)[1].lower()
        if extension == ".txt":
            return ScriptData.from_txt(path)
        if extension == ".jsonl":
            return ScriptData.from_jsonl(path)
        if extension == ".csv":
            return ScriptData.from_csv(path)
        raise ValueError(f"Unsupported file type: {extension}")
