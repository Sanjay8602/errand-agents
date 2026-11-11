"""Tiny atomic JSON document store used for all persisted state.

Every write goes to a temp file then os.replace(), so a crash mid-write never
leaves a half-written document (relevant for the restart-mid-booking tests).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


class JsonStore:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self, default: Any) -> Any:
        if not self.path.exists():
            return default
        with open(self.path, "r", encoding="utf-8") as f:
            return json.load(f)

    def save(self, data: Any) -> None:
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        os.replace(tmp, self.path)
