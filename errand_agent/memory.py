"""Persistent user memory with explicit/inferred provenance and forgetting.

Rules enforced here (and proven in tests):
  * Only ACTIVE memories that are EXPLICIT, or INFERRED *and* confirmed, are
    eligible to reach the model.
  * Forgotten memories are kept as tombstones (so we can prove they were not
    sent) but are never retrievable for context.
  * Retrieval is relevance-ranked (service / area / general tags) and token
    budgeted; the caller logs exactly which ids were sent.
"""
from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Optional

from .storage import JsonStore


@dataclass
class Memory:
    id: str
    key: str                      # e.g. "preferred_area", "earliest_time", "preferred_provider:plumber"
    value: Any
    kind: str                     # explicit | inferred
    confirmed: bool               # inferred facts start False
    status: str = "active"        # active | forgotten
    tags: list[str] = field(default_factory=list)   # e.g. ["general"], ["plumber"], ["ac servicing"]
    source: str = ""              # free text: "user said ... on 2026-11-02"
    created_at: str = ""
    updated_at: str = ""
    priority: int = 5             # 1 (highest) .. 9

    def eligible(self) -> bool:
        return self.status == "active" and (self.kind == "explicit" or self.confirmed)

    def render(self) -> str:
        label = "explicit" if self.kind == "explicit" else "inferred, confirmed"
        return f"- {self.key} = {self.value} ({label})"


class MemoryStore:
    def __init__(self, path: Path, now: Optional[callable] = None):
        self.store = JsonStore(Path(path))
        self._now = now or (lambda: datetime.now())
        raw = self.store.load({"memories": []})
        self.items: dict[str, Memory] = {m["id"]: Memory(**m) for m in raw["memories"]}

    # ------------------------------------------------------------------ io
    def _save(self) -> None:
        self.store.save({"memories": [asdict(m) for m in self.items.values()]})

    def _ts(self) -> str:
        return self._now().strftime("%Y-%m-%dT%H:%M:%S")

    # ------------------------------------------------------------- mutations
    def remember(self, key: str, value: Any, kind: str = "explicit", tags: Optional[list[str]] = None,
                 source: str = "", confirmed: Optional[bool] = None, priority: int = 5) -> Memory:
        """Upsert by key. An explicit statement overrides an inferred one and un-forgets the key."""
        if kind not in ("explicit", "inferred"):
            raise ValueError("kind must be explicit or inferred")
        if confirmed is None:
            confirmed = kind == "explicit"
        existing = self.get(key, include_forgotten=True)
        if existing:
            existing.value = value
            existing.kind = kind
            existing.confirmed = confirmed
            existing.status = "active"
            existing.tags = tags or existing.tags or ["general"]
            existing.source = source or existing.source
            existing.updated_at = self._ts()
            existing.priority = priority
            self._save()
            return existing
        m = Memory(id="mem-" + uuid.uuid4().hex[:8], key=key, value=value, kind=kind, confirmed=confirmed,
                   tags=tags or ["general"], source=source, created_at=self._ts(), updated_at=self._ts(),
                   priority=priority)
        self.items[m.id] = m
        self._save()
        return m

    def confirm(self, key: str) -> Optional[Memory]:
        m = self.get(key)
        if m:
            m.confirmed = True
            m.updated_at = self._ts()
            self._save()
        return m

    def forget(self, key_or_id: str) -> Optional[Memory]:
        m = self.items.get(key_or_id) or self.get(key_or_id, include_forgotten=True)
        if m:
            m.status = "forgotten"
            m.updated_at = self._ts()
            self._save()
        return m

    def forget_all(self) -> int:
        n = 0
        for m in self.items.values():
            if m.status == "active":
                m.status = "forgotten"
                n += 1
        self._save()
        return n

    # ---------------------------------------------------------------- reads
    def get(self, key: str, include_forgotten: bool = False) -> Optional[Memory]:
        for m in self.items.values():
            if m.key == key and (include_forgotten or m.status == "active"):
                return m
        return None

    def value(self, key: str) -> Any:
        """Convenience: value of an *eligible* memory, else None."""
        m = self.get(key)
        return m.value if m and m.eligible() else None

    def inspect(self) -> list[dict]:
        """Everything the user is entitled to see, including unconfirmed inferences and tombstones."""
        return [asdict(m) for m in sorted(self.items.values(), key=lambda x: (x.status, x.kind, x.key))]

    def retrieve(self, context_tags: Iterable[str], token_budget: int, estimate_tokens) -> tuple[list[Memory], list[str]]:
        """Return (selected, rejected_reasons). Only eligible + relevant memories, within budget.

        Relevance: a memory is relevant if it carries the 'general' tag or any tag
        in context_tags. Ranking: priority, then explicit before inferred.
        """
        tags = {t.lower() for t in context_tags if t}
        candidates = []
        rejected = []
        for m in self.items.values():
            if m.status != "active":
                rejected.append(f"{m.id}:forgotten")
                continue
            if not m.eligible():
                rejected.append(f"{m.id}:unconfirmed_inferred")
                continue
            if "general" not in m.tags and not tags.intersection(t.lower() for t in m.tags):
                rejected.append(f"{m.id}:irrelevant")
                continue
            candidates.append(m)
        candidates.sort(key=lambda m: (m.priority, 0 if m.kind == "explicit" else 1, m.key))
        selected, used = [], 0
        for m in candidates:
            cost = estimate_tokens(m.render())
            if used + cost > token_budget:
                rejected.append(f"{m.id}:over_budget")
                continue
            selected.append(m)
            used += cost
        return selected, rejected
