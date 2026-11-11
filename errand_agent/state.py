"""Explicit workflow state.

The workflow is a small state machine; every transition is traced. The task
slots are a plain dataclass so "collect only missing details" is a function of
data, not of what the model remembers.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Optional


class WorkflowState(str, Enum):
    IDLE = "IDLE"                          # nothing in flight
    COLLECTING = "COLLECTING"              # gathering missing task details
    SEARCHING = "SEARCHING"                # tools being called to find / hold / inspect
    AWAITING_APPROVAL = "AWAITING_APPROVAL"  # a consequential action is pending explicit user approval
    EXECUTING = "EXECUTING"                # approved action running (book / cancel / reschedule)
    VERIFYING = "VERIFYING"                # ambiguous outcome; consulting source of truth
    COMPLETED = "COMPLETED"                # action done and verified
    FAILED = "FAILED"                      # clear terminal failure reported to user


# Allowed transitions. Anything else is a bug and raises.
TRANSITIONS: dict[WorkflowState, set[WorkflowState]] = {
    WorkflowState.IDLE: {WorkflowState.COLLECTING, WorkflowState.SEARCHING, WorkflowState.AWAITING_APPROVAL,
                         WorkflowState.FAILED, WorkflowState.IDLE},
    WorkflowState.COLLECTING: {WorkflowState.COLLECTING, WorkflowState.SEARCHING, WorkflowState.IDLE,
                               WorkflowState.FAILED, WorkflowState.AWAITING_APPROVAL},
    WorkflowState.SEARCHING: {WorkflowState.SEARCHING, WorkflowState.COLLECTING, WorkflowState.AWAITING_APPROVAL,
                              WorkflowState.IDLE, WorkflowState.FAILED},
    WorkflowState.AWAITING_APPROVAL: {WorkflowState.EXECUTING, WorkflowState.IDLE, WorkflowState.SEARCHING,
                                      WorkflowState.COLLECTING, WorkflowState.AWAITING_APPROVAL, WorkflowState.FAILED},
    WorkflowState.EXECUTING: {WorkflowState.VERIFYING, WorkflowState.COMPLETED, WorkflowState.FAILED,
                              WorkflowState.SEARCHING, WorkflowState.AWAITING_APPROVAL},
    WorkflowState.VERIFYING: {WorkflowState.COMPLETED, WorkflowState.FAILED, WorkflowState.EXECUTING,
                              WorkflowState.SEARCHING},
    WorkflowState.COMPLETED: {WorkflowState.IDLE, WorkflowState.COLLECTING, WorkflowState.SEARCHING,
                              WorkflowState.AWAITING_APPROVAL},
    WorkflowState.FAILED: {WorkflowState.IDLE, WorkflowState.COLLECTING, WorkflowState.SEARCHING},
}


class IllegalTransition(Exception):
    pass


REQUIRED_FOR_BOOKING = ("service", "area", "date_range", "budget_max")


@dataclass
class Task:
    """What the user wants. Only facts the user (or confirmed memory) supplied."""

    intent: Optional[str] = None            # book | reschedule | cancel | inquire | memory
    service: Optional[str] = None
    issue: Optional[str] = None             # e.g. "kitchen tap"
    area: Optional[str] = None
    date_range: Optional[dict] = None       # {"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"}
    time_window: Optional[dict] = None      # {"earliest": "HH:MM", "latest": "HH:MM"}
    budget_max: Optional[int] = None
    preferences: dict[str, Any] = field(default_factory=dict)  # e.g. {"provider_id": "..."} / {"same_as_last": true}
    target_booking_id: Optional[str] = None
    target_slot: Optional[str] = None       # for reschedule
    filled_from_memory: list[str] = field(default_factory=list)

    def missing(self) -> list[str]:
        if self.intent in ("cancel", "reschedule"):
            need = ["target_booking_id"] + (["target_slot"] if self.intent == "reschedule" else [])
            return [f for f in need if getattr(self, f) in (None, "", {})]
        return [f for f in REQUIRED_FOR_BOOKING if getattr(self, f) in (None, "", {})]

    def update(self, **changes: Any) -> list[str]:
        """Apply a partial update; untouched fields are preserved. Returns changed field names."""
        changed = []
        for k, v in changes.items():
            if v is None or not hasattr(self, k):
                continue
            if k == "preferences" and isinstance(v, dict):
                self.preferences.update(v)
                changed.append(k)
                continue
            if getattr(self, k) != v:
                setattr(self, k, v)
                changed.append(k)
        return changed

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Task":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class PendingOperation:
    """A consequential side effect that is (or was) in flight. Persisted before the call."""

    op: str                      # book | cancel | reschedule
    idempotency_key: str
    args: dict
    approval_id: str
    status: str = "in_flight"    # in_flight | done | failed
    result: Optional[dict] = None
