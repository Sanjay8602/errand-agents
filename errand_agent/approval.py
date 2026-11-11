"""Approval gate: code-enforced consent for consequential actions.

The model never executes book / cancel / reschedule. When it asks for one, the
orchestrator turns the request into an ApprovalRequest whose fingerprint is a
hash of the exact action and its material details (provider, slot, price,
booking id). The user must explicitly approve *that* request. Execution later
re-computes the fingerprint from live data; any mismatch (price drift, new
slot, different provider) invalidates the approval and a fresh one is required.
"""
from __future__ import annotations

import hashlib
import json
import re
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

CONSEQUENTIAL_TOOLS = ("book", "cancel", "reschedule")

MATERIAL_FIELDS = {
    "book": ("provider_id", "slot", "price"),
    "cancel": ("booking_id",),
    "reschedule": ("booking_id", "slot"),
}

APPROVE_RE = re.compile(
    r"^\s*(yes|y|yeah|yep|yup|sure|ok|okay|confirm|confirmed|approved?|go ahead|do it|book it|proceed|"
    r"please do|that works|sounds good|yes please|go for it|cancel it|reschedule it)\b[.!]*\s*$",
    re.IGNORECASE,
)
REJECT_RE = re.compile(
    r"^\s*(no|nope|nah|don'?t|do not|stop|never ?mind|cancel that|not now|hold on|wait|leave it|skip)\b.*$",
    re.IGNORECASE,
)


def fingerprint(action: str, details: dict[str, Any]) -> str:
    material = {k: details.get(k) for k in MATERIAL_FIELDS[action]}
    blob = json.dumps({"action": action, **material}, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


@dataclass
class ApprovalRequest:
    approval_id: str
    action: str
    details: dict[str, Any]
    fingerprint: str
    summary: str
    status: str = "pending"      # pending | approved | rejected | invalidated | consumed
    history: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ApprovalRequest":
        return cls(**d)


class ApprovalGate:
    def __init__(self, pending: Optional[ApprovalRequest] = None):
        self.pending: Optional[ApprovalRequest] = pending

    # ----------------------------------------------------------- lifecycle
    def request(self, action: str, details: dict[str, Any], summary: str) -> ApprovalRequest:
        if action not in CONSEQUENTIAL_TOOLS:
            raise ValueError(f"{action} does not need approval")
        req = ApprovalRequest(
            approval_id="apr-" + uuid.uuid4().hex[:8],
            action=action,
            details=dict(details),
            fingerprint=fingerprint(action, details),
            summary=summary,
        )
        self.pending = req
        return req

    @staticmethod
    def classify_reply(text: str) -> str:
        """'approve' | 'reject' | 'other'. Deterministic; the model is not consulted."""
        if APPROVE_RE.match(text or ""):
            return "approve"
        if REJECT_RE.match(text or ""):
            return "reject"
        return "other"

    def approve(self) -> ApprovalRequest:
        if not self.pending or self.pending.status != "pending":
            raise PermissionError("no pending approval")
        self.pending.status = "approved"
        self.pending.history.append("approved by user")
        return self.pending

    def reject(self, reason: str = "user declined") -> Optional[ApprovalRequest]:
        if self.pending:
            self.pending.status = "rejected"
            self.pending.history.append(reason)
        req, self.pending = self.pending, None
        return req

    def invalidate(self, reason: str) -> Optional[ApprovalRequest]:
        if self.pending:
            self.pending.status = "invalidated"
            self.pending.history.append(reason)
        req, self.pending = self.pending, None
        return req

    def consume(self) -> None:
        """Mark the approval spent so a repeated 'yes' cannot re-run the action."""
        if self.pending:
            self.pending.status = "consumed"
            self.pending = None

    # ----------------------------------------------------------- enforcement
    def authorize(self, action: str, live_details: dict[str, Any]) -> tuple[bool, str]:
        """The only path that may green-light a side effect.

        Returns (ok, reason). ok is True only when there is an *approved*
        request for this exact action and the live material details still
        match the fingerprint the user approved.
        """
        req = self.pending
        if req is None:
            return False, "no approval on record"
        if req.status != "approved":
            return False, f"approval status is {req.status}"
        if req.action != action:
            return False, f"approval was for {req.action}, not {action}"
        live_fp = fingerprint(action, live_details)
        if live_fp != req.fingerprint:
            changed = [k for k in MATERIAL_FIELDS[action] if live_details.get(k) != req.details.get(k)]
            return False, "material details changed: " + ", ".join(changed)
        return True, "ok"
