"""Generate seeds/memories.json (50+ memories) and seeds/long_conversation.json.

Deterministic: running it twice produces identical files. Used by the CLI
(--seed) and by the context-budget tests.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEEDS = ROOT / "seeds"

SERVICES = ["plumber", "electrician", "ac servicing", "cleaning", "carpenter", "pest control", "appliance repair"]


def build_memories() -> list[dict]:
    rng = random.Random(7)
    mems: list[dict] = []

    def add(key, value, kind, confirmed, status="active", tags=None, priority=5, source=""):
        mems.append({"id": f"mem-{len(mems)+1:04d}", "key": key, "value": value, "kind": kind, "confirmed": confirmed,
                     "status": status, "tags": tags or ["general"], "source": source,
                     "created_at": "2026-10-01T09:00:00", "updated_at": "2026-10-01T09:00:00", "priority": priority})

    # Explicit, active, general (should reach the model)
    add("preferred_area", "Koramangala", "explicit", True, priority=1, source="user: I live in Koramangala")
    add("earliest_time", "10:00", "explicit", True, priority=1, source="user: never before 10 am")
    add("default_budget", "2000", "explicit", True, priority=2, source="user: keep it under 2000 unless I say")
    add("contact_preference", "WhatsApp before arrival", "explicit", True, priority=3)
    add("home_type", "2BHK, 3rd floor, no lift", "explicit", True, priority=4)
    add("pets", "one dog - technicians should be okay with dogs", "explicit", True, priority=4)
    # Explicit, service-tagged
    add("preferred_provider:electrician", "Ramesh Fix-It (electrician-02)", "explicit", True, tags=["electrician"], priority=2)
    add("preferred_provider:plumber", "any verified plumber", "explicit", True, tags=["plumber"], priority=3)
    add("ac_model", "Daikin 1.5T split, 2 units", "explicit", True, tags=["ac servicing"], priority=3)
    add("cleaning_scope", "kitchen + 2 bathrooms, no balcony", "explicit", True, tags=["cleaning"], priority=3)
    # Inferred & confirmed (eligible)
    add("prefers_upi", "yes", "inferred", True, priority=5, source="paid by UPI 3 times")
    add("typical_day", "weekday mornings", "inferred", True, priority=5)
    # Inferred & unconfirmed (must NEVER reach the model)
    add("guess_second_home", "Whitefield", "inferred", False, priority=5, source="searched Whitefield once")
    add("guess_budget_flexible", "will go 20% over budget", "inferred", False, priority=5)
    add("guess_dislikes_provider", "plumber-03", "inferred", False, tags=["plumber"], priority=5)
    add("guess_vegetarian", "yes", "inferred", False, priority=9)
    # Forgotten (must NEVER reach the model)
    add("old_area", "Indiranagar", "explicit", True, status="forgotten", priority=1, source="moved in 2025")
    add("old_phone_brand", "Samsung", "explicit", True, status="forgotten", priority=6)
    add("ex_roommate_pref", "no visits after 7 pm", "explicit", True, status="forgotten", priority=2)
    # Filler: many low-priority explicit facts, some irrelevant to any service, to stress the budget
    hobbies = ["gardening", "cycling", "chess", "baking", "sketching", "running", "yoga", "photography"]
    for i in range(16):
        add(f"note_{i:02d}", f"likes {rng.choice(hobbies)} on weekends ({i})", "explicit", True, priority=8)
    for i, svc in enumerate(SERVICES):
        add(f"last_rating:{svc}", f"{rng.choice([3, 4, 5])} stars", "inferred", True, tags=[svc], priority=6)
        add(f"avg_spend:{svc}", f"₹{rng.randint(600, 2500)}", "inferred", False, tags=[svc], priority=7)
    for i in range(8):
        add(f"stale_{i}", f"stale fact {i}", "inferred", False, status="forgotten", priority=9)
    assert len(mems) >= 50, len(mems)
    return mems


def build_conversation(n_turns: int = 240) -> list[dict]:
    rng = random.Random(11)
    msgs = []
    topics = ["plumber", "electrician", "cleaning", "ac servicing", "carpenter"]
    for i in range(n_turns // 2):
        t = rng.choice(topics)
        msgs.append({"role": "user", "content": f"(old turn {i}) Earlier I asked about a {t} in {rng.choice(['Koramangala', 'HSR Layout'])} "
                                                f"for {rng.choice(['Monday', 'Friday', 'the weekend'])}. {'Lorem ipsum ' * rng.randint(3, 20)}"})
        msgs.append({"role": "assistant", "content": [{"type": "text", "text": f"(old reply {i}) Sure - noted the {t} request. "
                                                                               f"{'Details details ' * rng.randint(3, 20)}"}]})
    return msgs


def main() -> None:
    SEEDS.mkdir(exist_ok=True)
    mems = build_memories()
    (SEEDS / "memories.json").write_text(json.dumps({"memories": mems}, indent=2, ensure_ascii=False), encoding="utf-8")
    conv = build_conversation()
    (SEEDS / "long_conversation.json").write_text(json.dumps(conv, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {len(mems)} memories and {len(conv)} conversation messages to {SEEDS}")


if __name__ == "__main__":
    sys.exit(main())
