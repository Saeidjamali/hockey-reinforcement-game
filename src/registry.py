"""
Append-only record of every checkpoint ever trained.

Nothing here is deleted or overwritten. A generation that turns out to be the
most fun to play stays playable forever, and the difficulty tiers come from a
measured win rate rather than a guess.
"""

import json
import os
import zipfile
from datetime import datetime, timezone

from src.config import REGISTRY_PATH

# Nobody can win every point on this ladder. A ball served at full speed can be
# reached from anywhere in the court before it arrives, so a perfect returner is
# unbeatable until a rally is long enough for the ball to hit its speed cap.
# CEILING is what a perfect predictor scores playing as the agent against this
# same ladder: the best any policy could do. Re-derive it with measure_ceiling().
CEILING = 0.748

# Bands over *fraction of that ceiling*, so a tier means "this close to optimal
# play" rather than a raw number the game cannot produce. Scoring the ceiling
# itself is Unbeatable; an untrained policy is around 0.14 of it, so Rookie is
# where everything starts.
TIERS = [
    ("Rookie", 0.00, 0.33),
    ("Amateur", 0.33, 0.53),
    ("Contender", 0.53, 0.70),
    ("Pro", 0.70, 0.84),
    ("Champion", 0.84, 0.94),
    ("Unbeatable", 0.94, 1.01),
]


def tier_for(win_rate):
    fraction = float(win_rate) / CEILING
    for name, lo, hi in TIERS:
        if lo <= fraction < hi:
            return name
    return TIERS[-1][0]


def tier_threshold(name):
    """The raw win rate a tier starts at, for progress reporting."""
    for tier, lo, _ in TIERS:
        if tier == name:
            return round(lo * CEILING, 3)
    return None


def measure_ceiling(points=60):
    """Score a perfect returner as the agent against the ladder it belongs to."""
    from src.opponents import Predictor
    from src.evaluate import evaluate
    return round(evaluate(Predictor(error=0.0), points=points)["win_rate"], 3)


class ModelRegistry:
    def __init__(self, path=None):
        self.path = path or REGISTRY_PATH
        self.data = {"agent": []}
        self.load()

    def load(self):
        if os.path.exists(self.path):
            with open(self.path) as f:
                self.data["agent"] = json.load(f).get("agent", [])
        return self

    def save(self):
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(self.data, f, indent=2)
        return self

    # -----------------------------------------------------------------

    def next_generation(self):
        return max((e["generation"] for e in self.data["agent"]), default=0) + 1

    def checkpoint_path(self, generation):
        """Derived from this registry's own location, never a global constant,
        so a registry pointed elsewhere cannot write into the real one."""
        directory = os.path.join(os.path.dirname(self.path) or ".", "agent")
        os.makedirs(directory, exist_ok=True)
        return os.path.join(directory, f"agent_gen_{generation:04d}.zip")

    def add(self, path, generation, steps, metrics):
        entry = {
            "id": os.path.splitext(os.path.basename(path))[0],
            "generation": generation,
            "path": path,
            "steps": int(steps),
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "metrics": metrics,
            "tier": tier_for(metrics.get("win_rate", 0.0)),
        }
        self.data["agent"].append(entry)
        self.save()
        return entry

    # -----------------------------------------------------------------

    def entries(self):
        """Only checkpoints whose files are actually loadable — existence alone
        is not enough, a truncated file used to be served as valid."""
        alive = [e for e in self.data["agent"]
                 if os.path.exists(e["path"]) and zipfile.is_zipfile(e["path"])]
        return sorted(alive, key=lambda e: e["generation"])

    def checkpoints(self):
        return [e["path"] for e in self.entries()]

    def latest(self):
        entries = self.entries()
        return entries[-1] if entries else None

    def best(self):
        entries = self.entries()
        return max(entries, key=lambda e: e["metrics"].get("win_rate", -1)) if entries else None

    def find(self, key):
        for entry in self.entries():
            if key in (entry["id"], entry["path"], str(entry["generation"])):
                return entry
        matches = [e for e in self.entries() if (e.get("tier") or "").lower() == str(key).lower()]
        return max(matches, key=lambda e: e["metrics"]["win_rate"]) if matches else None

    def difficulty_menu(self):
        """The strongest checkpoint in each tier the run actually reached."""
        by_tier = {}
        for entry in self.entries():
            rate = entry["metrics"].get("win_rate")
            if rate is None:
                continue
            tier = tier_for(rate)
            if tier not in by_tier or rate > by_tier[tier]["metrics"]["win_rate"]:
                by_tier[tier] = entry
        return [by_tier[name] for name, _, _ in TIERS if name in by_tier]
