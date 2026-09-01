"""How strong is this agent? Win rate against opponents that never change."""

import numpy as np

from src.config import MAX_RALLY_FRAMES
from src.pong import PongSim, LEFT, RIGHT, other
from src.opponents import baseline_ladder, BASELINE_NAMES, PolicyOpponent


def play_point(agent, opponent, side, rng, sim=None):
    """1.0 if the agent wins the point, 0.0 if it loses, 0.5 if it never ends."""
    sim = (sim or PongSim(rng)).reset()
    foe = other(side)
    if hasattr(agent, "reset"):
        agent.reset(rng)
    opponent.reset(rng)

    for _ in range(MAX_RALLY_FRAMES):
        sim.move(side, agent.act(sim, side))
        sim.move(foe, opponent.act(sim, foe))
        result = sim.advance()
        if result in (LEFT, RIGHT):
            return 1.0 if result == side else 0.0
    return 0.5


def evaluate(agent, points=24, seed=0, opponents=None):
    """Win rate against each fixed baseline, and overall."""
    rng = np.random.default_rng(seed)
    ladder = opponents or baseline_ladder(rng)
    names = BASELINE_NAMES if opponents is None else [
        getattr(o, "name", f"opp{i}") for i, o in enumerate(opponents)
    ]

    per_opponent = {}
    for name, opponent in zip(names, ladder):
        # Play both sides equally: a policy that is only good from one side is
        # not good, and mirrored observations mean it should not matter.
        scores = [play_point(agent, opponent, LEFT if i % 2 == 0 else RIGHT, rng)
                  for i in range(points)]
        per_opponent[name] = float(np.mean(scores))

    return {
        "win_rate": round(float(np.mean(list(per_opponent.values()))), 4),
        "worst": round(float(min(per_opponent.values())), 4),
        "per_opponent": {k: round(v, 3) for k, v in per_opponent.items()},
    }


def evaluate_checkpoint(path, points=24, seed=0, league=()):
    metrics = evaluate(PolicyOpponent(path), points=points, seed=seed)
    if league:
        recent = [PolicyOpponent(p) for p in list(league)[-3:]]
        result = evaluate(PolicyOpponent(path), points=max(8, points // 2),
                          seed=seed, opponents=recent)
        metrics["vs_league"] = round(result["win_rate"], 3)
    return metrics


class RandomAgent:
    name = "random"

    def __init__(self, rng=None):
        self.rng = rng or np.random.default_rng()

    def reset(self, rng=None):
        if rng is not None:
            self.rng = rng

    def act(self, sim, side):
        return int(self.rng.integers(3))
