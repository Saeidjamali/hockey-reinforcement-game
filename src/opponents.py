"""
Opponents: scripted baselines for measuring, and past selves for training.

The scripted set never changes, so a win rate against it is comparable across
every checkpoint ever trained. It also spans a difficulty ladder, which is what
lets "has it actually got better?" be a number rather than an impression.
"""

import numpy as np

from src.config import HEIGHT, PADDLE_HALF, PADDLE_SPEED
from src.pong import PongSim, other

_POLICY_CACHE = {}


def load_policy(path):
    if path not in _POLICY_CACHE:
        from stable_baselines3 import PPO

        policy = PPO.load(path, device="cpu").policy
        policy.set_training_mode(False)
        _POLICY_CACHE[path] = policy
    return _POLICY_CACHE[path]


def clear_policy_cache(path=None):
    """Always call this if a checkpoint file is replaced or deleted: the cache
    is keyed on path, so a reused path would keep serving the old policy."""
    if path is None:
        _POLICY_CACHE.clear()
    else:
        _POLICY_CACHE.pop(path, None)


def _toward(current, target, dead=PADDLE_SPEED / 2):
    if abs(target - current) < dead:
        return 1
    return 2 if target > current else 0


class Sitter:
    """Never moves. The floor: anything that loses to this has learned nothing."""

    name = "sitter"

    def reset(self, rng=None):
        pass

    def act(self, sim, side):
        return 1


class Chaser:
    """Follows the ball's current height. Beatable by anything that bounces."""

    name = "chaser"

    def __init__(self, lag=0):
        self.lag = lag
        self._seen = []

    def reset(self, rng=None):
        self._seen = []

    def act(self, sim, side):
        self._seen.append(sim.ball_y)
        target = self._seen[max(0, len(self._seen) - 1 - self.lag)]
        return _toward(sim.paddle[side], target)


class Predictor:
    """Works out where the ball will arrive, bounces included. Strong."""

    name = "predictor"

    def __init__(self, error=0.0, rng=None):
        self.error = error
        self.rng = rng or np.random.default_rng()
        self._leg = None
        self._offset = 0.0

    def reset(self, rng=None):
        if rng is not None:
            self.rng = rng
        self._leg = None
        self._offset = 0.0

    def arrival(self, sim, side):
        frames = sim.frames_until(side)
        if frames is None:
            return HEIGHT / 2
        span = HEIGHT - 2 * 8.0
        raw = sim.ball_y - 8.0 + sim.ball_vy * frames
        folded = raw % (2 * span)
        if folded > span:
            folded = 2 * span - folded
        return folded + 8.0

    def act(self, sim, side):
        # The misjudgement is drawn once per incoming ball, not once per frame.
        # Re-drawing every frame let the noise average away over the ~80 frames
        # of flight: a 25px error landed 3.4px from the true arrival, well
        # inside a 50px paddle, so every "sloppy" predictor played perfectly.
        leg = (sim.rally, int(np.sign(sim.ball_vx)))
        if leg != self._leg:
            self._leg = leg
            self._offset = float(self.rng.normal(0, self.error)) if self.error else 0.0
        return _toward(sim.paddle[side], self.arrival(sim, side) + self._offset)


class LiveAgent:
    """Wraps a policy that is still training, so it can play a full point.

    Lives here rather than in train.py: importing it from the entry point
    re-executes that module, which re-runs its one-time torch thread setup and
    crashes.
    """

    name = "training"

    def __init__(self, model):
        self.model = model

    def reset(self, rng=None):
        pass

    def act(self, sim, side):
        action, _ = self.model.predict(sim.observe(side)[None, :], deterministic=False)
        return int(action[0])


class PolicyOpponent:
    """A frozen checkpoint playing the other side."""

    def __init__(self, path, deterministic=False):
        self.path = path
        self.name = f"policy:{path}"
        self.deterministic = deterministic
        self.policy = load_policy(path)

    def reset(self, rng=None):
        pass

    def act(self, sim, side):
        action, _ = self.policy.predict(
            sim.observe(side)[None, :], deterministic=self.deterministic
        )
        return int(action[0])


def baseline_ladder(rng=None):
    """Fixed opponents, weakest first. Never change these."""
    rng = rng or np.random.default_rng(0)
    return [
        Sitter(),
        Chaser(lag=12),
        Chaser(lag=4),
        Chaser(lag=0),
        Predictor(error=60.0, rng=rng),
        Predictor(error=25.0, rng=rng),
        Predictor(error=0.0, rng=rng),
    ]


BASELINE_NAMES = ["sitter", "chaser_slow", "chaser_lagged", "chaser",
                  "predictor_sloppy", "predictor_ok", "predictor"]


class Smoothed:
    """Stops a paddle flickering, without changing what it is trying to do.

    The paddle moves in discrete 7px steps at 60fps, so once it is near its
    target it overshoots and corrects every frame — roughly 22 direction
    changes a second, which reads as a jitter rather than as play. This
    requires a new direction to be chosen `patience` frames running before it
    is obeyed, so brief flicker is ignored and genuine moves are not.

    Used for playing and watching only. Training and evaluation see the raw
    policy, so measured strength means the same thing as it always did.
    """

    def __init__(self, inner, patience=5):
        self.inner = inner
        self.patience = max(1, int(patience))
        self.name = getattr(inner, "name", "?")
        self.reset()

    def reset(self, rng=None):
        if hasattr(self.inner, "reset"):
            self.inner.reset(rng)
        self._current = 1                      # hold
        self._candidate = 1
        self._streak = 0

    def act(self, sim, side):
        wanted = self.inner.act(sim, side)
        if wanted == self._current:
            self._streak = 0
            return self._current
        if wanted == self._candidate:
            self._streak += 1
        else:
            self._candidate, self._streak = wanted, 1
        if self._streak >= self.patience:
            self._current, self._streak = self._candidate, 0
        return self._current


class OpponentPool:
    """Samples who to play each episode: a past self, or a scripted baseline."""

    def __init__(self, checkpoints=None, scripted_rate=0.3, recency=1.5, rng=None):
        self.checkpoints = list(checkpoints or [])
        self.base_scripted_rate = scripted_rate
        self.scripted_rate = 1.0 if not self.checkpoints else scripted_rate
        self.recency = recency
        self.rng = rng or np.random.default_rng()
        self._scripted = baseline_ladder(self.rng)

    def sample(self):
        if self.rng.random() < self.scripted_rate or not self.checkpoints:
            pick = self._scripted[int(self.rng.integers(len(self._scripted)))]
            pick.reset(self.rng)
            return pick
        n = len(self.checkpoints)
        weights = np.arange(1, n + 1, dtype=float) ** self.recency
        weights /= weights.sum()
        return PolicyOpponent(self.checkpoints[int(self.rng.choice(n, p=weights))])
