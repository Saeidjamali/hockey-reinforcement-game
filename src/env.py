"""Gymnasium environment: one point of the paddle duel, from one side."""

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.config import (
    MAX_RALLY_FRAMES, REWARD_POINT, REWARD_RETURN,
)
from src.pong import PongSim, ACTIONS, OBS_SIZE, LEFT, RIGHT, other
from src.opponents import OpponentPool


class DuelEnv(gym.Env):
    """One episode is one point.

    The agent is assigned a side at random each episode. Observations are
    mirrored, so it always sees itself in the same place and one policy covers
    both sides — which is what makes training against past selves meaningful.
    """

    metadata = {"render_modes": []}

    def __init__(self, pool=None, seed=None):
        super().__init__()
        self.action_space = spaces.Discrete(ACTIONS)
        self.observation_space = spaces.Box(-1.0, 1.0, (OBS_SIZE,), dtype=np.float32)
        self.rng = np.random.default_rng(seed)
        self.sim = PongSim(self.rng)
        self.pool = pool if pool is not None else OpponentPool(rng=self.rng)
        self.side = LEFT
        self.opponent = None

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self.sim.rng = self.rng
            self.pool.rng = self.rng
        super().reset(seed=seed)

        self.sim.reset()
        self.side = LEFT if self.rng.random() < 0.5 else RIGHT
        self.opponent = self.pool.sample()
        self.opponent.reset(self.rng)
        return self.sim.observe(self.side), {}

    def add_checkpoint(self, path):
        """Add a past self to the pool mid-run, so the league grows as the
        agent does without tearing down and rebuilding the workers."""
        self.pool.checkpoints.append(path)
        self.pool.scripted_rate = self.pool.base_scripted_rate

    def step(self, action):
        sim = self.sim
        foe = other(self.side)

        sim.move(self.side, int(action))
        sim.move(foe, self.opponent.act(sim, foe))

        before = sim.returns[self.side]
        result = sim.advance()

        reward = REWARD_RETURN if sim.returns[self.side] > before else 0.0
        done = False
        info = {}

        if result in (LEFT, RIGHT):
            won = result == self.side
            reward += REWARD_POINT if won else -REWARD_POINT
            done = True
            info = {"won": float(won), "rally": sim.rally,
                    "opponent": getattr(self.opponent, "name", "?")}
        elif sim.frame >= MAX_RALLY_FRAMES:
            done = True
            info = {"won": 0.5, "rally": sim.rally,
                    "opponent": getattr(self.opponent, "name", "?")}

        return sim.observe(self.side), reward, done, False, info
