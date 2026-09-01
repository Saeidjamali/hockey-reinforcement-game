"""Watching it play: during training, or replaying a saved checkpoint."""

import numpy as np

from src.config import MAX_RALLY_FRAMES
from src.pong import PongSim, LEFT, RIGHT, other
from src.render import Renderer, line, gap, WHITE, DIM, GOLD, MINE, THEIRS


def _bar(fraction, width=16):
    filled = int(round(float(np.clip(fraction, 0, 1)) * width))
    return "█" * filled + "·" * (width - filled)


def show_point(renderer, agent, opponent, side=LEFT, fps=90, panel=(), rng=None):
    """Render one point. Returns (agent_won, quit_requested)."""
    rng = rng or np.random.default_rng()
    sim = PongSim(rng).reset()
    foe = other(side)
    if hasattr(agent, "reset"):
        agent.reset(rng)
    opponent.reset(rng)

    flash = 0.0
    for _ in range(MAX_RALLY_FRAMES):
        if renderer.pump()["quit"]:
            return 0.0, True
        sim.move(side, agent.act(sim, side))
        sim.move(foe, opponent.act(sim, foe))
        result = sim.advance()

        extra = [gap(8),
                 line(f"rally {sim.rally}", DIM),
                 line(f"ball speed {sim.speed:.0f}", DIM)]
        renderer.frame(sim, human_side=side, panel=list(panel) + extra, flash=flash)
        flash = max(0.0, flash - 0.1)
        renderer.tick(fps)

        if result in (LEFT, RIGHT):
            return (1.0 if result == side else 0.0), False
    return 0.5, False


class TrainingViewer:
    """Plays a point with the policy exactly as it stands, mid-training."""

    def __init__(self, fps=110):
        self.fps = fps
        self.renderer = None
        self.enabled = True
        self.history = []

    def show(self, agent, opponent, steps, strength):
        if not self.enabled:
            return
        if self.renderer is None:
            self.renderer = Renderer("Paddle Duel — training")
        self.renderer.caption(f"training — {steps:,} steps")

        panel = [line("TRAINING", WHITE, "lg"),
                 line(f"{steps:,} steps", DIM), gap(8),
                 line("strength", WHITE)]
        if strength is not None:
            panel.append(line(f"now  {_bar(strength)} {strength:.2f}", GOLD))
        for label, value in self.history[-8:]:
            panel.append(line(f"{label:<5} {_bar(value)} {value:.2f}", DIM))
        panel += [gap(10), line(f"vs {getattr(opponent, 'name', '?')}", THEIRS),
                  gap(10), line("ESC stops watching", DIM),
                  line("(training continues)", DIM)]

        won, quit_requested = show_point(self.renderer, agent, opponent,
                                         fps=self.fps, panel=panel)
        self.history.append((f"{steps // 1000}k", strength if strength is not None else won))
        if quit_requested:
            self.enabled = False
            self.renderer.quit()
            self.renderer = None
            print("    (viewer closed — training continues)", flush=True)

    def close(self):
        if self.renderer is not None:
            self.renderer.quit()
            self.renderer = None


class WatchCallback:
    """Bridges the trainer to the viewer without the trainer knowing about pygame."""

    def __init__(self, viewer, every, trainer):
        self.viewer = viewer
        self.every = every
        self.trainer = trainer
        self._next = every
        self.rng = np.random.default_rng(0)
        # duck-typed as an SB3 callback
        self.model = None
        self.training_env = None

    def init_callback(self, model):
        self.model = model

    def on_training_start(self, *_):
        pass

    def on_rollout_start(self):
        pass

    def on_rollout_end(self):
        pass

    def on_training_end(self):
        self.viewer.close()

    def update_locals(self, locals_):
        pass

    def on_step(self):
        steps = self.model.num_timesteps
        if not self.viewer.enabled or steps < self._next:
            return True
        self._next = steps + self.every
        from src.opponents import baseline_ladder, LiveAgent
        ladder = baseline_ladder(self.rng)
        opponent = ladder[int(self.rng.integers(len(ladder)))]
        self.viewer.show(LiveAgent(self.model), opponent, steps, self.trainer.last_rate)
        return True
