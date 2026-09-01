#!/usr/bin/env python
"""
Train the paddle agent by self-play.

It starts knowing nothing — not where the ball goes, not what the paddle does —
and learns by playing. Early on its only opponents are scripted baselines;
as it improves, its own past checkpoints join the pool, so it keeps facing
something roughly its own strength.

    python train.py                 # train, snapshotting every difficulty tier
    python train.py --watch         # ...and watch it play as it learns
    python train.py --fresh         # start over
    python train.py --resume        # continue from the latest checkpoint
    python train.py --list          # what has been trained so far
"""

import argparse
import os
import shutil
import time
from functools import partial

# Apple Silicon: this is a CPU job by design. The policy is a ~50k-parameter
# MLP, so Metal/MPS kernel-launch overhead costs more than its matmuls save,
# and the real bottleneck is the NumPy simulator anyway. The win comes from
# running many environments across the performance cores, so every worker is
# pinned to a single thread and they are never allowed to fight each other for
# one.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

from src.config import MODELS_DIR, REGISTRY_PATH
from src.env import DuelEnv
from src.opponents import OpponentPool, LiveAgent
from src.registry import ModelRegistry, tier_for, tier_threshold, TIERS
from src.evaluate import evaluate, evaluate_checkpoint

torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass                      # already set, or work has started; harmless


def auto_envs():
    """Measured, not assumed: on a 14-core M-series, throughput still rises
    past the core count (6 envs 14.1k steps/s, 12 envs 17.0k, 16 envs 19.6k),
    because each worker spends much of its time waiting on the pipe rather
    than computing. Reserving cores cost throughput, so it does not."""
    return max(2, min(16, os.cpu_count() or 4))


def hardware_note():
    import platform

    bits = [platform.machine(), f"{os.cpu_count()} cores", "torch cpu"]
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        bits.append("mps available but unused: the net is too small to benefit")
    return "  |  ".join(bits)


def _build_env(checkpoints, scripted_rate, seed):
    rng = np.random.default_rng(seed)
    return DuelEnv(pool=OpponentPool(checkpoints, scripted_rate=scripted_rate, rng=rng),
                   seed=seed)


def make_vec(checkpoints, n_envs, scripted_rate, seed, subproc=True):
    fns = [partial(_build_env, checkpoints, scripted_rate, seed + i) for i in range(n_envs)]
    venv = SubprocVecEnv(fns, start_method="spawn") if subproc and n_envs > 1 else DummyVecEnv(fns)
    return VecMonitor(venv)


class Trainer(BaseCallback):
    """Reports progress, snapshots each new tier, and grows the self-play pool."""

    def __init__(self, registry, probe_every, eval_points, report_every=25_000):
        super().__init__()
        self.registry = registry
        self.probe_every = probe_every
        self.eval_points = eval_points
        self.report_every = report_every
        # Probe early the first time. The first checkpoint otherwise does not
        # appear until a full probe interval has passed, which with --watch can
        # be minutes of an apparently empty models directory.
        self._next_probe = min(10_000, probe_every)
        self._next_report = report_every
        self.captured = {e["tier"] for e in registry.entries()}
        self.best_rate = max((e["metrics"].get("win_rate", 0.0)
                              for e in registry.entries()), default=0.0)
        self.margin = 0.02        # ignore noise-sized "improvements"
        self.wins = []
        self.last_rate = None
        self.saved = []

    def _on_step(self):
        for info in self.locals.get("infos", []):
            if "won" in info:
                self.wins.append(info["won"])

        if self.num_timesteps >= self._next_report:
            self._next_report = self.num_timesteps + self.report_every
            window = self.wins[-400:]
            rate = f"{np.mean(window):.2f}" if window else "  - "
            progress = ""
            if self.last_rate is not None:
                # Say what it would take to bank the next checkpoint, so an
                # empty models directory is explicable rather than alarming.
                # TIERS bands are fractions of the measured ceiling, so compare
                # the raw strength against the win rate each tier starts at.
                nxt = next((f"{name} at {tier_threshold(name):.2f}"
                            for name, _, _ in TIERS
                            if name not in self.captured
                            and self.last_rate < tier_threshold(name)), None)
                progress = f"   strength {self.last_rate:.3f}"
                progress += f"   next: {nxt}" if nxt else "   all tiers captured"
            print(f"    {self.num_timesteps:>9,} steps   training win rate {rate}{progress}",
                  flush=True)

        if self.num_timesteps < self._next_probe:
            return True
        self._next_probe = self.num_timesteps + self.probe_every

        # Score the LIVE policy. Never write a file and load it back to measure:
        # checkpoint loading caches by path, so a rejected save would poison
        # that path and freeze progress silently.
        metrics = evaluate(LiveAgent(self.model), points=self.eval_points)
        self.last_rate = metrics["win_rate"]
        tier = tier_for(metrics["win_rate"])

        # Two reasons to bank: a tier never seen before, or simply playing better
        # than anything banked so far. A tier is only a label, and once all six
        # were collected the run used to stop -- so it quit improving the moment
        # it had a full set of names.
        if tier in self.captured and metrics["win_rate"] <= self.best_rate + self.margin:
            return True

        # Winner's curse: the cheap probe fires the moment noise carries it over
        # a threshold, so the crossing itself is evidence of a lucky sample. One
        # run banked 0.708 (Unbeatable) that was really 0.642 (Champion) on fresh
        # episodes. Confirm on more points and an unseen seed before believing it.
        metrics = evaluate(LiveAgent(self.model), points=max(60, self.eval_points * 4),
                           seed=self.num_timesteps % 100_000)
        self.last_rate = metrics["win_rate"]
        tier = tier_for(metrics["win_rate"])
        if tier in self.captured and metrics["win_rate"] <= self.best_rate + self.margin:
            return True

        generation = self.registry.next_generation()
        path = self.registry.checkpoint_path(generation)
        self.model.save(path)
        entry = self.registry.add(path, generation, self.num_timesteps, metrics)
        why = "reached" if tier not in self.captured else "improved"
        self.captured.add(tier)
        self.best_rate = max(self.best_rate, metrics["win_rate"])
        self.saved.append(entry)

        # A new checkpoint joins the pool: from here it also plays its past self.
        self.training_env.env_method("add_checkpoint", path)

        print(f"    {tier:<11} {why:<9}@{self.num_timesteps:>9,} steps  ->  {entry['id']}"
              f"   win rate {metrics['win_rate']:.3f}  worst {metrics['worst']:.3f}",
              flush=True)
        # Always keep going: the step budget ends a run, not a full set of labels.
        return True


def main():
    p = argparse.ArgumentParser(description="Train the paddle agent by self-play")
    p.add_argument("--steps", type=int, default=3_000_000)
    p.add_argument("--n-envs", type=int, default=0,
                   help="0 picks one worker per core, less two")
    p.add_argument("--probe-every", type=int, default=50_000)
    p.add_argument("--eval-points", type=int, default=16)
    p.add_argument("--scripted-rate", type=float, default=0.3)
    p.add_argument("--return-bonus", type=float, default=0.0,
                   help="reward per successful return. 0 by default: a bonus "
                        "for hitting the ball back tells the agent the object "
                        "of the game instead of letting it discover it")
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--gamma", type=float, default=0.999,
                   help="discount. A point lasts ~700 frames, so 0.995 (horizon "
                        "200) discounts the winning reward to 0.03 by the start "
                        "of the rally and the agent cannot learn to pre-position")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--watch", action="store_true", help="show it playing as it learns")
    p.add_argument("--watch-every", type=int, default=50_000)
    p.add_argument("--fresh", action="store_true", help="delete checkpoints and start over")
    p.add_argument("--resume", action="store_true", help="continue from the latest checkpoint")
    p.add_argument("--no-subproc", action="store_true")
    p.add_argument("--list", action="store_true", help="list what has been trained")
    args = p.parse_args()

    if args.list:
        from play import list_models
        list_models()
        return

    if args.return_bonus:
        os.environ["PADDLE_RETURN_BONUS"] = str(args.return_bonus)
        import importlib

        import src.config
        import src.env

        importlib.reload(src.config)
        importlib.reload(src.env)

    if args.fresh:
        shutil.rmtree(os.path.join(MODELS_DIR, "agent"), ignore_errors=True)
        if os.path.exists(REGISTRY_PATH):
            os.remove(REGISTRY_PATH)
        print("cleared existing checkpoints\n")

    n_envs = args.n_envs or auto_envs()
    print(hardware_note())

    registry = ModelRegistry()
    venv = make_vec(registry.checkpoints(), n_envs, args.scripted_rate,
                    args.seed, subproc=not args.no_subproc)

    latest = registry.latest() if args.resume else None
    if latest:
        print(f"resuming from {latest['id']}")
        model = PPO.load(latest["path"], env=venv, device="cpu")
        model.gamma = args.gamma        # load() restores the saved value
    else:
        model = PPO("MlpPolicy", venv, seed=args.seed, verbose=0, device="cpu",
                    policy_kwargs=dict(net_arch=[128, 128]),
                    learning_rate=3e-4, n_steps=256, batch_size=512, n_epochs=10,
                    gamma=args.gamma, gae_lambda=0.95, clip_range=0.2,
                    ent_coef=args.ent_coef, vf_coef=0.5, max_grad_norm=0.5)
    print(f"discount {model.gamma} -> credit reaches back "
          f"{1 / (1 - model.gamma):,.0f} frames; a point lasts about 700")

    from src.evaluate import RandomAgent
    floor = evaluate(RandomAgent(np.random.default_rng(0)), points=12)["win_rate"]
    print(f"an untrained agent wins {floor:.2f} of points against the fixed ladder")
    print(f"training up to {args.steps:,} steps on {n_envs} envs")
    print(f"a checkpoint is saved each time it reaches a new tier "
          f"({', '.join(name for name, _, _ in TIERS)});")
    print(f"strength is measured every {args.probe_every:,} steps\n", flush=True)

    trainer = Trainer(registry, args.probe_every, args.eval_points)
    callbacks = [trainer]
    if args.watch:
        from src.viewer import TrainingViewer, WatchCallback
        callbacks.append(WatchCallback(TrainingViewer(), args.watch_every, trainer))

    started = time.time()
    model.learn(total_timesteps=args.steps, reset_num_timesteps=not latest,
                callback=callbacks, progress_bar=False)
    venv.close()

    print(f"\ndone in {time.time() - started:.0f}s")
    for entry in registry.entries():
        print(f"  {entry['id']:<18} {entry['tier']:<11} win rate {entry['metrics']['win_rate']:.3f}")
    best = registry.best()
    if best:
        print(f"\nplay it:  python play.py --model {best['id']}")


if __name__ == "__main__":
    main()
