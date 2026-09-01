#!/usr/bin/env python
"""
Watch trained agents play, without playing yourself.

    python watch.py                  # the best one against the baseline
    python watch.py --ladder         # every generation in order, weakest first
    python watch.py --model Pro --opponent predictor
    python watch.py --self-play      # the best agent against itself
"""

import argparse
import sys

import numpy as np

from src.registry import ModelRegistry
from src.opponents import baseline_ladder, BASELINE_NAMES, PolicyOpponent, Smoothed


def pick_opponent(name, rng):
    ladder = dict(zip(BASELINE_NAMES, baseline_ladder(rng)))
    if name in ladder:
        opponent = ladder[name]
        opponent.name = name
        return opponent
    return None


def main():
    p = argparse.ArgumentParser(description="Watch the paddle duel")
    p.add_argument("--model")
    p.add_argument("--opponent", default="predictor")
    p.add_argument("--ladder", action="store_true", help="walk every generation")
    p.add_argument("--self-play", action="store_true", help="best agent against itself")
    p.add_argument("--points", type=int, default=3)
    p.add_argument("--fps", type=int, default=70)
    p.add_argument("--smoothing", type=int, default=5,
                   help="frames a new direction must persist before it is obeyed")
    p.add_argument("--list-opponents", action="store_true")
    args = p.parse_args()

    rng = np.random.default_rng(0)
    if args.list_opponents:
        print("opponents:")
        for name in BASELINE_NAMES:
            print(f"  {name}")
        print("  (or --self-play)")
        return

    registry = ModelRegistry()
    entries = registry.entries()
    if not entries:
        print("Nothing trained yet. Run:  python train.py")
        sys.exit(1)

    if args.ladder:
        chosen = entries
    elif args.model:
        entry = registry.find(args.model)
        if entry is None:
            print(f"No checkpoint matching {args.model!r}")
            sys.exit(1)
        chosen = [entry]
    else:
        chosen = [registry.best()]

    from src.render import Renderer, line, gap, WHITE, DIM, THEIRS
    from src.viewer import show_point, _bar

    renderer = Renderer("Paddle Duel — spectating")
    history = []

    for entry in chosen:
        agent = Smoothed(PolicyOpponent(entry["path"], deterministic=True),
                         patience=args.smoothing)
        opponent = (Smoothed(PolicyOpponent(entry["path"], deterministic=True),
                             patience=args.smoothing) if args.self_play
                    else pick_opponent(args.opponent, rng))
        if opponent is None:
            print(f"Unknown opponent {args.opponent!r}; try --list-opponents")
            renderer.quit()
            sys.exit(1)
        label = "itself" if args.self_play else args.opponent
        renderer.caption(f"{entry['id']} ({entry['tier']}) vs {label}")

        won = 0.0
        for i in range(args.points):
            panel = [line("SPECTATING", WHITE, "lg"),
                     line(entry["tier"], THEIRS), gap(4),
                     line(f"generation {entry['generation']}", DIM),
                     line(f"{entry['steps']:,} steps", DIM), gap(10),
                     line(f"vs {label}", THEIRS), gap(10),
                     line("ladder so far", WHITE)]
            for tag, value in history[-8:]:
                panel.append(line(f"{tag:<7} {_bar(value, 12)} {value:.2f}", DIM))
            panel += [gap(10), line(f"point {i + 1}/{args.points}", DIM),
                      line("ESC to stop", DIM)]
            result, quit_requested = show_point(renderer, agent, opponent,
                                                fps=args.fps, panel=panel, rng=rng)
            if quit_requested:
                renderer.quit()
                return
            won += result

        rate = won / args.points
        history.append((f"gen {entry['generation']}", rate))
        print(f"  gen {entry['generation']:>3} [{entry['tier']:>11}]  won {rate:.0%} of {args.points}")

    renderer.quit()


if __name__ == "__main__":
    main()
