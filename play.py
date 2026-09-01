#!/usr/bin/env python
"""
Play against a trained agent.

    python play.py                    # choose a difficulty tier
    python play.py --model Champion   # or jump straight in
    python play.py --list             # what has been trained
    python play.py --scripted         # play the built-in baseline instead
"""

import argparse
import sys

from src.registry import ModelRegistry


def list_models():
    entries = ModelRegistry().entries()
    if not entries:
        print("Nothing trained yet. Run:  python train.py")
        return
    header = f"{'id':20s} {'tier':12s} {'win rate':>9s} {'worst':>7s} {'steps':>12s}"
    print(header)
    print("-" * len(header))
    for e in entries:
        m = e["metrics"]
        print(f"{e['id']:20s} {e['tier']:12s} {m['win_rate']:>9.3f} "
              f"{m['worst']:>7.3f} {e['steps']:>12,}")
    from src.registry import CEILING
    print("\n  win rate = points won against a fixed ladder of scripted opponents")
    print("  worst    = win rate against the single hardest of them")
    print(f"  for scale: an untrained agent scores about 0.11, and {CEILING:.3f} is the")
    print("  ceiling — what a flawless returner scores against this same ladder")


def main():
    p = argparse.ArgumentParser(description="Play the paddle duel")
    p.add_argument("--model", help="checkpoint id, generation number, or tier name")
    p.add_argument("--list", action="store_true")
    p.add_argument("--scripted", action="store_true", help="face the built-in baseline")
    p.add_argument("--right", action="store_true", help="play the right paddle")
    p.add_argument("--first-to", type=int, default=7)
    p.add_argument("--smoothing", type=int, default=5,
                   help="frames a new direction must persist before the "
                        "opponent obeys it. 0 for the raw, jittery paddle")
    args = p.parse_args()

    if args.list:
        list_models()
        return

    from src.game import Game, select
    from src.pong import LEFT, RIGHT

    from src.opponents import Smoothed

    if args.scripted:
        from src.opponents import Predictor
        opponent, label = Predictor(), "built-in baseline"
    else:
        registry = ModelRegistry()
        if args.model:
            entry = registry.find(args.model)
            if entry is None:
                print(f"No checkpoint matching {args.model!r}. Try: python play.py --list")
                sys.exit(1)
        else:
            entries = registry.difficulty_menu()
            if not entries:
                print("Nothing trained yet — facing the built-in baseline instead.")
                print("Train one with:  python train.py")
                from src.opponents import Predictor
                Game(Predictor(), "built-in baseline",
                     RIGHT if args.right else LEFT, args.first_to).run()
                return
            entry = select(entries)
            if entry is None:
                return
        from src.opponents import PolicyOpponent
        opponent = PolicyOpponent(entry["path"], deterministic=True)
        label = f"{entry['tier']} · generation {entry['generation']}"

    # Smoothed for play: the raw paddle changes direction ~22 times a second,
    # which reads as jitter rather than as an opponent. It also plays slightly
    # better this way, because the flicker was wasted movement.
    opponent = Smoothed(opponent, patience=args.smoothing)
    Game(opponent, label, RIGHT if args.right else LEFT, args.first_to).run()


if __name__ == "__main__":
    main()
