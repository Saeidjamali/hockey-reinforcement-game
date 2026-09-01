# Paddle Duel

Two paddles, one ball, identical rules for both sides. An agent starts knowing
nothing about any of it and learns to play by playing — first against scripted
opponents, then against its own past selves.

Every difficulty level you can play is a genuine snapshot of that agent partway
through learning. Rookie is not Pro with its aim turned down; it is the same
network, 10,000 steps in, when it was actually bad.

```bash
pip install -r requirements.txt
```

## Which command does what

Only one of these puts you at the keyboard.

| command | who is playing | you |
|---|---|---|
| `python train.py --watch` | the AI, against scripted opponents | **watch only** — this is a progress view of training, there is no input |
| `python watch.py` | the AI, against an opponent or itself | **watch only** |
| `python play.py` | **you**, against a trained AI | W/S or arrow keys |

`--watch` means "show me what it can do", not "let me play it". You cannot play
against the agent while it is training; train first, then play any checkpoint it
saved.

```bash
python train.py --fresh --watch    # 1. it learns from scratch, you watch
python play.py                     # 2. you play what it became
python watch.py --ladder           # 3. or watch every generation in order
```

## The game

Paddles sit on the left and right walls and move up and down. The ball bounces
off the top and bottom; a point ends when it gets past someone.

**Serving:** the winner of a point serves the next one, so the ball travels
away from them and toward whoever just conceded. A three-second countdown opens
the match; between points there is a short beat with the ball held at the
server's paddle, so you see it start its journey rather than appear mid-court.

The one rule that matters: **the return angle is set by where on your paddle the
ball strikes.** Hitting near the top edge sends it up, near the bottom sends it
down, dead centre sends it flat. So attacking and defending are the same act —
you are not choosing "where to aim", you are choosing *where to intercept*.

The ball speeds up on every strike, up to four times the serve speed. That is
what makes points end, and it changes what the game is about as a rally goes on:

| rally stage | ball speed | how much of the wall a paddle can reach |
|---|---:|---:|
| serve | 9 | 203% — nearly everything comes back |
| middle | 18 | 110% |
| late | 27 | 79% |
| end | 36 | **63% — placement decides it** |

Rallies begin as reflex and end as aim.

## What the agent is told

Nothing about the rules. It sees its own paddle, the opponent's, the ball's
position and velocity, how fast the ball is going, roughly how long until it
arrives, the rally length, and the last four contact points the opponent used.
It has three actions: up, hold, down.

Reward is +1 for winning a point and −1 for losing. Nothing else. There is no
bonus for returning the ball, because that would be telling it the objective —
it has to discover that bouncing the ball back is worth doing. (A
`--return-bonus` flag exists to reintroduce one, and defaults to off.)

Observations are **mirrored**, so a player always sees itself in the same place.
That is what lets one policy play either side, which is what makes training
against its own past selves meaningful.

## Training

```bash
python train.py                  # 3M steps by default
python train.py --fresh          # start over — DELETES every checkpoint
python train.py --resume         # continue from the latest checkpoint
python train.py --watch          # open a window and watch it play as it learns
python train.py --watch --watch-every 10000
python train.py --steps 6000000 --scripted-rate 0.5
python train.py --gamma 0.999    # credit horizon; see "the discount" below
python train.py --list           # what has been trained
```

It begins against a fixed ladder of scripted opponents. A checkpoint is saved
whenever it reaches a new tier *or* simply plays better than anything saved so
far, and each one *joins the opponent pool*, so from then on it also plays
versions of itself. A run ends when its step budget does. `--scripted-rate` sets how much
of its training stays against the fixed opponents (default 0.3).

`--watch` pauses every `--watch-every` steps to play one full point with the
policy exactly as it stands, and shows a running history of its strength beside
it. `ESC` closes the window; training continues. It does not change the outcome:
the viewer draws with its own RNG and only replays the current policy.

**`--fresh` is destructive.** It runs `rmtree` on `models/agent` and removes the
registry. Every tier you can currently play is gone, with no undo and nothing in
git. Copy them first if you want them back:

```bash
cp -r models models_backup
```

### The full run

One command, from nothing to the strongest agent this repo has produced:

```bash
python train.py --fresh --watch --steps 12000000 --scripted-rate 0.5 --gamma 0.999
```

About 40 minutes on a 14-core Apple Silicon machine. Drop `--watch` and it runs
appreciably faster, since it stops to play a point every 50,000 steps.

Training is deterministic: two runs at the same `--seed` produce byte-identical
curves and bit-identical weights, so this is repeatable. It will **not** be a
bit-exact reproduction of the shipped checkpoints, though — those came from
three chained runs, the first two at `gamma 0.995`, using a trainer that has
since changed (it used to halt once it had collected all six tier names). The
command above is the cleaner experiment: the long credit horizon applies from
the first step instead of arriving 4.7M steps in.

## Measuring strength

Win rate against a ladder of scripted opponents that never changes, so the
number is comparable across every checkpoint ever trained.

Nobody can win every point on this ladder. A ball served at full speed can be
reached from anywhere in the court before it arrives, so a perfect returner is
unbeatable until a rally has run long enough for the ball to hit its speed cap.
The reference is therefore not 1.00 but the **ceiling**: what a flawless
predictor scores playing as the agent against this same ladder.

| reference point | win rate |
|---|---:|
| untrained agent | 0.11 |
| naive ball-chaser | 0.51 |
| **ceiling — best any policy could do** | **0.748** |

Tiers are bands over *fraction of that ceiling*, so a tier means "this close to
optimal play" rather than a raw number the game cannot produce:

| tier | of optimal | win rate |
|---|---:|---:|
| Rookie | 0–33% | 0.00+ |
| Amateur | 33–53% | 0.25+ |
| Contender | 53–70% | 0.40+ |
| Pro | 70–84% | 0.52+ |
| Champion | 84–94% | 0.63+ |
| Unbeatable | 94%+ | 0.70+ |

Re-derive the ceiling any time with `registry.measure_ceiling()`. Checkpoints
are append-only — nothing is ever overwritten, so a version you enjoy playing
stays playable forever.

## Playing

```bash
python play.py                    # menu of every tier trained so far
python play.py --model Pro        # or by tier name
python play.py --model agent_gen_0004
python play.py --right            # play the right paddle instead
python play.py --first-to 11
python play.py --scripted         # face the built-in baseline
python play.py --list
```

W/S or the arrow keys. First to seven.

The paddle you face is *smoothed*: a raw policy changes direction about 22 times
a second, which reads as jitter rather than as an opponent. A new direction has
to persist five frames before it is obeyed. This is presentation only — training
and evaluation see the raw policy, so measured strength keeps its meaning — and
it costs nothing, because the flicker was wasted movement and removing it makes
the agent play slightly *better*. Tune with `--smoothing 3`, or `--smoothing 0`
for the raw paddle.

## Watching

```bash
python watch.py                   # the best agent against the baseline
python watch.py --ladder          # every generation in order, weakest first
python watch.py --self-play       # the best agent against itself
python watch.py --model Contender --opponent chaser
python watch.py --list-opponents
```

`--ladder` is the clearest view of what was learned: the same agent across every
generation, back to back.

## Hardware

A CPU job by design. The policy is a ~50,000-parameter MLP, so Metal/MPS
kernel-launch overhead costs more than its matrix multiplies save, and the
simulator is the real bottleneck anyway. Every worker is pinned to a single
thread so they never contend, and the win comes from running many of them.

Measured on a 14-core Apple Silicon machine:

| workers | steps/sec |
|---:|---:|
| 6 | 14,059 |
| 12 | 16,952 |
| 16 | 19,608 |

Throughput keeps rising past the core count, because each worker spends much of
its time waiting on a pipe rather than computing — reserving cores for the OS
cost about 15%. `--n-envs` therefore defaults to one worker per core. A 6M-step
run takes about 15 minutes.

## Results so far

Three runs at `--scripted-rate 0.5`, about 40 minutes total on an M-series
laptop: one from scratch, then two resumed — the third with `--gamma 0.999`.

| tier | steps | win rate | of optimal | worst rung |
|---|---:|---:|---:|---:|
| Rookie | 10k | 0.133 | 18% | 0.00 |
| Amateur | 250k | 0.255 | 34% | 0.00 |
| Contender | 330k | 0.429 | 57% | 0.01 |
| Pro | 610k | 0.530 | 71% | 0.13 |
| Champion | 1.09M | 0.638 | 85% | 0.19 |
| Champion | 4.49M | 0.642 | 86% | 0.13 |
| Champion | 4.69M | 0.660 | 88% | 0.12 |
| Champion | 8.61M | **0.685** | **92%** | 0.21 |

Only the last row ran at `gamma 0.999`; everything above it at 0.995.

Every figure is re-scored over 300 points on seeds the agent was not selected
on. That matters: the cheap 24-point probe that fires during training banked the
last checkpoint at 0.708 — Unbeatable — and it was really 0.642. The probe goes
off the moment noise carries it over a line, so crossing a threshold is itself
evidence of a lucky sample. Training now re-checks any crossing on more points
and an unseen seed before believing it.

So the result is **Champion, about 92% of optimal play**. Unbeatable starts at
0.703 and it has not got there.

Every number above is 300 points on seeds 1-3, and the protocol matters: this
same checkpoint scores 0.685 on those seeds and 0.703 on seeds 21-23. About
±0.01 of seed noise, which is enough to cross a tier line — so a figure quoted
without its seeds is how you end up believing a checkpoint is better than it is.

### What was in the way

Four things, found by measurement rather than guesswork. Three were in the
measuring stick; the fourth was the learner.

**The ladder had no top.** `Predictor(error=...)` re-drew its aiming error every
frame, so the noise averaged out over the ~80 frames of ball flight — a 25px
error landed 3.4px from the true arrival, well inside a 50px paddle. All three
"sloppy" predictors were secretly the perfect one. The agent scored 0.00-0.03 on
all three, so improving against them earned nothing and there was no gradient to
climb. Fixed: the misjudgement is drawn once per incoming ball.

**Champion was above the ceiling.** Thresholds were raw win rates set without
checking what the game could produce. A perfect returner scored **0.674** against
the broken ladder while Champion was set at 0.70 — unreachable by anything. Tiers
are now fractions of the measured ceiling.

**Training stopped when it ran out of names.** The callback returned
`len(captured) < len(TIERS)`, so a run halted the moment it had collected six
labels — one stopped at 4.5M of a 12M budget. It now runs the full budget and
banks a checkpoint whenever it plays better than anything banked so far.

**The discount was shorter than a point.** The real one. A point lasts a median
of **708 frames**, but `gamma=0.995` reaches back only 200 — the reward for
winning was discounted to **0.03** by the start of the rally. Anything the agent
did early was worth almost nothing to the optimiser. `--gamma` now defaults to
**0.999**, a 1000-frame horizon that covers a whole point.

That last change moved it from 0.642 to 0.685, and the gain holds at about
+0.04 on either seed set. Worth being precise about *how*, because it was not
the predicted mechanism:

| | returns reachable balls | returns marginal balls | hits unreturnable shots |
|---|---:|---:|---:|
| gamma 0.995 | 94.7% | 10.2% | 2.0% |
| gamma 0.999 | 95.4% | 7.8% | 3.3% |
| perfect returner | 100.0% | 28.5% | 3.8% |

The long horizon taught it **offence**, not defence. Placing a shot pays off a
full exchange later, which a 200-frame horizon cannot see. Its shot placement is
now near optimal. Its defence on balls at the edge of reach is not — 7.8%
against an achievable 28.5% — and that is the remaining gap.

### Where it is still beatable

It drops about 5% of balls it could physically reach, and it lunges rather than
pre-positions. Hit wide, to the corner furthest from where it is standing, and
it will not get there. Closing that means teaching it to move before it knows
where the ball is going, which is a different problem from the one solved here.

## Layout

| path | what it does |
|---|---|
| `src/config.py` | every game constant in one place |
| `src/pong.py` | the simulation: physics, and what each side sees |
| `src/env.py` | Gymnasium wrapper; one episode is one point |
| `src/opponents.py` | scripted baselines, past selves, checkpoint loading |
| `src/evaluate.py` | win rate against the fixed ladder |
| `src/registry.py` | append-only checkpoint record and difficulty tiers |
| `src/render.py` | the one place that knows how the game looks |
| `src/game.py` | playing against a checkpoint |
| `src/viewer.py` | watching, during training or afterwards |
| `train.py` / `play.py` / `watch.py` | entry points |
| `reports/negative_results/` | why the previous game was abandoned |

## Before this

This started as an asymmetric game — a turret shooting, a player dodging — and a
lot of work went into the learner before anyone asked whether the game rewarded
the thing being trained for. It did not: the turret's best strategy was to herd
the player into a wall, which needs no perception at all, so no amount of
training could produce an opponent that *read* you.

`reports/negative_results/` has the measurements and the run logs. It is worth a
look before changing this game's mechanics, because the same trap is easy to
walk into twice.
