# Paddle Duel

Two paddles, one ball, identical rules for both sides. An agent starts knowing
nothing about any of it and learns to play by playing — first against scripted
opponents, then against its own past selves.

Every difficulty level you can play is a genuine snapshot of that agent partway
through learning. Rookie is not Pro with its aim turned down; it is the same
network, 50,000 steps in, when it was actually bad.

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
the match; between points there is a short beat with the ball held at the centre,
so a serve is never sprung on you.

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

Reward is +1 for winning a point and −1 for losing, plus a small bonus for
returning the ball at all — that last part only so a policy that has never
touched the ball has something to climb.

Observations are **mirrored**, so a player always sees itself in the same place.
That is what lets one policy play either side, which is what makes training
against its own past selves meaningful.

## Training

```bash
python train.py                  # 3M steps by default
python train.py --fresh          # start over
python train.py --resume         # continue from the latest checkpoint
python train.py --watch          # open a window and watch it play as it learns
python train.py --watch --watch-every 10000
python train.py --steps 6000000 --scripted-rate 0.5
python train.py --list           # what has been trained
```

It begins against a fixed ladder of scripted opponents. Each time it reaches a
new difficulty tier, that checkpoint is saved *and joins the opponent pool*, so
from then on it also plays versions of itself. `--scripted-rate` sets how much
of its training stays against the fixed opponents (default 0.3).

`--watch` pauses every `--watch-every` steps to play one full point with the
policy exactly as it stands, and shows a running history of its strength beside
it. `ESC` closes the window; training continues.

## Measuring strength

Win rate against a ladder of scripted opponents that never changes, so the
number is comparable across every checkpoint ever trained.

Nobody can win every point on this ladder. A ball served at full speed can be
reached from anywhere in the court before it arrives, so a perfect returner is
unbeatable until a rally has run long enough for the ball to hit its speed cap.
The honest reference is therefore not 1.00 but the **ceiling**: what a flawless
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

## Results so far, honestly

| tier | win rate | steps |
|---|---:|---:|
| Rookie | 0.125 | 50k |
| Amateur | 0.250 | 200k |
| Contender | 0.402 | 400k |
| Pro | 0.580 | 2.1M |

It climbed from 0.10 to 0.58 and then stopped. Over the last 4M steps strength
oscillated between 0.38 and 0.61 without trending, and it never reached Champion
(0.70) or the perfect tracker's 0.84.

The reason is visible in the logs: its win rate *against its own training pool*
stayed near 0.65 the entire time. Self-play keeps handing it opponents of its
own strength, so it never had to solve the hard scripted opponents — 70% of its
episodes were against past selves. The untested fix is `--scripted-rate 0.6`,
which makes the fixed ladder the majority of training rather than the minority.

## Layout

| path | what it does |
|---|---|
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
