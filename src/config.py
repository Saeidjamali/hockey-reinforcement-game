"""Constants for the paddle duel."""

import os

WIDTH = 800
HEIGHT = 600

PADDLE_X = 40.0            # distance of each paddle from its wall
PADDLE_HALF = 50.0         # half the paddle's height
PADDLE_SPEED = 7.0

# Paddle speed and the ball's top speed together decide whether a point can be
# won at all. A paddle reaches `flight x speed` of the wall, so at the serve
# speed it covers the wall twice over and even a perfect predictor cannot be
# beaten — which is exactly what stalled the first training run: three of the
# seven baselines were unbeatable and capped the agent's win rate near 0.47.
# By the end of a rally the ball is fast enough that reach falls to ~0.6 of the
# wall, so late exchanges are decided by aiming rather than by reflexes.
BALL_RADIUS = 8.0
BALL_SPEED = 9.0           # serve speed
BALL_ACCEL = 1.08          # multiplier per strike, so rallies end
BALL_SPEED_CAP = 4.0       # as a multiple of the serve speed
MAX_BOUNCE_DEG = 60.0      # steepest return, from the horizontal

# One episode is one point.
MAX_RALLY_FRAMES = 3000

# Reward: winning the point, and nothing else.
#
# There was a +0.05 bonus for returning the ball, meant to give a policy that
# had never touched it something to climb. But that is a direct hint about the
# object of the game — it tells the agent that hitting the ball back is good,
# which is exactly the thing it should have to work out. It is off by default;
# `--return-bonus` puts it back if a run cannot get started without it.
#
# Sparse reward is viable here because the opponent ladder includes a paddle
# that never moves, so even a flailing policy wins some points and has a real
# +1 to learn from.
REWARD_POINT = 1.0
REWARD_RETURN = float(os.environ.get("PADDLE_RETURN_BONUS", "0.0"))

MODELS_DIR = os.environ.get("PADDLE_MODELS_DIR", "models")
REGISTRY_PATH = os.path.join(MODELS_DIR, "registry.json")

OPPONENT_HISTORY = 4       # how many of their past returns the agent can see
