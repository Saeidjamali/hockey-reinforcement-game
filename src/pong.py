"""
The paddle duel: two paddles, one ball, identical rules for both sides.

Symmetric on purpose. Both players see the same kind of observation and have
the same three actions, so one policy can play either side and self-play is the
natural way to train it. The agent is told nothing about the rules — where the
ball will go, how it bounces, what the paddle does — it has to find all of it.
"""

import numpy as np

from src.config import (
    WIDTH, HEIGHT, PADDLE_X, PADDLE_HALF, PADDLE_SPEED,
    BALL_RADIUS, BALL_SPEED, BALL_ACCEL, BALL_SPEED_CAP, MAX_BOUNCE_DEG,
    OPPONENT_HISTORY,
)

LEFT, RIGHT = -1, 1
SIDES = (LEFT, RIGHT)

ACTIONS = 3                                   # up, hold, down
MAX_BOUNCE = np.radians(MAX_BOUNCE_DEG)

# my y, my last move, their y, ball x, ball y, ball vx, ball vy, ball speed,
# frames until it reaches me, rally length
OBS_SIZE = 10 + OPPONENT_HISTORY


def other(side):
    return RIGHT if side == LEFT else LEFT


class PongSim:
    """Exact physics. No rendering, no learning."""

    def __init__(self, rng=None):
        self.rng = rng if rng is not None else np.random.default_rng()
        self.reset()

    # -----------------------------------------------------------------

    def reset(self, server=None):
        """Start a point. `server` is the side putting the ball in play.

        The serve leaves the server's paddle, not the middle of the court — the
        ball should come from somewhere, and the receiver gets the full width to
        read it.
        """
        self.speed = BALL_SPEED
        self.paddle = {LEFT: HEIGHT / 2, RIGHT: HEIGHT / 2}

        self.server = server if server is not None else self.rng.choice(SIDES)
        outgoing = other(self.server)
        angle = float(self.rng.uniform(-MAX_BOUNCE / 2, MAX_BOUNCE / 2))
        self.ball_x = self.paddle_x(self.server) + outgoing * (BALL_RADIUS + 1)
        self.ball_y = self.paddle[self.server]
        self.ball_vx = outgoing * self.speed * np.cos(angle)
        self.ball_vy = self.speed * np.sin(angle)
        self.last_move = {LEFT: 0, RIGHT: 0}
        self.contacts = {LEFT: [0.0] * OPPONENT_HISTORY,
                         RIGHT: [0.0] * OPPONENT_HISTORY}
        self.returns = {LEFT: 0, RIGHT: 0}
        self.frame = 0
        self.rally = 0
        return self

    def paddle_x(self, side):
        return PADDLE_X if side == LEFT else WIDTH - PADDLE_X

    # -----------------------------------------------------------------

    def move(self, side, action):
        """action: 0 up, 1 hold, 2 down."""
        step = int(action) - 1
        self.paddle[side] = float(np.clip(
            self.paddle[side] + step * PADDLE_SPEED, PADDLE_HALF, HEIGHT - PADDLE_HALF
        ))
        self.last_move[side] = step

    def advance(self):
        """One frame. Returns None, 'return', or the side that WON the point."""
        self.frame += 1
        self.ball_x += self.ball_vx
        self.ball_y += self.ball_vy

        if self.ball_y < BALL_RADIUS:
            self.ball_y = 2 * BALL_RADIUS - self.ball_y
            self.ball_vy = -self.ball_vy
        elif self.ball_y > HEIGHT - BALL_RADIUS:
            self.ball_y = 2 * (HEIGHT - BALL_RADIUS) - self.ball_y
            self.ball_vy = -self.ball_vy

        side = LEFT if self.ball_vx < 0 else RIGHT
        plane = self.paddle_x(side)
        reached = self.ball_x <= plane if side == LEFT else self.ball_x >= plane
        if not reached:
            return None

        offset = (self.ball_y - self.paddle[side]) / PADDLE_HALF
        if abs(offset) <= 1.0:
            self.strike(side, offset)
            return "return"
        if -BALL_RADIUS < self.ball_x < WIDTH + BALL_RADIUS:
            return None                        # still level with the paddle
        return other(side)                     # it went past: the other side wins

    def strike(self, side, offset):
        """Return the ball. The angle comes from where on the paddle it hit."""
        offset = float(np.clip(offset, -1.0, 1.0))
        angle = offset * MAX_BOUNCE
        outgoing = other(side)
        self.speed = min(self.speed * BALL_ACCEL, BALL_SPEED * BALL_SPEED_CAP)
        self.ball_vx = outgoing * self.speed * np.cos(angle)
        self.ball_vy = self.speed * np.sin(angle)
        self.ball_x = self.paddle_x(side) + outgoing * (BALL_RADIUS + 1)

        self.contacts[side] = self.contacts[side][1:] + [offset]
        self.returns[side] += 1
        self.rally += 1

    # -----------------------------------------------------------------

    def frames_until(self, side):
        """Frames before the ball reaches this side's plane; None if outgoing."""
        if (side == LEFT) != (self.ball_vx < 0):
            return None
        return abs(self.paddle_x(side) - self.ball_x) / max(abs(self.ball_vx), 1e-6)

    def observe(self, side):
        """What one player sees, mirrored so every player sees itself on the left.

        Mirroring is what makes one policy able to play either side, which is
        what makes self-play meaningful here.
        """
        flip = 1.0 if side == LEFT else -1.0
        mine, theirs = self.paddle[side], self.paddle[other(side)]
        approaching = self.frames_until(side)
        top = BALL_SPEED * BALL_SPEED_CAP

        return np.array([
            mine / HEIGHT * 2 - 1,
            float(self.last_move[side]),
            theirs / HEIGHT * 2 - 1,
            (self.ball_x / WIDTH * 2 - 1) * flip,
            self.ball_y / HEIGHT * 2 - 1,
            np.clip(self.ball_vx * flip / top, -1, 1),
            np.clip(self.ball_vy / top, -1, 1),
            self.speed / top * 2 - 1,
            -1.0 if approaching is None else np.clip(approaching / 120.0, 0, 1),
            np.clip(self.rally / 20.0, 0, 1),
            # where the opponent has been putting the ball lately: the raw
            # material for noticing that someone favours one kind of return
            *self.contacts[other(side)],
        ], dtype=np.float32)
