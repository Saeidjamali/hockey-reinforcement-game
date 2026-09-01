"""Shared drawing. The game, the training viewer and the spectator all use it,
so there is exactly one place that knows what the duel looks like."""

import numpy as np
import pygame

from src.config import WIDTH, HEIGHT, PADDLE_X, PADDLE_HALF, BALL_RADIUS
from src.pong import LEFT, RIGHT

PANEL = 260
SCREEN_W = WIDTH + PANEL

BG = (10, 10, 16)
COURT = (15, 15, 22)
EDGE = (60, 60, 80)
WHITE = (245, 245, 245)
DIM = (120, 120, 140)
MINE = (80, 160, 255)
THEIRS = (240, 90, 90)
BALL = (255, 230, 120)
GOLD = (255, 210, 80)


def line(text, colour=WHITE, size="sm"):
    return (text, colour, size)


def gap(px=10):
    return (None, None, px)


class Renderer:
    def __init__(self, caption="Paddle Duel"):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_W, HEIGHT))
        pygame.display.set_caption(caption)
        self.clock = pygame.time.Clock()
        self.fonts = {"lg": pygame.font.SysFont(None, 40),
                      "md": pygame.font.SysFont(None, 30),
                      "sm": pygame.font.SysFont(None, 23)}

    def caption(self, text):
        pygame.display.set_caption(text)

    def tick(self, fps=60):
        self.clock.tick(fps)

    def quit(self):
        pygame.quit()

    def pump(self):
        events = {"quit": False, "keys": []}
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                events["quit"] = True
            elif event.type == pygame.KEYDOWN:
                events["keys"].append(event.key)
                if event.key == pygame.K_ESCAPE:
                    events["quit"] = True
        return events

    def held(self):
        return pygame.key.get_pressed()

    # -----------------------------------------------------------------

    def frame(self, sim, human_side=LEFT, panel=(), flash=0.0, centre=None, sub=None):
        self.screen.fill(BG)
        court = COURT if flash <= 0 else tuple(
            int(c + (70 - c) * flash) for c in COURT)
        pygame.draw.rect(self.screen, court, (0, 0, WIDTH, HEIGHT))
        pygame.draw.rect(self.screen, BG, (WIDTH, 0, PANEL, HEIGHT))
        pygame.draw.line(self.screen, EDGE, (WIDTH, 0), (WIDTH, HEIGHT), 2)
        for y in range(0, HEIGHT, 32):                     # centre line
            pygame.draw.line(self.screen, (34, 34, 46),
                             (WIDTH // 2, y), (WIDTH // 2, y + 16), 2)

        for side in (LEFT, RIGHT):
            colour = MINE if side == human_side else THEIRS
            x = PADDLE_X if side == LEFT else WIDTH - PADDLE_X
            pygame.draw.rect(self.screen, colour,
                             (x - 7, sim.paddle[side] - PADDLE_HALF, 14, PADDLE_HALF * 2),
                             border_radius=6)

        pygame.draw.circle(self.screen, BALL,
                           (int(sim.ball_x), int(sim.ball_y)), int(BALL_RADIUS))

        if centre is not None:
            big = pygame.font.SysFont(None, 200)
            glyph = big.render(str(centre), True, WHITE)
            self.screen.blit(glyph, glyph.get_rect(center=(WIDTH // 2, HEIGHT // 2 - 30)))
        if sub is not None:
            small = self.fonts["md"]
            glyph = small.render(sub, True, GOLD)
            self.screen.blit(glyph, glyph.get_rect(center=(WIDTH // 2, HEIGHT // 2 + 90)))

        y = 20
        for text, colour, size in panel:
            if text is None:
                y += size
                continue
            self.screen.blit(self.fonts.get(size, self.fonts["sm"]).render(text, True, colour),
                             (WIDTH + 20, y))
            y += {"lg": 44, "md": 34, "sm": 24}.get(size, 24)

        pygame.display.flip()
