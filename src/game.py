"""Human against a trained agent. Same rules, same actions, both sides."""

import numpy as np
import pygame

from src.config import HEIGHT, PADDLE_SPEED
from src.pong import PongSim, LEFT, RIGHT, other
from src.render import Renderer, line, gap, WHITE, DIM, GOLD, MINE, THEIRS


class Game:
    COUNTDOWN = 180        # three seconds, once, before the match starts
    HOLD = 30              # half a second between points, only so the score
                           # registers. No countdown: the serve now leaves the
                           # server's paddle and crosses the full court, so it
                           # is visible coming and needs no warning.

    def __init__(self, opponent, label="opponent", human_side=LEFT, first_to=7):
        self.renderer = Renderer("Paddle Duel")
        self.opponent = opponent
        self.label = label
        self.side = human_side
        self.foe = other(human_side)
        self.first_to = first_to
        self.rng = np.random.default_rng()
        self.sim = PongSim(self.rng).reset()
        self.score = {LEFT: 0, RIGHT: 0}
        self.running = True
        self.flash = 0.0
        self.message = ""
        self.countdown = self.COUNTDOWN
        self.hold = 0
        self.last_point = None

    def human_action(self):
        keys = self.renderer.held()
        up = keys[pygame.K_UP] or keys[pygame.K_w]
        down = keys[pygame.K_DOWN] or keys[pygame.K_s]
        return 0 if up and not down else 2 if down and not up else 1

    def run(self):
        self.opponent.reset(self.rng)
        while self.running:
            self.renderer.tick(60)
            if self.renderer.pump()["quit"]:
                break

            # Both paddles may move during the countdown, so you can set up —
            # only the ball is held. It sits visible at the serving paddle
            # rather than arriving out of nowhere.
            if self.countdown > 0 or self.hold > 0:
                if self.countdown > 0:
                    self.countdown -= 1
                else:
                    self.hold -= 1
                self.sim.move(self.side, self.human_action())
                self.sim.move(self.foe, self.opponent.act(self.sim, self.foe))
                self.draw()
                self.flash = max(0.0, self.flash - 0.06)
                continue

            self.sim.move(self.side, self.human_action())
            self.sim.move(self.foe, self.opponent.act(self.sim, self.foe))
            result = self.sim.advance()

            if result in (LEFT, RIGHT):
                self.score[result] += 1
                self.flash = 1.0
                self.last_point = result
                self.message = "point for you" if result == self.side else "point against"
                if max(self.score.values()) >= self.first_to:
                    self.message = ("you win the match" if self.score[self.side] > self.score[self.foe]
                                    else "it wins the match")
                    self.countdown = 0
                    self.draw(); pygame.time.wait(2600)
                    break
                # The winner serves the next point, off their own paddle.
                self.sim.reset(server=result)
                self.opponent.reset(self.rng)
                self.hold = self.HOLD

            self.draw()
            self.flash = max(0.0, self.flash - 0.06)
        self.renderer.quit()

    def draw(self):
        sim = self.sim
        panel = [
            line("PADDLE DUEL", WHITE, "lg"),
            line(self.label, THEIRS), gap(12),
            line(f"you   {self.score[self.side]}", MINE, "md"),
            line(f"it    {self.score[self.foe]}", THEIRS, "md"),
            line(f"first to {self.first_to}", DIM), gap(12),
            line(f"rally  {sim.rally}"),
            line(f"speed  {sim.speed:.0f}"),
        ]
        if self.message:
            panel += [gap(10), line(self.message, GOLD)]
        panel += [gap(HEIGHT - 420),
                  line("W/S or arrows to move", DIM),
                  line("ESC to quit", DIM)]
        centre = sub = None
        if self.countdown > 0:
            # Ceiling division: floor+1 shows a phantom "4" on the first frame.
            centre = -(-self.countdown // 60)
            sub = "get ready"
        elif self.hold > 0:
            sub = ("your point — you serve" if self.last_point == self.side
                   else "its point — it serves")
        self.renderer.frame(sim, human_side=self.side, panel=panel, flash=self.flash,
                            centre=centre, sub=sub)


def select(entries):
    """Pick which trained version to face."""
    from src.render import SCREEN_W, BG
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_W, HEIGHT))
    pygame.display.set_caption("Paddle Duel — choose your opponent")
    clock = pygame.time.Clock()
    big = pygame.font.SysFont(None, 50)
    mid = pygame.font.SysFont(None, 32)
    small = pygame.font.SysFont(None, 23)

    index = len(entries) - 1
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    return None
                if event.key in (pygame.K_UP, pygame.K_w):
                    index = (index - 1) % len(entries)
                elif event.key in (pygame.K_DOWN, pygame.K_s):
                    index = (index + 1) % len(entries)
                elif event.key in (pygame.K_RETURN, pygame.K_SPACE):
                    return entries[index]
                elif pygame.K_1 <= event.key <= pygame.K_9:
                    choice = event.key - pygame.K_1
                    if choice < len(entries):
                        return entries[choice]

        screen.fill(BG)
        screen.blit(big.render("CHOOSE YOUR OPPONENT", True, WHITE), (70, 55))
        screen.blit(small.render("↑↓ or number keys, ENTER to play, ESC to quit", True, DIM),
                    (70, 108))
        for i, entry in enumerate(entries):
            y = 165 + i * 56
            picked = i == index
            if picked:
                pygame.draw.rect(screen, (30, 34, 52), (60, y - 10, SCREEN_W - 120, 48))
                pygame.draw.rect(screen, MINE, (60, y - 10, 4, 48))
            screen.blit(mid.render(f"{i + 1}.  {entry['tier']}", True,
                                   WHITE if picked else (150, 150, 165)), (84, y))
            m = entry["metrics"]
            detail = (f"generation {entry['generation']}   {entry['steps']:,} steps   "
                      f"wins {m['win_rate']:.0%} against the fixed ladder")
            screen.blit(small.render(detail, True, (140, 150, 170) if picked else (95, 95, 110)),
                        (330, y + 6))
        pygame.display.flip()
        clock.tick(60)
