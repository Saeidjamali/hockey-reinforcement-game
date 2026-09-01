# Decision record: why the dodge game was abandoned

The project began as an asymmetric game — a turret shooting, a player dodging —
and a great deal of work went into the learner before the real question was
asked: **does reading the opponent have any value in this game at all?**

The instrument scored four levels of information and took the marginal value of
knowing *who* you are playing. The code has been removed along with the dodge
game, but the method is worth keeping:

    fixed         no opponent-specific information
    state         sees the situation only; can position, cannot read
    state+type    sees the situation AND which opponent this is
    clairvoyant   knows the realised outcome; an upper bound

    reading_value = state+type  -  state

`state` has to be the baseline. Comparing against `fixed` credits reading with
the value of merely reacting to the situation. Policies were fit on half the
sampled situations and scored on the other half, split at random — splitting by
index parity once made a control built so that reading MUST pay report exactly
zero, because the control's state bucket was also index parity.

## Measured

| design | reading_value | note |
|---|---:|---|
| dodge, as built | +0.049 ± 0.016 | the whole learnable range was 9% → 15% of volleys |
| dodge + lanes | +0.231 ± 0.020 | but `state+type` = 1.00; no player agency |
| **dodge + inertia** | **+0.151 ± 0.015** | best real design measured; asymmetric |
| dodge, 5 shots | +0.132 ± 0.021 | ~15% of situations unavoidable; unplayable |
| control (prediction-friendly) | +0.204 ± 0.021 | reference for "reading obviously matters" |

The dodge game's optimal strategy was positional — herd the player into a wall —
and that needs no perception. So curriculum work, action-space work and metric
work were all orthogonal to the goal and each hit the same wall.

**If the paddle game is abandoned too, inertia-dodge is the fallback at +0.151.
Its code is gone; it is a rebuild, not a switch. The design was: continuous
movement with acceleration ~2.0 px/frame², friction 0.85, top speed 7.**

## Superseded run logs

Kept so the conclusions are checkable rather than re-derived by accident.

| log | what it shows |
|---|---|
| `training_log.txt` | 3M steps, no curriculum; hits/episode flat at ~0.3 |
| `curriculum_run_log.txt` | curriculum active; skill <0.10 at 1.95M where none reached 0.502 — it measured WORSE |
| `full_catalogue_log.txt` | all 1140 volley combinations; skill 0.004 at 750k vs 0.221 factorised — expressiveness that cannot be searched |
| `curated_catalogue_log.txt` | 190 coordinated volleys; plateaued at exactly the blind-volley baseline |
