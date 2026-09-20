#!/usr/bin/env python3
"""What is on the board, as opposed to what the last frame happened to show.

The vision side reads 64 cells thirty times a second and is not perfectly
steady: a ball at the edge of the board, where the sample is a third the size
it is in the middle, can wobble between its colour and empty for a frame. The
sequencer must not hear that.

So a reading has to hold for a while before it becomes part of the pattern,
and anything the vision side is unsure about is a gap in the evidence rather
than an empty hole. The distinction matters: empty means "somebody took the
ball out", no data means "I cannot see", and confusing the two is how a
pattern gets silently wiped.

Not a general-purpose debounce. Three things here are specific to this board:

- **no data holds.** If the tags are lost, the camera is knocked, or a cell
  cannot be read, the pattern stays exactly as it was. Every path that fails
  ends in the pattern being kept, never cleared.
- **a whole board changing at once is treated as suspect.** Balls arrive one
  at a time, so everything moving together is more likely a glitch than a
  player. It is not refused, which would deadlock on a board that already has
  balls on it when you start, but it is made to wait five times as long. A
  transient never survives that; a board someone really did rearrange does.
- **settling is per cell.** Dropping a ball while another cell is mid-change
  does not disturb either.
"""

import time

HOLD = 0.2          # seconds a reading must persist before it counts
CROWD = 12          # cells changing at once above which we get suspicious
CROWD_HOLD = 1.0    # and how long we then make them wait instead
EMPTY = "empty"


class Stabiliser:
    def __init__(self, cells=64, hold=HOLD, crowd=CROWD, crowd_hold=CROWD_HOLD):
        self.hold = hold
        self.crowd = crowd
        self.crowd_hold = crowd_hold
        self.pattern = [EMPTY] * cells
        self._wanted = [None] * cells
        self._since = [0.0] * cells
        self.blocked = False     # true while a crowd of changes is being made to wait
        self.blind = False       # true while the last reading carried no data

    def update(self, readings, now=None):
        """Fold one frame's readings in. Returns the cells that actually changed.

        `readings` is one entry per cell: a colour name, or None where the
        vision side could not say. All-None is the normal way to report that
        the board could not be located at all.
        """
        now = time.monotonic() if now is None else now
        self.blind = all(r is None for r in readings)

        # Anything disagreeing with the settled pattern is a candidate. Count
        # them before allowing any through, so a board-wide shift is spotted
        # while it is still a candidate rather than after it has landed.
        disagreeing = sum(1 for i, r in enumerate(readings)
                          if r is not None and r != self.pattern[i])
        self.blocked = disagreeing > self.crowd

        changed = []
        for i, reading in enumerate(readings):
            if reading is None or reading == self.pattern[i]:
                # Nothing to do, and a cell that goes quiet forgets whatever
                # it was building towards rather than resuming it later.
                self._wanted[i] = None
                continue
            if reading != self._wanted[i]:
                self._wanted[i] = reading
                self._since[i] = now
                continue
            wait = self.crowd_hold if self.blocked else self.hold
            if now - self._since[i] >= wait:
                was = self.pattern[i]
                self.pattern[i] = reading
                self._wanted[i] = None
                changed.append((i, was, reading))
        return changed

    def pending(self):
        """Cells currently working towards a change, for showing on screen."""
        return {i: w for i, w in enumerate(self._wanted) if w is not None}

    def filled(self):
        return {i: c for i, c in enumerate(self.pattern) if c != EMPTY}


def readings_from(cells, placed):
    """Turn one frame of classified cells into readings the stabiliser takes.

    `placed` says whether the geometry was trustworthy this frame. When it was
    not, every cell reports no data, which is what keeps a lost tag or a
    knocked camera from wiping the pattern.
    """
    if not placed or not cells:
        return [None] * 64
    return [c.get("colour") for c in cells]
