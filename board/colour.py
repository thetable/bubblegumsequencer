"""Reading a cell's colour, and deciding which taught colour it is.

Empty is deliberately not a colour to recognise. It is whatever the room
happens to be doing through a 12 mm hole, which is not a thing that can be
learned once, so a cell is empty exactly when no taught colour explains it.

The other idea running through this file is that position costs brightness,
not hue. The same pink ball reads L 50 mid-board and L 30 in the corner, where
the light is weakest and the sample smallest, while its a* stays near 48. So
brightness is distrusted when classifying, and ignored entirely when deciding
which cells belong to the colour being taught.
"""

import json
import os

import cv2
import numpy as np

PROTOTYPE_FILE = "prototypes.json"

# Floors on the spread, per Lab channel. Four balls in one frame understate
# how much a colour really varies across the board, and a spread measured as
# near zero would make the classifier absurdly sure of itself. L is floored
# much higher than a and b for the reason above.
SPREAD_FLOOR = (12.0, 4.0, 4.0)

# Teaching throws out strays by hue alone, never brightness. An empty hole
# that happens to be bright lands in the filled group and is the wrong colour;
# a real ball at the board's edge is the right colour but dim, and trimming on
# brightness threw that one away while keeping the stray. Floored at 7 because
# a dim ball loses some chroma too: the corner pink reads a* 34 where the
# others read 47.
TRIM_FLOOR = 7.0        # Lab units of hue that still count as the same colour
TRIM_DISTANCE = 3.0     # how many of those a colour may span
# Blue is the least saturated thing we teach: its chroma is 10 where green's
# is 34 and yellow's 64. That makes it the nearest prototype to plain neutral
# grey, so every dim empty hole is faintly "blue" and 4.5 spreads let a row
# of them through. Measured over eight frames in two lightings, real balls
# reach 2.8 and the nearest empty sits at 3.1; anything from 2.8 to 3.4 is
# clean, so this is the middle of that.
MAX_DISTANCE = 3.1      # spreads away from a prototype before we disown it
MARGIN = 1.4            # how much closer the winner must be than the runner-up
TOO_ALIKE = 8.0         # Lab units below which two taught colours are the same

# Shape, for when colour has stopped being enough. In afternoon sun the room
# comes down through the empty holes hard enough that an empty reads L 48
# against a green ball's L 49, on the same hue: 18 empties called green, and
# by eye the only thing telling them apart was the shading on the balls.
#
# A gumball is a sphere lit from below, so its middle faces the camera and is
# brighter than its rim. An empty hole is a flat, defocused view of the
# ceiling and has no such doming. Measured as a fraction of the cell's own
# brightness, which is what makes it survive a change of light: it is a shape
# in the cell, not a level.
#
# Only consulted where colour is already unsure. A ball whose colour is an
# obvious match is not made to prove itself again, which matters for the dim
# outer columns, where real balls dome as little as 0.025.
CONFIDENT = 2.5         # spreads within which colour alone settles it
DOME_MIN = 0.06         # middle brighter than rim, as a fraction of the median


def otsu_split(values):
    """Data-driven threshold, so there is no number to hand-tune."""
    v = np.asarray(values, np.float64)
    if v.max() <= v.min():
        return float(v.mean()), 0.0
    norm = ((v - v.min()) / (v.max() - v.min()) * 255).astype(np.uint8)
    t, _ = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh = v.min() + (t / 255.0) * (v.max() - v.min())
    lo, hi = v[v <= thresh], v[v > thresh]
    gap = (hi.min() - lo.max()) if len(lo) and len(hi) else 0.0
    return float(thresh), float(gap)


def load_prototypes():
    if not os.path.exists(PROTOTYPE_FILE):
        return {}
    with open(PROTOTYPE_FILE) as fh:
        return json.load(fh)


def save_prototypes(protos):
    with open(PROTOTYPE_FILE, "w") as fh:
        json.dump(protos, fh, indent=1)


def forget(names):
    """Drop these colours, before teaching them again.

    A re-teach happens because the stored values are wrong, so leaving them in
    place means the first colour taught is measured against the very numbers
    it is replacing. That is not hypothetical: a stale green sitting on a
    neutral grey refused a perfectly good blue for being 3 Lab units from it.
    """
    protos = load_prototypes()
    gone = [n for n in names if n in protos]
    for n in gone:
        protos.pop(n)
    if gone:
        save_prototypes(protos)
    return gone


def split_board(cells):
    """The cells holding one colour, the strays, and everything else.

    Brightness separates filled from empty, which is clean once the exposure
    is pinned. What is left can still hold an empty hole that happened to be
    looking at something bright, so the survivors are grouped by hue and only
    one group is kept: the brightest, because the balls are lit from inside
    the box and the room is not.

    Grouping rather than trimming from the median, because with four balls and
    four bright holes there is no majority for a median to sit in and the
    answer flips from frame to frame. Picking a whole group settles it.

    Hue and not brightness decides which cells belong together, because
    position costs brightness: the corner ball is dim, not differently
    coloured, and judging it on brightness threw it away.
    """
    cells = sampled(cells)
    if not cells:
        return [], [], []
    L = np.array([c["L"] for c in cells])
    cut, _ = otsu_split(L)
    bright = [c for c in cells if c["L"] > cut]
    dark = [c for c in cells if c["L"] <= cut]
    if not bright:
        return [], [], dark

    # Single linkage on hue: two cells join if they are closer than one step.
    reach = TRIM_FLOOR * TRIM_DISTANCE
    ab = np.array([[c["a"], c["b"]] for c in bright])
    group = list(range(len(bright)))
    for i in range(len(bright)):
        for j in range(i + 1, len(bright)):
            if np.hypot(*(ab[i] - ab[j])) <= reach and group[i] != group[j]:
                stale, fresh = group[j], group[i]
                group = [fresh if g == stale else g for g in group]

    best, best_light = None, -1.0
    for tag in set(group):
        members = [i for i, g in enumerate(group) if g == tag]
        light = float(np.median([bright[i]["L"] for i in members]))
        if light > best_light:
            best, best_light = members, light

    kept = [bright[i] for i in best]
    strays = [b for i, b in enumerate(bright) if i not in set(best)]
    return kept, strays, dark


def summarise(group):
    v = np.array([[c["L"], c["a"], c["b"]] for c in group])
    middle = np.median(v, axis=0)
    spread = np.maximum(
        1.4826 * np.median(np.abs(v - middle), axis=0), SPREAD_FLOOR)
    return {"L": float(middle[0]), "a": float(middle[1]), "b": float(middle[2]),
            "spread": [float(x) for x in spread], "n": int(len(v))}


def teach(cells, name):
    """Learn one colour from a board holding only that colour.

    Nothing to click: split_board finds the balls, and every run also
    re-learns what empty looks like from the rest of the board, which costs
    nothing and keeps it current.
    """
    kept, strays, dark = split_board(cells)
    if not 1 <= len(kept) <= 32:      # half a board is not one colour
        print(f"  Found {len(kept)} cells holding something, which does not "
              "look like a board holding one colour.")
        return None

    protos = load_prototypes()
    fresh = summarise(kept)

    # Two colours landing on the same point is not a close call, it is a sign
    # the sampler was not looking at balls at all. Taught against a grid that
    # was a row out, every colour came back the same neutral grey, and the
    # classifier then called the whole board empty: with the prototypes on top
    # of each other no winner can ever be MARGIN closer than the runner up.
    # Nothing in the output said so, which is what made it expensive.
    for other, was in protos.items():
        if other in (name, "empty"):
            continue
        apart = np.hypot(fresh["a"] - was["a"], fresh["b"] - was["b"])
        if apart < TOO_ALIKE:
            print(f"  {name} came out {apart:.0f} Lab units from {other}, "
                  f"which is no distance at all.")
            print("  Refusing to save it. Taught colours sit tens apart; this")
            print("  close means the cells were not on the balls. Check the "
                  "grid in probe.py before teaching again.")
            return None

    protos[name] = fresh
    protos["empty"] = summarise(dark)
    save_prototypes(protos)
    p = protos[name]
    print(f"  Taught {name} from {len(kept)} balls: "
          f"L {p['L']:.0f} a {p['a']:.0f} b {p['b']:.0f}, "
          f"spread {[round(v, 1) for v in p['spread']]}")
    if strays:
        print(f"  Ignored {len(strays)} bright "
              f"{'cell' if len(strays) == 1 else 'cells'} of the wrong colour.")
    return protos


def sampled(cells):
    """Only the cells this frame could actually measure.

    A cell whose hole has left the camera's view gets no sample at all, so it
    has no L to compare, plot or average. That is not the same as an empty
    hole, and the difference matters: empty is a reading, off-frame is the
    absence of one.
    """
    return [c for c in cells if "L" in c]


def classify(cells, protos):
    """Nearest taught colour, in spreads rather than raw Lab units.

    Empty is deliberately not a class to recognise. It is whatever the room
    happens to be doing through a 12 mm hole, which is not a thing that can be
    learned once, so a cell is empty exactly when no taught colour explains
    it. That is mitigation (c) in HANDOVER.md section 7, and it is what makes
    the classifier robust to a room nobody controls.

    Scaling by each colour's own spread is what lets two clusters sit close
    together without being confused: pale blue is tight, yellow is not.
    """
    names = [n for n in protos if n != "empty"]
    if not names:
        return
    for c in cells:
        if "L" not in c:
            # Nothing was read here, because the hole is off the edge of the
            # frame. None rather than "empty": it tells the stabiliser to hold
            # whatever this cell already was instead of wiping it, which is
            # what you want when someone slides the board half out of shot.
            c["colour"] = None
            continue
        scored = sorted(
            (np.sqrt(sum(((c[k] - protos[n][k]) / s) ** 2
                         for k, s in zip("Lab", protos[n]["spread"]))), n)
            for n in names)
        best = scored[0]
        second = scored[1][0] if len(scored) > 1 else 1e9
        near = best[0] <= MAX_DISTANCE and second >= MARGIN * best[0]
        # Shape only has to settle the cases colour could not.
        shaped = best[0] <= CONFIDENT or c.get("dome", 1.0) >= DOME_MIN
        c["colour"] = best[1] if near and shaped else "empty"
        c["distance"] = float(best[0])


# ---------------------------------------------------------------- sampling


def sample_cell(frame, gray, cx, cy, rx, ry):
    """Median Lab plus a texture score for one hole.

    The sample is an ellipse, not a circle, because that is the shape a round
    hole presents to a camera looking at it from 49 degrees off-axis. A circle
    would have to shrink to the narrow dimension and throw away most of the
    aperture on the outer columns, which are the cells with the least light to
    spare.

    Median, not mean: one glint or one speckle on a mottled shell drags a
    mean anywhere. Also drops the brightest 5%, which is where specular
    highlights live.
    """
    h, w = gray.shape
    x0, x1 = int(max(0, cx - rx)), int(min(w, cx + rx + 1))
    y0, y1 = int(max(0, cy - ry)), int(min(h, cy + ry + 1))
    patch = frame[y0:y1, x0:x1]
    gpatch = gray[y0:y1, x0:x1]
    if patch.size == 0:
        return None

    yy, xx = np.ogrid[y0:y1, x0:x1]
    radius = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2
    disc = radius <= 1.0
    if disc.sum() < 20:
        return None

    vals = gpatch[disc]
    keep_below = np.percentile(vals, 95)
    mask = disc & (gpatch <= keep_below)
    if mask.sum() < 20:
        mask = disc

    lab = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB)
    sel = lab[mask]
    L = float(np.median(sel[:, 0])) * 100.0 / 255.0
    a = float(np.median(sel[:, 1])) - 128.0
    b = float(np.median(sel[:, 2])) - 128.0

    # Texture: a sharp ball surface has detail, a defocused ceiling seen
    # through an empty hole does not. Independent of colour and brightness.
    lap = cv2.Laplacian(gpatch.astype(np.float64), cv2.CV_64F)
    texture = float(lap[mask].var())

    # A clipped channel has thrown its value away, and a pale ball is the first
    # thing to clip. Measured on the full disc, not the masked one: dropping
    # the brightest 5% hides the very pixels this is looking for.
    clipped = float((patch[disc].max(axis=1) >= 250).mean())

    # How much brighter the middle of the cell is than its rim, relative to
    # the cell itself. A ball is a lit sphere and domes; an empty hole is flat.
    middle, rim = radius <= 0.36, (radius > 0.49) & disc
    level = float(np.median(gpatch[disc]))
    dome = 0.0
    if middle.sum() >= 8 and rim.sum() >= 8 and level >= 1.0:
        dome = (float(np.median(gpatch[middle]))
                - float(np.median(gpatch[rim]))) / level

    return {
        "L": L, "a": a, "b": b,
        "dome": dome,
        "texture": texture,
        "clipped": clipped,
        "mean_gray": float(gpatch[mask].mean()),
    }


# ---------------------------------------------------------------- output
