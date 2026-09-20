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
# much higher than a and b for the reason above. Measured on a full board,
# 4.5 spreads sits between the worst real ball at 3.9 and the nearest empty
# hole at 5.7.
SPREAD_FLOOR = (12.0, 4.0, 4.0)

# Teaching throws out strays by hue alone, never brightness. An empty hole
# that happens to be bright lands in the filled group and is the wrong colour;
# a real ball at the board's edge is the right colour but dim, and trimming on
# brightness threw that one away while keeping the stray. Floored at 7 because
# a dim ball loses some chroma too: the corner pink reads a* 34 where the
# others read 47.
TRIM_FLOOR = 7.0        # Lab units of hue that still count as the same colour
TRIM_DISTANCE = 3.0     # how many of those a colour may span
MAX_DISTANCE = 4.5      # spreads away from a prototype before we disown it
MARGIN = 1.4            # how much closer the winner must be than the runner-up


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
    protos[name] = summarise(kept)
    protos["empty"] = summarise(dark)
    with open(PROTOTYPE_FILE, "w") as fh:
        json.dump(protos, fh, indent=1)
    p = protos[name]
    print(f"  Taught {name} from {len(kept)} balls: "
          f"L {p['L']:.0f} a {p['a']:.0f} b {p['b']:.0f}, "
          f"spread {[round(v, 1) for v in p['spread']]}")
    if strays:
        print(f"  Ignored {len(strays)} bright "
              f"{'cell' if len(strays) == 1 else 'cells'} of the wrong colour.")
    return protos


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
        scored = sorted(
            (np.sqrt(sum(((c[k] - protos[n][k]) / s) ** 2
                         for k, s in zip("Lab", protos[n]["spread"]))), n)
            for n in names)
        best = scored[0]
        second = scored[1][0] if len(scored) > 1 else 1e9
        c["colour"] = (best[1] if best[0] <= MAX_DISTANCE
                       and second >= MARGIN * best[0] else "empty")
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
    disc = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1.0
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

    return {
        "L": L, "a": a, "b": b,
        "texture": texture,
        "clipped": clipped,
        "mean_gray": float(gpatch[mask].mean()),
    }


# ---------------------------------------------------------------- output
