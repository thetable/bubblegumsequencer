#!/usr/bin/env python3
"""Bubblegum sequencer, step 1: measure all 64 cells.

Answers one question: how far apart do filled and empty holes actually sit?

The holes are found in the image rather than predicted from a grid: they are
the only round mid-bright things on a black sheet, so detecting them directly
is easier and far more accurate than modelling where they ought to be. A
smooth polynomial warp is then fitted to the detections, which both fills in
any hole the detector missed and absorbs the lens's barrel distortion.

You still click the four corner holes once (cached to corners.json), but only
to say which part of the frame the board occupies. Nothing is positioned from
them.

Usage:
    python cell_probe.py --index 0              # live camera
    python cell_probe.py --image frame.png      # a saved frame
    python cell_probe.py --image frame.png --reclick   # redo the corners

Outputs, next to the working directory:
    cells.csv       one row per cell
    cells_map.png   the frame with every sampled disc drawn on it
    cells_plot.png  two scatter plots: colour plane, and brightness vs texture
    frame_last.png  the raw frame, so a run can be repeated offline
    warp.json       the fitted geometry, reused when the board is too dark to
                    find the empty holes
    tag_reference.json  where the grid sits relative to the AprilTags, so the
                    geometry follows the board when it is moved
    exposure.json   the exposure --dial-exposure settled on
    prototypes.json the taught colours and their spreads

Exposure is set on the camera through uvc-util, which lives next to this file.
OpenCV accepts the exposure property under macOS and then ignores it.
"""

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys

import cv2
import numpy as np

COLS, ROWS = 16, 4
WARP_FILE = "warp.json"

# How far out to look when working out what a hole should be brighter than.
# 90 was too far: it reached up into the LED strips sitting just beyond the
# top and bottom rows, inflated the local reference there, and lost half of
# row 0 and a third of row 3. 40 keeps the strips out. Much below that and it
# starts averaging the hole into its own background.
BACKGROUND_SIGMA = 40
CORNER_PROMPTS = [
    "top-left hole (row 0, col 0)",
    "top-right hole (row 0, col 15)",
    "bottom-right hole (row 3, col 15)",
    "bottom-left hole (row 3, col 0)",
]


# ---------------------------------------------------------------- corners

def click_corners(frame, preview_width=1280):
    """Ask for the four corner hole centres, in order."""
    scale = min(1.0, preview_width / frame.shape[1])
    view0 = cv2.resize(frame, None, fx=scale, fy=scale) if scale < 1.0 else frame.copy()
    picked = []

    def on_mouse(event, mx, my, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN and len(picked) < 4:
            picked.append((mx / scale, my / scale))

    win = "click the 4 corner holes"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)

    while len(picked) < 4:
        view = view0.copy()
        for i, (px, py) in enumerate(picked):
            cv2.circle(view, (int(px * scale), int(py * scale)), 8, (0, 255, 0), 2)
            cv2.putText(view, str(i), (int(px * scale) + 12, int(py * scale)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        msg = f"click {CORNER_PROMPTS[len(picked)]}    (u = undo, q = abort)"
        cv2.putText(view, msg, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(view, msg, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 0), 1, cv2.LINE_AA)
        cv2.imshow(win, view)
        key = cv2.waitKey(20) & 0xFF
        if key == ord("q"):
            cv2.destroyWindow(win)
            return None
        if key == ord("u") and picked:
            picked.pop()

    cv2.destroyWindow(win)
    return picked


# ---------------------------------------------------------------- finding holes

def find_holes(gray, corners, pitch, sens):
    """Every round mid-bright blob inside the board area.

    Dividing by a heavily blurred copy removes the lens's vignetting and the
    box's uneven lighting, so one ratio works from the middle of the board to
    the corner. What survives is scale-free: a hole is brighter than the sheet
    around it whatever the absolute level.

    Only bright blobs, deliberately. Looking for dark ones too, so that an
    empty hole could also be found as a void against a lit sheet, was tried
    and made both cases worse: the dark sheet between the holes forms its own
    blobs and they crowd out the real ones. Geometry is a property of the rig,
    so it is calibrated once in ordinary room light and reused from warp.json.
    """
    g = gray.astype(np.float32)
    background = cv2.GaussianBlur(g, (0, 0), BACKGROUND_SIGMA)
    mask = ((g / (background + 1e-6)) > sens).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))

    n, _, stats, centres = cv2.connectedComponentsWithStats(mask, 8)
    blobs = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        w, h = int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT])
        if 300 < area < 6000 and 0.45 < w / max(h, 1) < 2.2:
            blobs.append({"x": float(centres[i][0]), "y": float(centres[i][1]),
                          "area": area, "w": w, "h": h})
    if not blobs:
        return []

    # Holes are all the same physical size, so the median area is the scale to
    # judge everything else by. No number to type in.
    median_area = float(np.median([b["area"] for b in blobs]))
    blobs = [b for b in blobs if 0.35 * median_area < b["area"] < 2.5 * median_area]

    # The corners say where the board is; anything outside it is box or room.
    inside = cv2.fillPoly(np.zeros(gray.shape, np.uint8), np.int32([corners]), 255)
    margin = max(3, int(0.8 * pitch)) | 1
    inside = cv2.dilate(inside, np.ones((margin, margin), np.uint8))
    return [b for b in blobs if inside[int(b["y"]), int(b["x"])] > 0]


def group_rows(blobs, corners):
    """Split the blobs into four rows.

    Clustering on pixel height alone is not safe. The board can sit tilted and
    the rows bow, and between them the four rows' height ranges overlap, at
    which point y clustering merges two rows and splits another.

    So use the corner clicks for the one thing they are reliable for: a rough
    rectification. Mapped through the corner quad, a blob's row number falls
    out directly. That first guess is then refined against a curve fitted to
    each row, which is what absorbs the distortion the corners cannot see.
    """
    xs = np.array([b["x"] for b in blobs])
    ys = np.array([b["y"] for b in blobs])

    src = np.float32([[0, 0], [COLS - 1, 0], [COLS - 1, ROWS - 1], [0, ROWS - 1]])
    inverse = cv2.getPerspectiveTransform(np.float32(corners), src)
    grid = cv2.perspectiveTransform(
        np.float32([np.c_[xs, ys]]), inverse)[0]
    label = np.clip(np.round(grid[:, 1]), 0, ROWS - 1).astype(int)

    # Where the corner quad thinks each row sits mid-board, for any row too
    # sparsely detected to fit a curve to.
    forward = cv2.getPerspectiveTransform(src, np.float32(corners))
    mid = cv2.perspectiveTransform(
        np.float32([[[(COLS - 1) / 2, k] for k in range(ROWS)]]), forward)[0]

    for _ in range(4):
        curves = [np.polyfit(xs[label == k], ys[label == k], 2)
                  if (label == k).sum() >= 3 else np.array([0.0, 0.0, mid[k][1]])
                  for k in range(ROWS)]
        predicted = np.stack([np.polyval(c, xs) for c in curves], axis=1)
        label = np.argmin(np.abs(ys[:, None] - predicted), axis=1)
    return label


# ---------------------------------------------------------------- tags

TAG_DICT = cv2.aruco.DICT_APRILTAG_36h11
REFERENCE_FILE = "tag_reference.json"


def detect_tags(gray):
    """Every AprilTag in the frame, as id -> its four corners.

    The tags sit near the strips and the white paper tends to wash out, so if
    the plain frame comes up short the same detector is run again on contrast
    equalised copies and the best answer wins. Note that widening the adaptive
    threshold range, which looks like the obvious knob, made things worse on a
    real frame: four tags became two.
    """
    d = cv2.aruco.getPredefinedDictionary(TAG_DICT)
    detector = cv2.aruco.ArucoDetector(d, cv2.aruco.DetectorParameters())
    flat = gray.astype(np.float32)
    flat = np.clip(flat / (cv2.GaussianBlur(flat, (0, 0), 31) + 1e-6) * 128, 0, 255)

    best = {}
    for image in (gray, cv2.createCLAHE(6.0, (8, 8)).apply(gray),
                  flat.astype(np.uint8)):
        corners, ids, _ = detector.detectMarkers(image)
        if ids is None:
            continue
        found = {int(i): c[0].astype(np.float64) for i, c in zip(ids.ravel(), corners)}
        if len(found) > len(best):
            best = found
        if len(best) >= 4:
            break
    return best


def write_reference(tags, rows):
    with open(REFERENCE_FILE, "w") as fh:
        json.dump({"tags": {str(i): np.asarray(c).tolist() for i, c in tags.items()},
                   "cells": np.asarray(rows).tolist()}, fh)


def save_reference(tags, cells):
    """Remember where the grid sits relative to the tags.

    Kept as plain image coordinates from the frame it was measured in. On a
    later frame the tags give a homography from that frame to this one, and
    the cells come along with it. If the board has not moved the homography is
    the identity and nothing is lost.

    The fifth number per cell counts how many times that cell's position has
    been confirmed against a hole actually seen. Calibration starts everything
    at one; accumulate_reference raises it.
    """
    write_reference(tags, [[c["x"], c["y"], c["rx"], c["ry"], 1] for c in cells])


def load_reference():
    if not os.path.exists(REFERENCE_FILE):
        return None, None
    with open(REFERENCE_FILE) as fh:
        r = json.load(fh)
    rows = [row + [1] * (5 - len(row)) for row in r["cells"]]
    return ({int(k): np.array(v) for k, v in r["tags"].items()},
            np.array(rows, float))


def row_pitch(cells):
    return float(np.median([np.hypot(cells[i + 1]["x"] - cells[i]["x"],
                                     cells[i + 1]["y"] - cells[i]["y"])
                            for i in range(len(cells) - 1)
                            if cells[i]["row"] == cells[i + 1]["row"]]))


def accumulate_reference(gray, cells, H, sens):
    """Pin each cell's stored position using whatever holes this frame shows.

    No frame shows every hole. One can vanish entirely when the sheet beside it
    happens to reflect a strip back at the same brightness as the room seen
    through it, which is what puts five of the top row out of reach at the
    moment. But a different five vanish once the room changes, so remembering
    each cell from the frames where it was clear fills the map in over a few
    runs.

    Measurements are carried back into the reference's own frame before being
    stored, so they accumulate no matter where the board was standing. A
    running mean, because any single measurement carries a few px of scatter.
    """
    ref_tags, rows = load_reference()
    if ref_tags is None:
        return None
    pitch = row_pitch(cells)
    pts = np.float32([[c["x"], c["y"]] for c in cells])
    blobs = find_holes(gray, cv2.convexHull(pts).reshape(-1, 2), pitch, sens)
    if len(blobs) < 10:
        return None

    bx = np.array([b["x"] for b in blobs])
    by = np.array([b["y"] for b in blobs])
    back = np.linalg.inv(H)
    updated = 0
    for i, c in enumerate(cells):
        d = np.hypot(bx - c["x"], by - c["y"])
        j = int(np.argmin(d))
        if d[j] > 0.30 * pitch:
            continue
        here = cv2.perspectiveTransform(
            np.float32([[[bx[j], by[j]]]]), back)[0][0]
        # Weight falls as a cell gets confirmed, but never below a twelfth, so
        # the map can still follow a tag that is slowly peeling off.
        w = 1.0 / min(rows[i][4] + 1, 12)
        rows[i][0] += w * (float(here[0]) - rows[i][0])
        rows[i][1] += w * (float(here[1]) - rows[i][1])
        rows[i][4] += 1
        updated += 1
    write_reference(ref_tags, rows)
    return updated, int((rows[:, 4] > 1).sum())


def cells_from_tags(tags):
    """Place all 64 cells by moving the stored grid onto the tags seen now.

    Three tags is enough, which matters because one of the four is usually the
    marginal one. Four is better: the homography is then over-determined and
    the fit reports how well it agrees.
    """
    ref_tags, ref_cells = load_reference()
    if ref_tags is None:
        return None, None
    shared = sorted(set(ref_tags) & set(tags))
    if len(shared) < 3:
        return None, None

    src = np.vstack([ref_tags[i] for i in shared])
    dst = np.vstack([tags[i] for i in shared])
    H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
    if H is None:
        return None, None
    moved = cv2.perspectiveTransform(
        np.float32([ref_cells[:, :2]]), H)[0]

    # Carry the sample ellipses across too, scaled by however much the
    # homography stretches the image at each cell.
    nudged = cv2.perspectiveTransform(
        np.float32([ref_cells[:, :2] + [1.0, 0.0]]), H)[0]
    raised = cv2.perspectiveTransform(
        np.float32([ref_cells[:, :2] + [0.0, 1.0]]), H)[0]
    sx = np.linalg.norm(nudged - moved, axis=1)
    sy = np.linalg.norm(raised - moved, axis=1)

    fit = float(np.median(np.linalg.norm(
        cv2.perspectiveTransform(np.float32([src]), H)[0] - dst, axis=1)))
    cells = [{"idx": i, "row": i // COLS, "col": i % COLS,
              "x": float(moved[i][0]), "y": float(moved[i][1]),
              "rx": max(4, int(ref_cells[i][2] * sx[i])),
              "ry": max(4, int(ref_cells[i][3] * sy[i])),
              "found": 0}
             for i in range(COLS * ROWS)]
    return cells, (shared, fit, H)


# ---------------------------------------------------------------- board to pixels

def warp_basis(u, v, v_degree):
    """Polynomial terms in column and row, both scaled to -1..1.

    Cubic across the columns because barrel distortion makes the pitch a
    smooth hump: it runs about 67 px at the edge of the frame to 135 px in
    the middle. A homography cannot bend like that, which is why the four
    corners alone are not enough.
    """
    u = np.asarray(u, float) / (COLS - 1) * 2 - 1
    v = np.asarray(v, float) / (ROWS - 1) * 2 - 1
    return np.stack([u ** i * v ** j for j in range(v_degree + 1) for i in range(4)], axis=1)


def fit_warp(u, v, x, y, v_degree):
    basis = warp_basis(u, v, v_degree)
    cx = np.linalg.lstsq(basis, np.asarray(x, float), rcond=None)[0]
    cy = np.linalg.lstsq(basis, np.asarray(y, float), rcond=None)[0]
    return cx, cy


def seed_labels(blobs, label):
    """Column numbers for any row where all 16 holes were found.

    A complete row needs no interpretation: sorted left to right, its blobs
    are columns 0 to 15. Two such rows are enough to bootstrap the warp, and
    the rest of the board is then read off the model rather than guessed.
    """
    u, v, x, y = [], [], [], []
    complete = []
    for k in range(ROWS):
        row = sorted([b for b, l in zip(blobs, label) if l == k], key=lambda b: b["x"])
        if len(row) == COLS:
            complete.append(k)
            u += list(range(COLS))
            v += [k] * COLS
            x += [b["x"] for b in row]
            y += [b["y"] for b in row]
    return u, v, x, y, complete


def load_warp():
    if not os.path.exists(WARP_FILE):
        return None
    with open(WARP_FILE) as fh:
        w = json.load(fh)
    return np.array(w["cx"]), np.array(w["cy"]), int(w["v_degree"])


def save_warp(cx, cy, v_degree):
    with open(WARP_FILE, "w") as fh:
        json.dump({"cx": list(cx), "cy": list(cy), "v_degree": v_degree}, fh)


def locate_cells(blobs, corners, pitch):
    """All 64 centres, or None if the board could not be read confidently.

    This is the hole-fitting route, used when the tags cannot place the grid
    or when --recalibrate asks for the reference to be measured again. It fits
    the polynomial warp to the holes in this frame, which is the most accurate
    answer available but needs the empty holes to be visible, so it needs room
    light. Failing that it reuses warp.json, which is only correct while
    nothing has shifted.

    There is deliberately no fall back to the four corner clicks. Seeding from
    them was tried and it converges on a mirrored board: wrong by a whole
    column, yet fitting its own detections happily. A refusal you can act on
    beats a plausible wrong answer.
    """
    if blobs:
        label = group_rows(blobs, corners)
        print(f"  holes per row: {[int((label == k).sum()) for k in range(ROWS)]}")
        u, v, x, y, complete = seed_labels(blobs, label)
    else:
        complete = []

    fitting = len(complete) >= 2
    if not fitting:
        saved = load_warp()
        if saved is None:
            print("  Need two rows with all 16 holes to anchor the columns; "
                  f"got {len(complete)}, and no {WARP_FILE} to fall back on.")
            print("  If the board or camera has been moved, --reclick first: "
                  "the cached corners bound where holes are looked for, so a "
                  "board that has shifted loses its edge holes.")
            print("  Otherwise try a lower --sens, or more light on the rows "
                  "that came up short.")
            return None, None
        cx, cy, v_degree = saved
        print(f"  Only {len(complete)} complete rows, so using the geometry in "
              f"{WARP_FILE}.")
        print("  That holds only while the board and camera stay put.")
    else:
        v_degree = 1
        cx, cy = fit_warp(u, v, x, y, v_degree)

    bx = np.array([b["x"] for b in blobs])
    by = np.array([b["y"] for b in blobs])
    all_u = np.tile(np.arange(COLS), ROWS)
    all_v = np.repeat(np.arange(ROWS), COLS)

    matched = {}
    passes = (4 if fitting else 1) if blobs else 0
    for _ in range(passes):
        grid = warp_basis(all_u, all_v, v_degree)
        px, py = grid @ cx, grid @ cy
        dist = np.hypot(bx[:, None] - px[None, :], by[:, None] - py[None, :])
        nearest = np.argmin(dist, axis=1)
        d = dist[np.arange(len(blobs)), nearest]

        # One blob per cell, and only if it is comfortably inside that cell.
        matched = {}
        for i in np.where(d < 0.45 * pitch)[0]:
            cell = int(nearest[i])
            if cell not in matched or d[i] < d[matched[cell]]:
                matched[cell] = i
        if not fitting or len(matched) < 8:
            break
        v_degree = 2 if len({all_v[c] for c in matched}) >= 3 else 1
        cx, cy = fit_warp([all_u[c] for c in matched], [all_v[c] for c in matched],
                          [bx[i] for i in matched.values()],
                          [by[i] for i in matched.values()], v_degree)

    grid = warp_basis(all_u, all_v, v_degree)
    px, py = grid @ cx, grid @ cy
    residual = [np.hypot(bx[i] - px[c], by[i] - py[c]) for c, i in matched.items()]

    # A wide lens always magnifies the middle of the frame, so the column pitch
    # has to be a hump: widest mid-board, tightest at both edges. A fit that
    # comes out dished has locked onto the wrong columns, and it will fit its
    # own mistake perfectly well, so the residual will not give it away.
    middle = np.diff(px[COLS // 2 - 2:COLS // 2 + 2]).mean()
    edges = np.diff(px[:3]).mean(), np.diff(px[-3:]).mean()
    if fitting and middle <= max(edges):
        print(f"  Column pitch came out {middle:.0f} px mid-board against "
              f"{min(edges):.0f} to {max(edges):.0f} px at the edges.")
        print("  That is the wrong way round, so the columns are misassigned. "
              "Stopping.")
        return None, None
    if fitting:
        save_warp(cx, cy, v_degree)

    # Sample at the fitted position, not the blob's own centroid. A dim hole is
    # only partly caught by the detector and its centroid drifts towards the lit
    # side; the warp is one smooth surface through every hole on the board, so it
    # averages that away. The sheet is rigid, so the true centres really are
    # smooth in (col, row) and nothing real is being smoothed out.
    # How much taller than wide a hole reads, as a smooth function of column.
    # The lens squashes the edge holes across the board but not along it, so a
    # hole at column 15 comes out 32 x 53 px where a middle one is 73 x 72.
    # This is measured off the holes themselves rather than worked out from the
    # off-axis angle, so it needs no numbers about the camera.
    ratio = [blobs[i]["h"] / max(blobs[i]["w"], 1) for i in matched.values()]
    shape = (np.polyfit([all_u[c] for c in matched], ratio, 2)
             if len(matched) >= 8 else np.array([0.0, 0.0, 1.0]))

    cells = []
    for c in range(COLS * ROWS):
        # Hole is 12 mm across a 24 mm pitch, so a disc safely inside the
        # aperture is about a fifth of the local pitch. Taking it from the warp
        # means edge cells, which the lens squashes to half the width of the
        # middle ones, get a disc that shrinks with them.
        u = all_u[c]
        step = warp_basis([u + (1 if u < COLS - 1 else -1)], [all_v[c]], v_degree)
        gap = float(np.hypot(step @ cx - px[c], step @ cy - py[c])[0])
        rx = max(4, int(0.19 * gap))
        # Stretch it back out along the hole. Never squash below a circle: an
        # aspect under 1 would mean the fit has gone wrong, not that the hole
        # is wider than it is tall.
        aspect = float(np.clip(np.polyval(shape, u), 1.0, 2.5))
        cells.append({"idx": c, "row": c // COLS, "col": c % COLS,
                      "x": float(px[c]), "y": float(py[c]),
                      "rx": rx, "ry": max(4, int(rx * aspect)),
                      "found": int(c in matched)})
    return cells, residual


# ---------------------------------------------------------------- colours

PROTOTYPE_FILE = "prototypes.json"
# Floors on the spread, per Lab channel. Four balls in one frame understate
# how much a colour really varies across the board, and a spread measured as
# near zero would make the classifier absurdly sure of itself.
#
# L is floored much higher than a and b because brightness is what position
# costs you: the same pink ball reads L 50 mid-board and L 30 in the corner,
# where the light is weakest and the sample smallest, while its a stays near
# 48 either way. Measured on a full board, 4.5 spreads then sits between the
# worst real ball at 3.9 and the nearest empty hole at 5.7.
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
    if not 1 <= len(kept) <= COLS * ROWS // 2:
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


def draw_map(frame, cells, path):
    """Every disc sits where the warp says. Colour says whether the detector
    also saw a hole there, which is a lighting check, not a position check."""
    out = frame.copy()
    for c in cells:
        p = (int(c["x"]), int(c["y"]))
        colour = (0, 255, 0) if c["found"] else (0, 140, 255)
        cv2.ellipse(out, p, (c["rx"], c["ry"]), 0, 0, 360, colour, 1)
        cv2.putText(out, str(c["idx"]), (p[0] - 14, p[1] - c["ry"] - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(out, "green = detector saw a hole here   orange = it did not", (20, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(out, "green = detector saw a hole here   orange = it did not", (20, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(path, out)


def draw_plots(cells, path):
    """Two scatter plots, drawn with OpenCV so there is no matplotlib to install."""
    W, H, pad = 520, 520, 50
    canvas = np.full((H, W * 2 + 20, 3), 24, np.uint8)

    def panel(ox, xs, ys, xlabel, ylabel, title):
        xs, ys = np.asarray(xs, float), np.asarray(ys, float)
        xr = (xs.min(), xs.max()) if xs.max() > xs.min() else (xs.min() - 1, xs.max() + 1)
        yr = (ys.min(), ys.max()) if ys.max() > ys.min() else (ys.min() - 1, ys.max() + 1)
        cv2.rectangle(canvas, (ox + pad, pad), (ox + W - pad, H - pad), (70, 70, 70), 1)
        cv2.putText(canvas, title, (ox + pad, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(canvas, xlabel, (ox + W // 2 - 30, H - 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (170, 170, 170), 1, cv2.LINE_AA)
        cv2.putText(canvas, ylabel, (ox + 6, pad - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (170, 170, 170), 1, cv2.LINE_AA)
        for cell, xv, yv in zip(cells, xs, ys):
            px = ox + pad + int((xv - xr[0]) / (xr[1] - xr[0]) * (W - 2 * pad))
            py = H - pad - int((yv - yr[0]) / (yr[1] - yr[0]) * (H - 2 * pad))
            cv2.circle(canvas, (px, py), 5, cell["bgr"], -1)
            # Orange ring where no hole was detected, so the sample rests
            # entirely on the warp.
            edge = (200, 200, 200) if cell["found"] else (0, 140, 255)
            cv2.circle(canvas, (px, py), 5, edge, 1)
        for corner, val in (((ox + pad, H - pad + 18), xr[0]),
                            ((ox + W - pad - 40, H - pad + 18), xr[1])):
            cv2.putText(canvas, f"{val:.0f}", corner, cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, (150, 150, 150), 1, cv2.LINE_AA)

    panel(0, [c["a"] for c in cells], [c["b"] for c in cells],
          "a*  (green <-> red)", "b*  (blue <-> yellow)", "colour plane")
    panel(W + 20, [c["L"] for c in cells], [c["texture"] for c in cells],
          "L*  (brightness)", "texture", "brightness vs texture")
    cv2.imwrite(path, canvas)


# ---------------------------------------------------------------- main

EXPOSURE_FILE = "exposure.json"

# The camera's own exposure-time-abs control runs 1 to 5000 and auto picks about
# 157, which blows the balls out. These are the settings worth trying, spaced
# geometrically so each step is roughly half a stop.
EXPOSURE_CANDIDATES = [4, 6, 9, 13, 19, 27, 38, 55, 78, 110, 157, 220]

UVC_MANUAL, UVC_AUTO = 1, 8


def uvc_util(index, *args):
    """Talk to the camera's UVC controls through uvc-util.

    OpenCV cannot set exposure on this camera under macOS: the property is
    accepted and silently ignored. uvc-util walks the USB bus with IOKit and
    sets the control on the device itself, which the capture then inherits.

    The binary sits next to this file and is not in the repo. To rebuild it:

        git clone --depth 1 https://github.com/jtfrey/uvc-util
        clang -fno-objc-arc -O2 -Wno-everything -framework Foundation \
              -framework IOKit -framework CoreFoundation \
              uvc-util/src/*.m -o uvc-util
    """
    exe = shutil.which("uvc-util") or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "uvc-util")
    if not os.path.exists(exe):
        return None
    try:
        done = subprocess.run([exe, "-I", str(index), *args],
                              capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout.strip() if done.returncode == 0 else None


def clipped_fraction(frame, corners=None):
    """How much of the board is pinned at 255.

    Restricted to the board, because the strips themselves are in shot and a
    light source is clipped by definition. Measuring the whole frame chased a
    floor it could never reach and drove the exposure four stops too dark.
    """
    blown = frame.max(axis=2) >= 250
    if corners is None:
        return float(blown.mean())
    board = cv2.fillPoly(np.zeros(blown.shape, np.uint8),
                         np.int32([corners]), 255) > 0
    return float(blown[board].mean()) if board.any() else float(blown.mean())


class quiet:
    """Hold the OS quiet while the camera opens.

    macOS logs a deprecation notice about Continuity Cameras straight to fd 2
    from inside AVFoundation, every single run. It is not ours to fix and it
    says nothing useful, so fd 2 is parked on /dev/null for the one call that
    provokes it and put straight back. Anything we actually want to report
    about the camera, we report ourselves.
    """

    def __enter__(self):
        self.saved = os.dup(2)
        self.null = os.open(os.devnull, os.O_WRONLY)
        os.dup2(self.null, 2)

    def __exit__(self, *_):
        os.dup2(self.saved, 2)
        os.close(self.null)
        os.close(self.saved)


def open_camera(args):
    with quiet():
        cap = cv2.VideoCapture(args.index, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        print(f"Could not open camera index {args.index}.")
        print("On macOS the terminal app needs camera permission under "
              "Privacy & Security.")
        return None
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    return cap


def grab(cap, n=8):
    """Throw away a few frames so the sensor has settled before we look."""
    ok, frame = False, None
    for _ in range(n):
        ok, frame = cap.read()
    return frame if ok else None


def set_exposure(index, value):
    """Fix the exposure, or hand it back to the camera when value is None."""
    if value is None:
        return uvc_util(index, "-s", f"auto-exposure-mode={UVC_AUTO}") is not None
    ok = uvc_util(index, "-s", f"auto-exposure-mode={UVC_MANUAL}") is not None
    return ok and uvc_util(index, "-s", f"exposure-time-abs={int(value)}") is not None


def dial_exposure(cap, index, target, corners):
    """Sweep the exposure and keep the brightest setting that does not clip.

    Balls are the brightest thing on the board, so they are what blows out
    first, and a blown pixel has lost its colour for good. Better a slightly
    dark picture than a white one.
    """
    if uvc_util(index, "-o", "exposure-time-abs") is None:
        print("  uvc-util is not answering, so the exposure cannot be set.")
        print("  It should sit next to this script, or on the PATH.")
        return None

    set_exposure(index, None)          # start from the camera's own choice
    base = grab(cap)
    if base is None:
        return None
    print(f"  auto exposure gives {clipped_fraction(base, corners) * 100:.2f}% clipped. "
          f"Sweeping for something under {target * 100:.2f}%.")

    results = []
    for value in EXPOSURE_CANDIDATES:
        if not set_exposure(index, value):
            continue
        frame = grab(cap, 6)
        if frame is None:
            continue
        c = clipped_fraction(frame, corners)
        results.append((value, c, cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean()))
        print(f"    exposure {value:>6}  ->  {c * 100:6.2f}% clipped, "
              f"mean {results[-1][2]:5.1f}")

    if not results:
        return None
    spread = max(c for _, c, _ in results) - min(c for _, c, _ in results)
    if spread < 0.002:
        print("\n  Nothing changed across the whole range, which should not "
              "happen now that the control is being set on the device.")
        print("  Check `uvc-util -I 0 -o exposure-time-abs` by hand.")
        return None

    usable = [r for r in results if r[1] <= target]
    if not usable:
        value, clip, _ = min(results, key=lambda r: r[1])
        print(f"\n  Nothing got under {target * 100:.2f}%. Best was exposure "
              f"{value} at {clip * 100:.2f}%. Take a strip out as well.")
        return value
    # The brightest of the clean settings: longest exposure means least noise,
    # and noise is what is left once clipping is dealt with.
    value, clip, _ = max(usable, key=lambda r: r[0])
    print(f"\n  Using exposure {value}: {clip * 100:.2f}% clipped.")
    return value


def get_frame(args, corners):
    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            print(f"Could not read {args.image}")
            return None
        return frame

    cap = open_camera(args)
    if cap is None:
        return None

    exposure = args.exposure
    if args.dial_exposure:
        exposure = dial_exposure(cap, args.uvc_index, args.clip_target, corners)
        if exposure is not None:
            with open(EXPOSURE_FILE, "w") as fh:
                json.dump({"exposure": exposure}, fh)
            print(f"  Saved to {EXPOSURE_FILE}; later runs will reuse it.")
    elif exposure is None and not args.ignore_saved and os.path.exists(EXPOSURE_FILE):
        with open(EXPOSURE_FILE) as fh:
            exposure = json.load(fh)["exposure"]
        print(f"Using exposure {exposure} from {EXPOSURE_FILE} "
              "(--dial-exposure to redo, --exposure auto for the camera's own).")

    # None means hand it back to the camera, which is what --exposure auto asks
    # for; anything else pins it.
    if exposure is not None or args.ignore_saved:
        set_exposure(args.uvc_index, exposure)

    frame = grab(cap, 10)
    cap.release()
    if frame is not None:
        c = clipped_fraction(frame, corners)
        print(f"Frame is {c * 100:.2f}% clipped.")
    return frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--image", type=str, default=None)
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--corners", type=str, default="corners.json")
    ap.add_argument("--reclick", action="store_true",
                    help="redo the corner clicks; implies --recalibrate")
    ap.add_argument("--recalibrate", action="store_true",
                    help="ignore the tags and re-fit the grid from the holes, "
                         f"rewriting {REFERENCE_FILE}. Needs the room lit.")
    ap.add_argument("--radius", type=int, default=None,
                    help="force every sample to a circle of this radius in px, "
                         "instead of an ellipse fitted to each hole")
    ap.add_argument("--sens", type=float, default=1.25,
                    help="hole detector: how much brighter than its surroundings "
                         "a hole has to be. Lower finds dimmer holes.")
    ap.add_argument("--teach", type=str, default=None, metavar="COLOUR",
                    help="learn this colour from a board holding only it, and "
                         f"re-learn empty from the rest. Writes {PROTOTYPE_FILE}.")
    ap.add_argument("--no-gui", action="store_true",
                    help="never open a window; requires a cached corners file")
    ap.add_argument("--dial-exposure", action="store_true",
                    help="sweep the exposure once, keep the brightest setting "
                         f"that does not blow out the balls, save to {EXPOSURE_FILE}")
    ap.add_argument("--exposure", type=str, default=None,
                    help=f"a fixed exposure value, or 'auto' to ignore "
                         f"{EXPOSURE_FILE} and let the camera decide")
    ap.add_argument("--uvc-index", type=int, default=0,
                    help="which camera uvc-util sees, from `uvc-util -d`")
    ap.add_argument("--clip-target", type=float, default=0.001,
                    help="how much of the frame --dial-exposure will let clip "
                         "at 255 (default 0.001, i.e. a tenth of a percent)")
    args = ap.parse_args()
    args.recalibrate = args.recalibrate or args.reclick
    if args.exposure is not None:
        args.exposure = None if args.exposure.lower() == "auto" else float(args.exposure)
        args.ignore_saved = True
    else:
        args.ignore_saved = False

    # Read the corners first where we can: the exposure sweep needs to know
    # which part of the frame is board, so that the strips in shot do not count
    # as clipping.
    corners = None
    if os.path.exists(args.corners) and not args.reclick:
        with open(args.corners) as fh:
            corners = json.load(fh)

    frame = get_frame(args, corners)
    if frame is None:
        return 1
    if not args.image:
        cv2.imwrite("frame_last.png", frame)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    tags = detect_tags(gray)
    print(f"Tags seen: {sorted(tags) if tags else 'none'}")

    # Once the tags can place the grid there is nothing left for the corner
    # clicks to do: they only ever bounded the hole search and gave the row
    # grouping a rough rectification, and both belong to the fitting path.
    # --recalibrate forces that path, to record the reference again.
    cells, residual = (None, None)
    if not args.recalibrate:
        cells, how = cells_from_tags(tags)
        if cells is not None:
            seen, fit, H = how
            print(f"Placed from tags {seen}, agreeing to {fit:.1f} px. "
                  "No corners needed.")
            grew = accumulate_reference(gray, cells, H, args.sens)
            if grew:
                print(f"  Confirmed {grew[0]} cells against holes seen now; "
                      f"{grew[1]} of {COLS * ROWS} pinned so far.")
                cells, _ = cells_from_tags(tags)

    if cells is None:
        if corners is None:
            if args.no_gui:
                print("No cached corners and --no-gui was given.")
                return 1
            corners = click_corners(frame)
            if corners is None:
                print("Aborted.")
                return 1
            with open(args.corners, "w") as fh:
                json.dump(corners, fh)
            print(f"Saved corners to {args.corners}.")

        else:
            print(f"Using cached corners from {args.corners} "
                  "(--reclick to redo).")

        pitch = float(np.linalg.norm(np.float32(corners[1]) - np.float32(corners[0]))
                      / (COLS - 1))
        blobs = find_holes(gray, corners, pitch, args.sens)
        print(f"Found {len(blobs)} holes in the frame "
              f"(expecting up to {COLS * ROWS}).")
        # With the room dark only the balls show up, which is the point, so a
        # low count is fine as long as there is a saved geometry to fall on.
        if len(blobs) < 16 and load_warp() is None:
            print("Too few to work with, and no saved geometry. Try a lower "
                  "--sens, or run once with the room lights on.")
            return 1
        cells, residual = locate_cells(blobs, corners, pitch)
        if cells is None:
            return 1
    # residual is None when the tags placed the grid; they have said their piece
    # already and none of the hole counts mean anything in that case.
    if residual is not None:
        found = sum(c["found"] for c in cells)
        print(f"The detector saw {found} of {len(cells)} holes; "
              "the warp covers the rest.")
        if residual:
            print(f"Warp fits the found holes to {np.median(residual):.1f} px median, "
                  f"{max(residual):.1f} px worst.")

    for c in cells:
        if args.radius:
            c["rx"] = c["ry"] = args.radius
        s = sample_cell(frame, gray, c["x"], c["y"], c["rx"], c["ry"])
        if s is None:
            s = {"L": 0.0, "a": 0.0, "b": 0.0, "texture": 0.0,
                 "clipped": 0.0, "mean_gray": 0.0}
        c.update(s)
        c["bgr"] = tuple(int(v) for v in cv2.cvtColor(
            np.uint8([[[s["L"] * 255 / 100, s["a"] + 128, s["b"] + 128]]]),
            cv2.COLOR_LAB2BGR)[0][0])

    with open("cells.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["idx", "row", "col", "x", "y", "rx", "ry", "found",
                    "colour", "distance",
                    "L", "a", "b", "texture", "clipped", "mean_gray"])
        for c in cells:
            w.writerow([c["idx"], c["row"], c["col"], round(c["x"], 1), round(c["y"], 1),
                        c["rx"], c["ry"], c["found"],
                        c.get("colour", ""), round(c.get("distance", 0.0), 2),
                        round(c["L"], 2), round(c["a"], 2), round(c["b"], 2),
                        round(c["texture"], 1), round(c["clipped"], 3),
                        round(c["mean_gray"], 1)])

    if args.teach:
        teach(cells, args.teach)
    else:
        protos = load_prototypes()
        if protos:
            classify(cells, protos)
            tally = {}
            for c in cells:
                tally[c["colour"]] = tally.get(c["colour"], 0) + 1
            named = {k: v for k, v in sorted(tally.items()) if k != "empty"}
            print(f"\nRead: {named}, and {tally.get('empty', 0)} empty.")
        else:
            print(f"\nNo {PROTOTYPE_FILE} yet. Put one colour on the board and "
                  "run with --teach <colour>, once per colour.")

    blown = [c["idx"] for c in cells if c["clipped"] > 0.02]
    if blown:
        print(f"\n{len(blown)} cells have over 2% of their pixels clipped at 255: "
              f"{blown[:12]}{' ...' if len(blown) > 12 else ''}")
        print("  A clipped pixel has lost its colour, and pale balls clip first.")
        print("  Shorten the exposure or take a strip out before reading anything "
              "into the colours.")

    # A good fit plus tags in shot is the moment to record where the grid sits
    # relative to them, so that later frames can find it again after the board
    # has been knocked.
    if residual and len(tags) >= 3:
        save_reference(tags, cells)
        print(f"Recorded the grid against tags {sorted(tags)} "
              f"in {REFERENCE_FILE}.")

    draw_map(frame, cells, "cells_map.png")
    draw_plots(cells, "cells_plot.png")

    for key, name in (("L", "brightness L*"), ("texture", "texture")):
        vals = [c[key] for c in cells]
        thresh, gap = otsu_split(vals)
        low = sum(1 for v in vals if v <= thresh)
        print(f"\n{name}: range {min(vals):.1f} to {max(vals):.1f}")
        print(f"  natural split at {thresh:.1f}  ->  {low} below, {len(vals) - low} above")
        print(f"  gap between the two groups: {gap:.1f}"
              f"   ({'clean' if gap > 0.08 * (max(vals) - min(vals)) else 'weak, they overlap'})")

    print(f"\n{len(cells)} cells written to cells.csv")
    print("Look at cells_map.png to check the discs landed on the holes,")
    print("and cells_plot.png to see whether things cluster.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
