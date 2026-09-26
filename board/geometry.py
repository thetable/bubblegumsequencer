"""Where the 64 cells are in the frame.

Three ways of knowing, in descending order of how much they can be trusted:
fit a warp to the holes visible right now, move a stored grid onto the
AprilTags, or reuse the last fit verbatim. The first is the most accurate and
needs room light; the second works in any light and survives the board being
moved; the third is only correct while nothing has shifted.

The warp is a polynomial rather than a homography because the lens bends the
column pitch from 63 px at the frame edge to 141 px mid-board, and a
plane-to-plane map cannot follow that. Fitting four clicked corners through a
homography was the original approach and it was wrong by up to 94 px.
"""

import json
import os

import cv2
import numpy as np

COLS, ROWS = 16, 4
WARP_FILE = "warp.json"
REFERENCE_FILE = "tag_reference.json"
TAG_DICT = cv2.aruco.DICT_APRILTAG_36h11

# How far out to look when working out what a hole should be brighter than.
# 90 was too far: it reached up into the LED strips sitting just beyond the
# top and bottom rows, inflated the local reference there, and lost half of
# row 0 and a third of row 3. 40 keeps the strips out. Much below that and it
# starts averaging the hole into its own background.
BACKGROUND_SIGMA = 40


def _round_blobs(mask, gray, corners, pitch):
    """Round, same-sized things inside the board, from a thresholded mask."""
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


def find_holes(gray, corners, pitch, sens, polarity=None):
    """Every round hole-sized blob inside the board area.

    Dividing by a heavily blurred copy removes the lens's vignetting and the
    box's uneven lighting, so one ratio works from the middle of the board to
    the corner. What survives is scale-free: a hole differs from the sheet
    around it by the same factor wherever it is.

    Which way it differs depends on the rig, and it inverted when the box was
    built. In the open crate the sheet was unlit and the room shone down
    through the holes, so a hole was the bright thing. Enclosed, with strips
    lighting the sheet from below and a dim room above, the sheet is the
    bright thing and a hole is a dark disc. Measured on the same board an hour
    apart: 19 bright blobs one way, 39 dark the other.

    So it tries both and keeps whichever finds more. That is not the
    two-sided search that failed twice before, which thresholded once and took
    bright and dark together, letting the sheet between the holes form its own
    blobs and crowd out the real ones. These are two separate searches and
    only the better one survives.
    """
    g = gray.astype(np.float32)
    background = cv2.GaussianBlur(g, (0, 0), BACKGROUND_SIGMA)
    ratio = g / (background + 1e-6)

    tries = {"bright": ratio > sens, "dark": ratio < 1.0 / sens}
    if polarity in tries:
        tries = {polarity: tries[polarity]}
    found = {k: _round_blobs(m.astype(np.uint8) * 255, gray, corners, pitch)
             for k, m in tries.items()}
    best = max(found, key=lambda k: len(found[k]))
    return found[best]


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


def bend(u, k):
    """Push column positions out towards the edges, by k.

    The four clicks give a perspective map, and a perspective map cannot bend:
    it spaces the columns evenly. This lens does not. Its pitch is a hump,
    widest mid-board and tightest at both ends, near enough two to one, so a
    blob near the edge sits a column or more from where the flat map puts it.

    One number describes that, and it is cheaper to search for it than to
    derive it: a cubic term, tried across a range, keeping whichever value
    makes the holes fall closest to whole columns.
    """
    t = np.asarray(u, float) / (COLS - 1) * 2 - 1
    return ((t + k * t ** 3) / (1 + k) + 1) / 2 * (COLS - 1)


def seed_from_corners(blobs, corners, k=0.0):
    """A column and row for every blob, read off the four clicked corners.

    Needed because a complete row is a lot to ask. It wants all sixteen holes
    of one row detected at once, and in a finished box, where the sheet and
    the holes are lit from opposite sides, most frames never manage it: 43 of
    64 found and not one row whole.

    The clicks are the ground truth that is always available. They say where
    cells 0, 15, 63 and 48 are, and a perspective map through them puts every
    other blob roughly in its place. Roughly is enough. The fit that follows
    re-reads its labels from its own prediction, four times over, so an edge
    hole that started one column out is pulled back as soon as the warp has a
    shape. What made this fail before was trusting the result; it is checked
    now, twice, and a wrong answer is refused rather than saved.
    """
    board = np.float32([[0, 0], [COLS - 1, 0], [COLS - 1, ROWS - 1], [0, ROWS - 1]])
    flat = cv2.perspectiveTransform(
        np.float32([[[b["x"], b["y"]] for b in blobs]]),
        cv2.getPerspectiveTransform(np.float32(corners), board))[0]
    flat = np.column_stack([bend(flat[:, 0], k), flat[:, 1]])

    # One blob per cell: two claiming the same one means at least one is wrong,
    # so keep whichever sits closer to the middle of it.
    best = {}
    for i, (u, v) in enumerate(flat):
        col = int(round(min(max(u, 0.0), COLS - 1.0)))
        row = int(round(min(max(v, 0.0), ROWS - 1.0)))
        off = abs(u - col) + abs(v - row)
        cell = row * COLS + col
        if cell not in best or off < best[cell][0]:
            best[cell] = (off, i)
    return ([c % COLS for c in best], [c // COLS for c in best],
            [blobs[i]["x"] for _, i in best.values()],
            [blobs[i]["y"] for _, i in best.values()])


def best_seed(blobs, corners):
    """Label the blobs from the clicks, searching for how much the lens bends.

    Scored on the thing that cannot be faked: a fit to the right labels leaves
    small residuals AND comes out with the pitch humped the correct way. A fit
    to the wrong ones can manage the first but not the second, because holes
    it has shuffled one column along force the middle of the board narrow.
    """
    best = ([], [], [], [], 0.0, np.inf)
    for k in np.arange(-0.4, 0.81, 0.05):
        u, v, x, y = seed_from_corners(blobs, corners, k)
        if len(u) < 12 or len(set(v)) < 2:
            continue
        cx, cy = fit_warp(u, v, x, y, 1)
        grid = warp_basis(np.tile(np.arange(COLS), ROWS),
                          np.repeat(np.arange(ROWS), COLS), 1)
        px = grid @ cx
        middle = np.diff(px[COLS // 2 - 2:COLS // 2 + 2]).mean()
        edges = max(np.diff(px[:3]).mean(), np.diff(px[-3:]).mean())
        if middle <= edges:                    # dished: the labels are wrong
            continue
        fitted = warp_basis(u, v, 1)
        err = float(np.median(np.hypot(fitted @ cx - np.asarray(x),
                                       fitted @ cy - np.asarray(y))))
        # Prefer the fit that explains the most holes, then the tightest one.
        score = err / max(len(u), 1) ** 0.5
        if score < best[5]:
            best = (u, v, x, y, float(k), score)
    return best[:5]


def load_warp():
    if not os.path.exists(WARP_FILE):
        return None
    with open(WARP_FILE) as fh:
        w = json.load(fh)
    return np.array(w["cx"]), np.array(w["cy"]), int(w["v_degree"])


def save_warp(cx, cy, v_degree):
    with open(WARP_FILE, "w") as fh:
        json.dump({"cx": list(cx), "cy": list(cy), "v_degree": v_degree}, fh)


def locate_cells(blobs, corners, pitch, allow_cached=True):
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
    if not fitting and blobs:
        u, v, x, y, k = best_seed(blobs, corners)
        fitting = len(u) >= 12          # the warp has twelve coefficients
        if fitting:
            print(f"  No row came up whole, so the four clicks place the "
                  f"columns instead: {len(u)} holes labelled, bend {k:+.2f}.")
    if not fitting:
        # Reusing the cached warp is only honest when nothing has moved. Asked
        # to recalibrate, the caller is saying something has, so falling back
        # to the old geometry produces a reference that looks freshly made and
        # is a row out. That failure cost an afternoon: the file dates said
        # recalibrated, the grid said otherwise.
        saved = load_warp() if allow_cached else None
        if saved is None and not allow_cached:
            print(f"  Need two rows with all 16 holes to anchor the columns; "
                  f"got {len(complete)}.")
            print(f"  Not falling back to {WARP_FILE}: you asked to "
                  "recalibrate, so it describes a rig that no longer exists.")
            print("\n  Holes show when they differ from the sheet, either way "
                  "round. Both lit at once is the worst case, and that is")
            print("  what this frame looks like. Try the strips OFF with a "
                  "lamp above: the sheet goes black and every hole lights up.")
            return None, None
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
    # The clicks are the one thing measured by hand, so the fit has to agree
    # with them. A grid that has locked onto the wrong columns lands a whole
    # cell off here while fitting its own detections beautifully, which is
    # exactly how the earlier attempt at this went wrong unnoticed.
    if fitting:
        want = np.float32(corners)
        ends = [0, COLS - 1, COLS * ROWS - 1, COLS * (ROWS - 1)]
        got = np.float32([[px[c], py[c]] for c in ends])
        off = np.hypot(*(got - want).T)
        print(f"  Corners land {off.min():.0f} to {off.max():.0f} px from "
              "where you clicked.")
        if off.max() > 0.6 * pitch:
            print(f"  More than {0.6 * pitch:.0f} px out, so the grid is not "
                  "the one you pointed at. Stopping.")
            print("  Usually a corner click that missed its hole, or a board "
                  "that moved between the clicks and now.")
            return None, None
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
