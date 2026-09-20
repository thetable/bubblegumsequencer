#!/usr/bin/env python3
"""A window on what the camera can see, for when something is wrong.

Shows every cell where the geometry thinks it is, what colour it is reading,
and how far the board has drifted since the geometry was recorded. The same
loop the instrument runs, with a picture attached.

    python inspect.py                    # the live window
    python inspect.py --once             # one frame to cells.csv and two plots
    python inspect.py --image frame.png  # look at a saved frame instead

Keys:
    q       quit
    space   freeze and unfreeze
    s       save the frame and the annotated view
    c       cycle what the cells are filled with

--learn pins each cell's position more firmly as holes become visible, which
fills in ones that no single frame can show. It is safe to leave on.
"""

import argparse
import csv
import sys
import time

import cv2
import numpy as np

from board import camera, colour, pattern, stages
from board.reader import read_board
from board.view import FILLS, draw


def write_csv(cells, path="cells.csv"):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["idx", "row", "col", "x", "y", "rx", "ry", "found",
                    "colour", "distance",
                    "L", "a", "b", "texture", "clipped", "mean_gray"])
        for c in cells:
            w.writerow([c["idx"], c["row"], c["col"],
                        round(c["x"], 1), round(c["y"], 1), c["rx"], c["ry"],
                        c["found"], c.get("colour", ""),
                        round(c.get("distance", 0.0), 2),
                        round(c["L"], 2), round(c["a"], 2), round(c["b"], 2),
                        round(c["texture"], 1), round(c["clipped"], 3),
                        round(c["mean_gray"], 1)])


def draw_map(frame, cells, path):
    """Every disc where the geometry says. Colour says whether the detector
    also saw a hole there, which is a lighting check, not a position check."""
    out = frame.copy()
    for c in cells:
        p = (int(c["x"]), int(c["y"]))
        shade = (0, 255, 0) if c["found"] else (0, 140, 255)
        cv2.ellipse(out, p, (c["rx"], c["ry"]), 0, 0, 360, shade, 1)
        cv2.putText(out, str(c["idx"]), (p[0] - 14, p[1] - c["ry"] - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1, cv2.LINE_AA)
    for thick, tone in ((4, (0, 0, 0)), (1, (255, 255, 255))):
        cv2.putText(out, "green = detector saw a hole here   orange = it did not",
                    (20, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.7, tone, thick, cv2.LINE_AA)
    cv2.imwrite(path, out)


def draw_plots(cells, path):
    """Two scatter plots, drawn with OpenCV so there is no matplotlib to add."""
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


def report(frame, cells):
    """The numbers worth looking at when something is not separating."""
    write_csv(cells)
    draw_map(frame, cells, "cells_map.png")
    draw_plots(cells, "cells_plot.png")

    blown = [c["idx"] for c in cells if c["clipped"] > 0.02]
    if blown:
        print(f"\n{len(blown)} cells have over 2% of their pixels clipped at 255: "
              f"{blown[:12]}{' ...' if len(blown) > 12 else ''}")
        print("  A clipped pixel has lost its colour, and pale balls clip first.")
        print("  Run setup.py to re-dial the exposure.")

    for key, name in (("L", "brightness L*"), ("texture", "texture")):
        vals = [c[key] for c in cells]
        thresh, gap = colour.otsu_split(np.array(vals))
        low = sum(1 for v in vals if v <= thresh)
        print(f"\n{name}: range {min(vals):.1f} to {max(vals):.1f}")
        print(f"  natural split at {thresh:.1f}  ->  {low} below, "
              f"{len(vals) - low} above")
        print(f"  gap between the two groups: {gap:.1f}"
              f"   ({'clean' if gap > 0.08 * (max(vals) - min(vals)) else 'weak'})")
    print("\ncells.csv, cells_map.png and cells_plot.png written")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int, default=None,
                    help="OpenCV camera index; camera.json otherwise")
    ap.add_argument("--image", type=str, default=None)
    ap.add_argument("--once", action="store_true",
                    help="one frame to cells.csv and the two plots, then exit")
    ap.add_argument("--shot", type=str, default=None,
                    help="render one annotated frame to this file and exit")
    ap.add_argument("--sheet", type=str, default=None, metavar="FILE",
                    help="every pipeline stage as one image, for a slide")
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--uvc-index", type=int, default=None,
                    help="uvc-util camera index; camera.json otherwise")
    ap.add_argument("--display-width", type=int, default=1280)
    ap.add_argument("--no-mirror", action="store_true",
                    help="show the camera's own view instead of flipping it to "
                         "match the board as you see it from above")
    ap.add_argument("--learn", action="store_true",
                    help="pin each cell's position as holes become visible")
    args = ap.parse_args()
    camera.resolve(args)
    fill = FILLS[0]

    if args.sheet:
        frame = (cv2.imread(args.image) if args.image
                 else camera.single_frame(args))
        if frame is None:
            print("no frame")
            return 1
        cv2.imwrite(args.sheet, stages.contact_sheet(frame))
        print(f"wrote {args.sheet}")
        return 0

    if args.image or args.once:
        frame = (cv2.imread(args.image) if args.image
                 else camera.single_frame(args))
        if frame is None:
            print("no frame")
            return 1
        cells, tags, note, drift = read_board(frame, args.learn)
        print(note)
        if cells is None:
            return 1
        if args.once:
            report(frame, cells)
        view = draw(frame, cells, tags, note, fill, drift, not args.no_mirror)
        if args.shot:
            cv2.imwrite(args.shot, view)
            print(f"wrote {args.shot}")
        return 0

    cap = camera.open_camera(args)
    if cap is None:
        return 1
    pinned = camera.settings().get("exposure")
    if pinned is not None:
        camera.set_exposure(args.uvc_index, pinned)
        print(f"exposure pinned at {pinned}")

    win = "bubblegum, what the camera sees"
    cv2.namedWindow(win)
    frozen, held, last, fps, frames = False, None, time.perf_counter(), 0.0, 0
    board = pattern.Stabiliser()

    while True:
        if not frozen:
            ok, frame = cap.read()
            if not ok:
                break
            frames += 1
            cells, tags, note, drift = read_board(
                frame, args.learn and frames % 10 == 0)
            held = (frame, cells, tags, note, drift)
            now = time.perf_counter()
            fps = 0.8 * fps + 0.2 / max(now - last, 1e-6)
            last = now
        frame, cells, tags, note, drift = held

        # The pattern only moves on a reading that has held; anything the
        # vision side could not say becomes no data, which holds it instead.
        for i, was, now_is in board.update(
                pattern.readings_from(cells, cells is not None)):
            print(f"  cell {i}: {was} -> {now_is}")
        waiting = board.pending()
        for c in cells or []:
            c["settled"] = board.pattern[c["idx"]]
            c["wanted"] = waiting.get(c["idx"])
        state = ("HOLDING, cannot see the board" if board.blind else
                 "HOLDING, too much changed at once" if board.blocked else
                 f"{len(board.filled())} balls")

        view = draw(frame, cells, tags,
                    f"{state}   {note}   {fps:4.1f} fps"
                    + ("   FROZEN" if frozen else ""),
                    fill, drift, not args.no_mirror)
        scale = args.display_width / view.shape[1]
        cv2.imshow(win, cv2.resize(view, None, fx=scale, fy=scale))

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord(" "):
            frozen = not frozen
            last = time.perf_counter()
        if key == ord("c"):
            fill = FILLS[(FILLS.index(fill) + 1) % len(FILLS)]
        if key == ord("s"):
            stamp = time.strftime("%H%M%S")
            cv2.imwrite(f"live_{stamp}.png", frame)
            cv2.imwrite(f"live_{stamp}_view.png", view)
            print(f"saved live_{stamp}.png and live_{stamp}_view.png")

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
