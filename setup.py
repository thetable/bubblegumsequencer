#!/usr/bin/env python3
"""Getting the rig ready. Everything you do once, in the order it has to happen.

    python setup.py              # the whole walk
    python setup.py --exposure   # just one step
    python setup.py --geometry
    python setup.py --colours

The order is not arbitrary. Exposure first, because the geometry and the
colours are both read off a correctly exposed frame and neither is worth
measuring until it is. Geometry second, because teaching a colour means
sampling cells, which means knowing where they are.

If the lens itself has moved, run focus_check.py before any of this.
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np

from board import camera, colour, geometry
from board.reader import read_board
from board.view import draw

CORNERS_FILE = "corners.json"
DEFAULT_COLOURS = ("pink", "yellow", "blue", "green")
CORNER_PROMPTS = [
    "top-left hole (row 0, col 0)",
    "top-right hole (row 0, col 15)",
    "bottom-right hole (row 3, col 15)",
    "bottom-left hole (row 3, col 0)",
]


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


def step_exposure(args):
    """Pin the exposure so that nothing clips and the room stops competing."""
    print("\n1. EXPOSURE")
    print("   Put a few balls on the board, including a pale one: they are")
    print("   what blows out, so they are what this has to see.")
    input("   Press return when ready. ")

    cap = camera.open_camera(args)
    if cap is None:
        return False
    corners = json.load(open(CORNERS_FILE)) if os.path.exists(CORNERS_FILE) else None
    value = camera.dial_exposure(cap, args.uvc_index, args.clip_target, corners)
    cap.release()
    if value is None:
        print("   Could not set the exposure. Is uvc-util next to these scripts?")
        return False
    with open(camera.EXPOSURE_FILE, "w") as fh:
        json.dump({"exposure": value}, fh)
    print(f"   Saved. Every run from now on pins it at {value}.")
    return True


def step_geometry(args):
    """Fit the grid to the holes, then record it against the tags."""
    print("\n2. GEOMETRY")
    print("   Room lights ON for this one: it has to see the empty holes, and")
    print("   an empty hole is only visible because the room shines through it.")
    input("   Press return when ready. ")

    frame = camera.single_frame(args)
    if frame is None:
        return False
    cv2.imwrite("frame_last.png", frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    tags = geometry.detect_tags(gray)
    print(f"   Tags seen: {sorted(tags) if tags else 'none'}")
    if len(tags) < 3:
        print("   Need at least three. Check they are flat, in view, and not")
        print("   sitting in the glare from a strip.")
        return False

    corners = None
    if os.path.exists(CORNERS_FILE) and not args.reclick:
        corners = json.load(open(CORNERS_FILE))
        print(f"   Using the corners in {CORNERS_FILE}. --reclick to redo them.")
    else:
        print("   Click the four corner holes, in the order it asks.")
        corners = click_corners(frame)
        if corners is None:
            return False
        json.dump(corners, open(CORNERS_FILE, "w"))

    pitch = float(np.linalg.norm(np.float32(corners[1]) - np.float32(corners[0]))
                  / (geometry.COLS - 1))
    blobs = geometry.find_holes(gray, corners, pitch, args.sens)
    print(f"   Found {len(blobs)} holes of {geometry.COLS * geometry.ROWS}.")
    cells, residual = geometry.locate_cells(blobs, corners, pitch)
    if cells is None:
        return False
    print(f"   Warp fits them to {np.median(residual):.1f} px median, "
          f"{max(residual):.1f} px worst.")
    geometry.save_reference(tags, cells)
    print(f"   Recorded against tags {sorted(tags)}. The board can move now.")
    return True


def step_colours(args):
    """Teach one colour at a time, guided, in a window."""
    print("\n3. COLOURS")
    queue = [c.strip() for c in args.colour_names.split(",")]
    print("   " + ", ".join(queue) + ", one at a time. The window says what to do.")

    cap = camera.open_camera(args)
    if cap is None:
        return False
    if os.path.exists(camera.EXPOSURE_FILE):
        with open(camera.EXPOSURE_FILE) as fh:
            camera.set_exposure(args.uvc_index, json.load(fh)["exposure"])

    win = "bubblegum, teaching"
    cv2.namedWindow(win)
    while queue:
        ok, frame = cap.read()
        if not ok:
            break
        cells, tags, note, drift = read_board(frame)
        kept, strays, _ = (colour.split_board(cells) if cells and "L" in cells[0]
                           else ([], [], []))
        ready = "press SPACE to learn it" if kept else "waiting for balls"
        aside = (f"   ignoring {len(strays)} bright but wrong-coloured"
                 if strays else "")
        banner = [f"Put only the {queue[0].upper()} balls on the board",
                  f"I can see {len(kept)}.{aside}  {ready}.",
                  f"{len(queue)} to go   s skip this colour   q give up"]
        view = draw(frame, cells, tags, note, "nothing", drift,
                    not args.no_mirror, banner,
                    {c["idx"] for c in kept}, {c["idx"] for c in strays})
        scale = args.display_width / view.shape[1]
        cv2.imshow(win, cv2.resize(view, None, fx=scale, fy=scale))

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            print(f"   skipped {queue.pop(0)}")
        if key == ord(" ") and kept and colour.teach(cells, queue[0]) is not None:
            print(f"   learnt {queue.pop(0)}")

    cap.release()
    cv2.destroyAllWindows()
    return not queue


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--uvc-index", type=int, default=0)
    ap.add_argument("--display-width", type=int, default=1280)
    ap.add_argument("--no-mirror", action="store_true")
    ap.add_argument("--sens", type=float, default=1.25,
                    help="how much brighter than its surroundings a hole has "
                         "to be. Lower finds dimmer holes.")
    ap.add_argument("--clip-target", type=float, default=0.001)
    ap.add_argument("--reclick", action="store_true",
                    help="redo the corner clicks rather than reusing them")
    ap.add_argument("--colour-names", type=str, default=",".join(DEFAULT_COLOURS))
    ap.add_argument("--exposure", action="store_true")
    ap.add_argument("--geometry", action="store_true")
    ap.add_argument("--colours", action="store_true")
    args = ap.parse_args()

    chosen = [n for n, on in (("exposure", args.exposure),
                              ("geometry", args.geometry),
                              ("colours", args.colours)) if on]
    steps = chosen or ["exposure", "geometry", "colours"]
    run = {"exposure": step_exposure, "geometry": step_geometry,
           "colours": step_colours}

    for name in steps:
        if not run[name](args):
            print(f"\nStopped at {name}. Nothing after it has run.")
            return 1
    print("\nReady. Start it with:  python play.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
