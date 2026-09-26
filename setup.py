#!/usr/bin/env python3
"""Getting the rig ready. Everything you do once, in the order it has to happen.

    python setup.py              # the whole walk
    python setup.py --centre     # once, when the box is first put together
    python setup.py --tools      # just one step
    python setup.py --exposure
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

from board import camera, colour, files, framing, geometry
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


# --------------------------------------------------------------------- framing


def step_centre(args):
    """Put the board in the middle of the frame. A build step, done once.

    Not part of the walk, because it is done with a screwdriver rather than a
    keyboard and only when the camera is first mounted. It is here rather than
    in probe.py because getting it wrong is silent: the board still reads
    perfectly, it simply has no room left on one side, and you find out when
    someone nudges it mid-song.
    """
    print("\nCENTRING")
    print("  Aim the camera at the middle of the sensor, NOT at the middle of")
    print("  the board. This lens images its axis at pixel 841 of 1920, so a")
    print("  camera centred under the board sits 119 px off-centre in frame.")
    print("\n  Slide the camera until the two crosses meet. If the gap grows,")
    print("  go the other way. Then tighten it down.")
    print("  q when done, s to save a frame.\n")

    cap = camera.open_camera(args)
    if cap is None:
        return False
    pinned = camera.settings().get("exposure")
    if pinned is not None:
        camera.set_exposure(args.uvc_index, pinned)

    win = "centring: slide the camera until the crosses meet"
    cv2.namedWindow(win)
    best = None
    while True:
        ok, frame = cap.read()
        frame = camera.orient(frame)
        if not ok:
            break
        cells, tags, note, _ = read_board(frame)
        m = framing.margins(frame.shape, cells, tags,
                            geometry.COLS, geometry.ROWS) if cells else None
        # Mirrored, like every other window here. It was not, on the theory
        # that you are moving the camera rather than the board so the camera's
        # own view is the honest one. In practice you stand over the board,
        # every other view matches that, and the odd one out is the one that
        # sends you the wrong way. Being consistent beats being literal.
        view = cv2.flip(frame, 1)
        h, w = view.shape[:2]
        mirror = lambda x: w - 1 - int(x)
        cv2.drawMarker(view, (w // 2, h // 2), (160, 160, 160),
                       cv2.MARKER_CROSS, 60, 2)
        # The tags, always: when placement fails it is nearly always because
        # something is lying across one, and seeing which is half the answer.
        for tag_id, corner in sorted((tags or {}).items()):
            shown = np.int32([[mirror(x), y] for x, y in corner])
            cv2.polylines(view, [shown], True, (255, 0, 255), 2)
            cv2.putText(view, str(tag_id), tuple(shown[0] + [6, -8]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2,
                        cv2.LINE_AA)
        if m:
            x0, y0, x1, y1 = (int(v) for v in m["box"])
            x0, x1 = mirror(x1), mirror(x0)
            cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
            good, worst, says = framing.verdict(m)
            tint = (90, 230, 90) if good else (60, 200, 255)
            cv2.rectangle(view, (x0, y0), (x1, y1), tint, 2)
            cv2.drawMarker(view, (cx, cy), tint, cv2.MARKER_TILTED_CROSS, 60, 3)
            cv2.arrowedLine(view, (cx, cy), (w // 2, h // 2), tint, 2,
                            tipLength=0.15)
            lines = [says,
                     "arrow = where the board has to go."
                     "  Camera moves the opposite way.",
                     f"out by {abs(m['shift_x']):.0f} mm across and "
                     f"{abs(m['shift_y']):.0f} mm the other way",
                     f"room to slide:  left {m['right']:.0f}  right {m['left']:.0f}"
                     f"  up {m['up']:.0f}  down {m['down']:.0f}  mm"]
            if best is None or worst > best:
                best = worst
        else:
            tint = (60, 60, 255)
            seen = sorted(tags or {})
            lines = ["cannot place the board", note or "",
                     f"tags found: {seen if seen else 'none'}."
                     "  Three are needed.",
                     "Check nothing is lying across the missing ones."]
        for i, text in enumerate(lines):
            at = (14, 34 + 30 * i)
            cv2.putText(view, text, at, cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(view, text, at, cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        tint, 1, cv2.LINE_AA)
        scale = min(1.0, args.display_width / w)
        cv2.imshow(win, cv2.resize(view, None, fx=scale, fy=scale)
                   if scale < 1.0 else view)
        key = cv2.waitKey(20) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            shot = files.capture("centring.png")
            cv2.imwrite(shot, frame)
            print(f"  wrote {files.shown(shot)}")

    cap.release()
    cv2.destroyWindow(win)
    if best is None:
        print("  Never saw the board, so nothing was measured.")
        return False
    print(f"  Best it got: {best:.0f} mm of room in the worst direction.")
    if best < 8:
        print("  Under 8 mm is tight. Worth another go before tightening down.")
    return True


# ---------------------------------------------------------------------- tools


def step_tools(args):
    """uvc-util, without which none of the rest is worth measuring.

    First, because it is the only step that needs the internet rather than
    the rig, and because everything after it is read off a frame whose
    exposure it sets.
    """
    if camera.uvc_path():
        return True

    print("\n--- tools ---")
    print("uvc-util is missing. Without it the exposure stays on automatic:")
    print("the camera meters a mostly-black board, opens right up, and the")
    print("room coming through the empty holes reads as bright as the balls.")
    print("Every step after this one is measured off that frame.")
    print(f"\nIt is a small Mac-only utility from {camera.UVC_SOURCE}.")
    print("Building it needs git and clang, and takes a few seconds.")

    if input("\nClone and compile it now? [Y/n] ").strip().lower() in ("n", "no"):
        print("  skipped. Build it yourself and run setup.py again.")
        return False
    return camera.build_uvc_util() is not None


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
    seen = camera.uvc_devices() or {}
    camera.remember(exposure=value, index=args.index, uvc_index=args.uvc_index,
                    device=seen.get(args.uvc_index))
    print(f"   Saved to {camera.CAMERA_FILE}: {seen.get(args.uvc_index, 'camera')}"
          f" at exposure {value}.")
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
    cv2.imwrite(files.capture("frame_last.png"), frame)
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

    # Said here because it is the one problem that hides behind a good result.
    # The board reads perfectly while sitting against the edge of the frame;
    # you only find out it had no room when somebody nudges it mid-song.
    m = framing.margins(frame.shape, cells, tags, geometry.COLS, geometry.ROWS)
    if m:
        good, worst, says = framing.verdict(m)
        print(f"   Room to slide: left {m['left']:.0f}  right {m['right']:.0f}"
              f"  up {m['up']:.0f}  down {m['down']:.0f} mm.")
        if not good:
            print(f"   {says.capitalize()}. The camera is aimed"
                  f" {m['off_x']:+.0f},{m['off_y']:+.0f} px off the middle of")
            print("   the frame. Worth fixing once, with:  python setup.py --centre")
    return True


def step_colours(args):
    """Teach one colour at a time, guided, in a window."""
    print("\n3. COLOURS")
    queue = [c.strip() for c in args.colour_names.split(",")]
    print("   " + ", ".join(queue) + ", one at a time. The window says what to do.")

    cap = camera.open_camera(args)
    if cap is None:
        return False
    pinned = camera.settings().get("exposure")
    if pinned is not None:
        camera.set_exposure(args.uvc_index, pinned)

    win = "bubblegum, teaching"
    cv2.namedWindow(win)
    while queue:
        ok, frame = cap.read()
        frame = camera.orient(frame)
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
    ap.add_argument("--index", type=int, default=None,
                    help="OpenCV camera index; camera.json otherwise")
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--uvc-index", type=int, default=None,
                    help="uvc-util camera index; camera.json otherwise")
    ap.add_argument("--display-width", type=int, default=1280)
    ap.add_argument("--no-mirror", action="store_true")
    ap.add_argument("--sens", type=float, default=1.25,
                    help="how much brighter than its surroundings a hole has "
                         "to be. Lower finds dimmer holes.")
    ap.add_argument("--clip-target", type=float, default=0.001)
    ap.add_argument("--reclick", action="store_true",
                    help="redo the corner clicks rather than reusing them")
    ap.add_argument("--colour-names", type=str, default=",".join(DEFAULT_COLOURS))
    ap.add_argument("--centre", action="store_true",
                    help="aim the camera; a build step, not part of the walk")
    ap.add_argument("--tools", action="store_true")
    ap.add_argument("--exposure", action="store_true")
    ap.add_argument("--geometry", action="store_true")
    ap.add_argument("--colours", action="store_true")
    args = ap.parse_args()
    camera.resolve(args)

    if args.centre:
        return 0 if step_centre(args) else 1

    chosen = [n for n, on in (("tools", args.tools),
                              ("exposure", args.exposure),
                              ("geometry", args.geometry),
                              ("colours", args.colours)) if on]
    steps = chosen or ["tools", "exposure", "geometry", "colours"]
    run = {"tools": step_tools, "exposure": step_exposure,
           "geometry": step_geometry, "colours": step_colours}

    for name in steps:
        if not run[name](args):
            print(f"\nStopped at {name}. Nothing after it has run.")
            return 1
    print("\nReady. Start it with:  python play.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
