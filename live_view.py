#!/usr/bin/env python3
"""Watch the geometry track the board, live.

Shows the camera with the tags outlined, every cell's sample ellipse drawn
where the geometry thinks it is, and the colour each cell is currently
reading. Move the board around and watch the grid follow it.

This is also the skeleton of the continuous loop the instrument will need, so
it deliberately does the same work per frame that the real thing will: grab,
find tags, place cells, sample.

Usage:
    python live_view.py                     # live camera
    python live_view.py --image frame.png   # one saved frame, same drawing
    python live_view.py --image f.png --shot out.png   # render and exit

Keys:
    q       quit
    space   freeze and unfreeze
    s       save the current frame and the annotated view
    c       cycle what the ellipses are filled with

Pass --learn to have it pin cells as it goes: any hole it can see clearly
updates that cell's stored position, so the ones that are invisible in one
lighting get fixed from a frame where they are not.
"""

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

import cell_probe as cp
import pattern as pat

FILLS = ("nothing", "colour", "brightness")
DEFAULT_COLOURS = ("pink", "yellow", "blue", "green")


def draw(frame, cells, tags, note, fill, drift, mirror=True,
         banner=None, highlight=(), ignored=()):
    """Mirrored by default, because the camera is under the sheet.

    Looking up from below reverses left and right against the board you are
    standing over, which makes watching it track the board needlessly hard.
    Only the picture is flipped; nothing upstream knows or cares.
    """
    out = cv2.flip(frame, 1) if mirror else frame.copy()
    wide = out.shape[1]
    flip = (lambda x: wide - 1 - int(x)) if mirror else int

    for tag_id, corner in sorted(tags.items()):
        shown = np.int32([[flip(x), y] for x, y in corner])
        cv2.polylines(out, [shown], True, (255, 0, 255), 2)
        cv2.putText(out, str(tag_id), tuple(shown[0] + [6, -8]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2, cv2.LINE_AA)

    for c in cells or []:
        p = (flip(c["x"]), int(c["y"]))
        if fill == "colour" and "bgr" in c:
            cv2.ellipse(out, p, (c["rx"], c["ry"]), 0, 0, 360, c["bgr"], -1)
        elif fill == "brightness" and "L" in c:
            v = int(np.clip(c["L"], 0, 100) * 2.55)
            cv2.ellipse(out, p, (c["rx"], c["ry"]), 0, 0, 360, (v, v, v), -1)
        if c["idx"] in highlight:
            edge, thick = (0, 255, 255), 3
        elif c["idx"] in ignored:
            edge, thick = (0, 0, 255), 2
        else:
            edge, thick = (0, 255, 0), 1
        cv2.ellipse(out, p, (c["rx"], c["ry"]), 0, 0, 360, edge, thick)
        shown = c.get("settled", c.get("colour"))
        if shown not in (None, "empty") or c.get("wanted"):
            # Under the hole rather than over it: a label on top of the ball
            # hides the thing you are trying to check it against.
            label = (f"{shown} -> {c['wanted']}" if c.get("wanted")
                     else shown) or c["wanted"]
            size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0]
            at = (p[0] - size[0] // 2, p[1] + c["ry"] + 20)
            cv2.putText(out, label, at, cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(out, label, at, cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (140, 200, 255) if c.get("wanted") else (255, 255, 255),
                        1, cv2.LINE_AA)

    if banner:
        board = out.shape
        cv2.rectangle(out, (0, board[0] - 120), (board[1], board[0]), (0, 0, 0), -1)
        for i, line in enumerate(banner):
            cv2.putText(out, line, (24, board[0] - 78 + i * 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0 if i == 0 else 0.7,
                        (0, 255, 255) if i == 0 else (230, 230, 230),
                        2 if i == 0 else 1, cv2.LINE_AA)

    lines = [note]
    if drift is not None:
        lines.append(f"board has moved {drift:.0f} px since calibration")
    lines.append(f"fill: {fill}   q quit   space freeze   s save   c cycle fill")
    for i, line in enumerate(lines):
        y = 34 + i * 30
        cv2.putText(out, line, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                    (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(out, line, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                    (255, 255, 255), 1, cv2.LINE_AA)
    return out


def load_counts():
    _, rows = cp.load_reference()
    return None if rows is None else int((rows[:, 4] > 1).sum())


def read_board(frame, learn=False):
    """One frame's worth of the real pipeline: tags, cells, and their colours."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    tags = cp.detect_tags(gray)
    cells, how = cp.cells_from_tags(tags)

    if cells is None:
        ref, _ = cp.load_reference()
        if ref is None:
            note = f"no {cp.REFERENCE_FILE} yet: run cell_probe.py --reclick once"
        else:
            note = f"tags seen {sorted(tags)}, need three of {sorted(ref)}"
        return None, tags, note, None

    seen, fit, H = how
    note = f"placed from tags {seen}, agreeing to {fit:.1f} px"

    # Same learning as cell_probe: every frame that clearly shows a hole pins
    # that cell a little more firmly, so watching it run is also what fills the
    # map in.
    if learn:
        grew = cp.accumulate_reference(gray, cells, H, 1.20)
        if grew:
            cells, _ = cp.cells_from_tags(tags)
            note += f"   pinned {grew[1]}/64"
    else:
        ref_rows = load_counts()
        if ref_rows is not None:
            note += f"   pinned {ref_rows}/64"

    ref, _ = cp.load_reference()
    drift = max(np.linalg.norm(tags[i].mean(0) - ref[i].mean(0))
                for i in set(tags) & set(ref))

    # Always sample. Classifying needs it, the teach counts need it, and tying
    # it to what the ellipses are filled with meant a plain run never labelled
    # anything.
    if True:
        for c in cells:
            s = cp.sample_cell(frame, gray, c["x"], c["y"], c["rx"], c["ry"])
            if s is None:
                continue
            c.update(s)
            c["bgr"] = tuple(int(v) for v in cv2.cvtColor(
                np.uint8([[[s["L"] * 255 / 100, s["a"] + 128, s["b"] + 128]]]),
                cv2.COLOR_LAB2BGR)[0][0])
        protos = cp.load_prototypes()
        if protos:
            cp.classify(cells, protos)
    return cells, tags, note, drift


def count_balls(cells):
    """Exactly what teaching would use, and what it would throw away.

    Showing the brightness split alone was misleading: it ringed empty holes
    that happened to be bright, which teaching then discarded anyway. The
    display should say what will actually be learned.
    """
    if not cells or "L" not in cells[0]:
        return [], []
    kept, strays, _ = cp.split_board(cells)
    return kept, strays


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--image", type=str, default=None)
    ap.add_argument("--shot", type=str, default=None,
                    help="render one annotated frame to this file and exit")
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--uvc-index", type=int, default=0)
    ap.add_argument("--display-width", type=int, default=1280)
    ap.add_argument("--no-mirror", action="store_true",
                    help="show the camera's own view instead of flipping it to "
                         "match the board as you see it from above")
    ap.add_argument("--teach", nargs="?", const=",".join(DEFAULT_COLOURS),
                    default=None, metavar="COLOURS",
                    help="walk through teaching each colour in turn; defaults "
                         f"to {', '.join(DEFAULT_COLOURS)}")
    ap.add_argument("--learn", action="store_true",
                    help="pin each cell's position as holes become visible, "
                         "which fills in the ones no single frame can show")
    args = ap.parse_args()

    fill = FILLS[0]

    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            print(f"Could not read {args.image}")
            return 1
        cells, tags, note, drift = read_board(frame, args.learn)
        view = draw(frame, cells, tags, note, fill, drift, not args.no_mirror)
        print(note)
        if args.shot:
            cv2.imwrite(args.shot, view)
            print(f"wrote {args.shot}")
            return 0
        cv2.imshow("bubblegum, live geometry", cv2.resize(
            view, None, fx=args.display_width / view.shape[1],
            fy=args.display_width / view.shape[1]))
        cv2.waitKey(0)
        return 0

    cap = cp.open_camera(args)
    if cap is None:
        return 1
    if os.path.exists(cp.EXPOSURE_FILE):
        with open(cp.EXPOSURE_FILE) as fh:
            value = json.load(fh)["exposure"]
        cp.set_exposure(args.uvc_index, value)
        print(f"exposure pinned at {value}")

    win = "bubblegum, live geometry"
    cv2.namedWindow(win)
    frozen, held, last, fps, frames = False, None, time.perf_counter(), 0.0, 0

    board = pat.Stabiliser()
    queue = [c.strip() for c in args.teach.split(",")] if args.teach else []
    if queue:
        fill = "nothing"
        print("teaching " + ", ".join(queue))

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
        if not queue:
            settled = board.update(pat.readings_from(cells, cells is not None))
            for i, was, now_is in settled:
                print(f"  cell {i}: {was} -> {now_is}")
            waiting = board.pending()
            for c in cells or []:
                c["settled"] = board.pattern[c["idx"]]
                c["wanted"] = waiting.get(c["idx"])
            state = ("HOLDING, cannot see the board" if board.blind else
                     "HOLDING, too much changed at once" if board.blocked else
                     f"{len(board.filled())} balls")
            note = f"{state}   {note}"

        banner, highlight, ignored = None, (), ()
        if queue:
            kept, strays = count_balls(cells)
            highlight = {c["idx"] for c in kept}
            ignored = {c["idx"] for c in strays}
            ready = "press SPACE to learn it" if kept else "waiting for balls"
            aside = (f"   ignoring {len(strays)} bright but wrong-coloured"
                     if strays else "")
            banner = [f"Put only the {queue[0].upper()} balls on the board",
                      f"I can see {len(kept)}.{aside}  {ready}.",
                      f"{len(queue)} to go   s skip this colour   q give up"]
        elif args.teach:
            banner = ["Done. Every colour taught.",
                      "The board is being read live; each ball is labelled.",
                      "q quit   c cycle fill"]

        view = draw(frame, cells, tags,
                    f"{note}   {fps:4.1f} fps" + ("   FROZEN" if frozen else ""),
                    fill, drift, not args.no_mirror, banner, highlight, ignored)
        scale = args.display_width / view.shape[1]
        cv2.imshow(win, cv2.resize(view, None, fx=scale, fy=scale))

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if queue and key == ord("s"):
            print(f"skipped {queue.pop(0)}")
            continue
        if queue and key == ord(" "):
            kept, _ = count_balls(cells)
            if not kept:
                print("  nothing on the board yet")
                continue
            if cp.teach(cells, queue[0]) is not None:
                print(f"  learnt {queue.pop(0)}")
            continue
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
