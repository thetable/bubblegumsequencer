#!/usr/bin/env python3
"""Print sheet for the two tag strips that place the grid.

Ten DICT_APRILTAG_36h11 tags, five to a strip, one strip above the grid and
one below, on the underside of the sheet facing the camera.

Why strips rather than four corner tags. The warp is cubic across the columns,
four coefficients, so it needs at least four *distinct* column positions to be
determined. Four tags in the corners cluster at two, one near column 0 and one
near column 15, and two clusters cannot constrain a cubic. That is not a
tuning problem, it is arithmetic, and it is why the corner clicks needed the
bend searched for separately. Five tags spread along each edge give five
distinct positions and over-determine it.

The sizes are set by how little clear sheet there is. About 50 mm beyond the
grid in the worst direction, and the camera only sees 114 mm fore and aft at
230 mm, so a 22 mm tag with a 3.5 mm quiet zone sits inside both with about
20 mm to spare. At this camera that is near 100 px of tag, where detection
wants 30.

The strips do not have to be placed accurately. Where each tag really ended up
is measured once, during calibration, and used from then on. They only have to
be straight, clear of the holes, and not moved afterwards.

Usage:
    python make_tags.py            # writes captures/tags.pdf and .png

Print at 100%, no scaling, no "fit to page". Then check the ruler against a
real one before cutting: if the 100 mm line is not 100 mm, the print scaled
and every tag is the wrong size.
"""

import sys

import cv2
import numpy as np
from PIL import Image

from board import files

DPI = 600
MM = DPI / 25.4

TAG_MM = 22.0          # black square
QUIET_MM = 3.5         # white margin the detector needs around it
PITCH_MM = 62.0        # between tag centres along a strip
PER_STRIP = 5
STRIP_MM = 280.0       # printed length, inside A4 landscape with margins
A4_MM = (297.0, 210.0)                 # landscape: a strip has to be long

# Ids well clear of 0 to 3, so the old corner tags can stay on the sheet or
# come off without either choice breaking anything.
STRIPS = [(range(10, 15), "BACK EDGE", "far side, away from you"),
          (range(20, 25), "FRONT EDGE", "nearest you, where you stand")]

CLEAR_OF_HOLES_MM = 6.0


def mm(v):
    return int(round(v * MM))


def tag_image(tag_id, side_px):
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    return cv2.aruco.generateImageMarker(d, tag_id, side_px)


def text(canvas, s, x, y, scale=1.6, thick=3, shade=0):
    cv2.putText(canvas, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, shade,
                thick, cv2.LINE_AA)


def strip(sheet, top_mm, ids, title, where, point_up):
    """One strip: five tags, a label, and an arrow to the sheet edge."""
    height = TAG_MM + 2 * QUIET_MM
    left = (A4_MM[0] - STRIP_MM) / 2
    side = mm(TAG_MM)

    # Cut box. Grey so the detector cannot mistake it for tag structure, and
    # far enough out that it sits beyond the quiet zone.
    cv2.rectangle(sheet, (mm(left), mm(top_mm)),
                  (mm(left + STRIP_MM), mm(top_mm + height)), 190, 2)

    first = STRIP_MM / 2 - PITCH_MM * (PER_STRIP - 1) / 2
    for k, tag_id in enumerate(ids):
        cx = left + first + k * PITCH_MM
        x0 = mm(cx - TAG_MM / 2)
        y0 = mm(top_mm + QUIET_MM)
        sheet[y0:y0 + side, x0:x0 + side] = tag_image(tag_id, side)
        text(sheet, str(tag_id), mm(cx - 4), mm(top_mm + height + 7), 1.1, 2, 120)

    # Centre mark, so the strip can be lined up with the middle of the grid.
    midx = mm(left + STRIP_MM / 2)
    for dy in (0, height):
        cv2.line(sheet, (midx, mm(top_mm + dy) - mm(4)),
                 (midx, mm(top_mm + dy) + mm(4)), 120, 3)

    # Which way round. Read from the underside, which is the side you are
    # looking at while you stick it on and while you put the board down.
    ty = mm(top_mm - 6)
    text(sheet, title, mm(left), ty, 2.0, 4)
    text(sheet, where, mm(left + 95), ty, 1.3, 2, 90)
    ax = mm(left + STRIP_MM - 30)
    tip = mm(top_mm - 16) if point_up else mm(top_mm + height + 16)
    base = mm(top_mm - 2) if point_up else mm(top_mm + height + 2)
    cv2.arrowedLine(sheet, (ax, base), (ax, tip), 0, 5, tipLength=0.45)
    text(sheet, "sheet edge", mm(left + STRIP_MM - 115),
         tip + (mm(2) if point_up else mm(6)), 1.2, 2, 90)


def main():
    W, H = mm(A4_MM[0]), mm(A4_MM[1])
    sheet = np.full((H, W), 255, np.uint8)
    height = TAG_MM + 2 * QUIET_MM

    text(sheet, "Bubblegum sequencer, board tag strips", mm(14), mm(16), 2.0, 4)
    text(sheet, f"AprilTag 36h11, {TAG_MM:.0f} mm. Print at 100%, no scaling. "
         "Underside of the sheet, facing the camera.", mm(14), mm(26), 1.2, 2, 90)

    strip(sheet, 52.0, *STRIPS[0], point_up=True)
    strip(sheet, 118.0, *STRIPS[1], point_up=False)

    text(sheet, f"Each strip about {CLEAR_OF_HOLES_MM:.0f} mm clear of the "
         "outer row of holes, centre mark on the middle column.",
         mm(14), mm(170), 1.2, 2, 90)
    text(sheet, "Straight matters, exact does not: where they really landed is "
         "measured once, during calibration.", mm(14), mm(178), 1.2, 2, 90)

    ry = mm(196)
    cv2.line(sheet, (mm(14), ry), (mm(114), ry), 0, 4)
    for t in range(0, 101, 10):
        h = mm(4) if t % 50 else mm(7)
        cv2.line(sheet, (mm(14 + t), ry), (mm(14 + t), ry - h), 0, 3)
    text(sheet, "measure me: this line is 100 mm", mm(120), ry, 1.3, 2)

    out = Image.fromarray(sheet)
    pdf = files.capture("tags.pdf")
    out.save(pdf, resolution=DPI)
    out.save(files.capture("tags.png"), dpi=(DPI, DPI))
    ids = [i for r, _, _ in STRIPS for i in r]
    print(f"{files.shown(pdf)} and tags.png written: {len(ids)} tags, "
          f"{TAG_MM:.0f} mm, ids {ids[0]}-{ids[PER_STRIP-1]} and "
          f"{ids[PER_STRIP]}-{ids[-1]}, on A4 landscape at {DPI} dpi.")
    print(f"Strip is {STRIP_MM:.0f} x {height:.0f} mm. Print at 100%, then "
          "check the 100 mm line with a ruler before cutting.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
