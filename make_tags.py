#!/usr/bin/env python3
"""Print sheet for the four AprilTags that replace the corner clicks.

Four DICT_APRILTAG_36h11 tags, ids 0 to 3, on A4. Each is 25 mm of black
square with a 4 mm white quiet zone, which the detector needs in order to find
the edges at all.

25 mm because that is what fits. There is only about 54 mm of clear sheet in
view above the grid and 31 mm below it, the rest of the frame being crate. A
36h11 tag is 8 modules across, so 25 mm gives roughly 3 mm per module, which
at the 2.6 to 5 px/mm this camera resolves is 8 to 16 px per module. Detection
wants 3 and is happy at 5.

Usage:
    python make_tags.py            # writes tags.pdf and tags.png

Print at 100%, no scaling, no "fit to page". Then check the ruler on the sheet
with a real one before cutting anything out: if the 100 mm line is not 100 mm,
the print scaled and the tags will be the wrong size.
"""

import sys

import cv2
import numpy as np
from PIL import Image

DPI = 600
MM = DPI / 25.4

TAG_MM = 25.0          # black square
QUIET_MM = 4.0         # white margin the detector needs around it
A4_MM = (210.0, 297.0)
IDS = [0, 1, 2, 3]


def mm(v):
    return int(round(v * MM))


def tag_image(tag_id, side_px):
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    return cv2.aruco.generateImageMarker(d, tag_id, side_px)


def text(canvas, s, x, y, scale=1.6, thick=3):
    cv2.putText(canvas, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, 0, thick,
                cv2.LINE_AA)


def main():
    W, H = mm(A4_MM[0]), mm(A4_MM[1])
    sheet = np.full((H, W), 255, np.uint8)

    block = TAG_MM + 2 * QUIET_MM
    # Two columns, two rows, spread across the page.
    xs = [A4_MM[0] / 2 - block - 15, A4_MM[0] / 2 + 15]
    ys = [60.0, 60.0 + block + 30]

    for i, tag_id in enumerate(IDS):
        ox, oy = xs[i % 2], ys[i // 2]
        side = mm(TAG_MM)
        img = tag_image(tag_id, side)
        x0, y0 = mm(ox + QUIET_MM), mm(oy + QUIET_MM)
        sheet[y0:y0 + side, x0:x0 + side] = img
        # Corner ticks on the quiet zone, so a cut line is obvious without
        # drawing a box that the detector might mistake for tag structure.
        for cx, cy in ((mm(ox), mm(oy)), (mm(ox + block), mm(oy)),
                       (mm(ox), mm(oy + block)), (mm(ox + block), mm(oy + block))):
            cv2.line(sheet, (cx - mm(3), cy), (cx + mm(3), cy), 128, 2)
            cv2.line(sheet, (cx, cy - mm(3)), (cx, cy + mm(3)), 128, 2)
        text(sheet, f"id {tag_id}", mm(ox), mm(oy + block + 8))

    text(sheet, "Bubblegum sequencer, board tags", mm(20), mm(30), 2.2, 4)
    text(sheet, "AprilTag 36h11, 25 mm. Print at 100%, no scaling.",
         mm(20), mm(42), 1.4, 2)

    # Scale check. If this is not 100 mm on paper, the print scaled.
    ry = mm(272)
    cv2.line(sheet, (mm(20), ry), (mm(120), ry), 0, 4)
    for t in range(0, 101, 10):
        h = mm(4) if t % 50 else mm(7)
        cv2.line(sheet, (mm(20 + t), ry), (mm(20 + t), ry - h), 0, 3)
    text(sheet, "measure me: this line is 100 mm", mm(20), mm(282), 1.4, 2)

    out = Image.fromarray(sheet)
    out.save("tags.pdf", resolution=DPI)
    out.save("tags.png", dpi=(DPI, DPI))
    print(f"tags.pdf and tags.png written: ids {IDS}, "
          f"{TAG_MM:.0f} mm tags on A4 at {DPI} dpi.")
    print("Print at 100%, then check the 100 mm line with a ruler.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
