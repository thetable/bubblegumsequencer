"""Drawing the board on a frame.

Shared by the setup walk and the inspection window, because both want the
same picture with a different message over it. Presentation only: nothing
here decides anything.
"""

import cv2
import numpy as np

FILLS = ("nothing", "colour", "brightness")

# Roughly what each ball looks like, for tinting the line that asks for it.
# Teaching means reading one instruction and then hunting through a bag, and
# the colour is quicker to recognise than the word. Display only: nothing
# decides anything from these, and a name that is not here just stays yellow.
TINTS = {
    "pink": (150, 120, 255),
    "yellow": (60, 220, 245),
    "blue": (235, 190, 130),
    "green": (140, 220, 150),
    "orange": (70, 160, 250),
}


def draw(frame, cells, tags, note, fill, drift, mirror=True,
         banner=None, highlight=(), ignored=(), tint=None):
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
            at = (24, board[0] - 78 + i * 38)
            size = 1.0 if i == 0 else 0.7
            ink = (tint or (0, 255, 255)) if i == 0 else (230, 230, 230)
            # A pale ball colour needs the dark outline to stay readable over
            # a lit sheet; the grey lines never did and still do not.
            if i == 0:
                cv2.putText(out, line, at, cv2.FONT_HERSHEY_SIMPLEX, size,
                            (0, 0, 0), 5, cv2.LINE_AA)
            cv2.putText(out, line, at, cv2.FONT_HERSHEY_SIMPLEX, size, ink,
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
