"""Where the board sits in the frame, and how much room is left around it.

A separate question from where the cells are. `geometry` answers "which pixel
is cell 37"; this answers "how far can the whole board slide before something
falls off the edge", which is what decides where the camera is bolted down and
how deep the box has to be.

The distinction matters because the obvious way to mount the camera is to put
it under the middle of the board, and on this module that is wrong. Its lens
axis images at pixel 841 of 1920 rather than at 960, so a camera centred under
the board puts the board 119 px off-centre in the frame: 8 px of clearance at
one end and 234 at the other. Aim it at the middle of the sensor instead.
"""

import numpy as np

COL_MM = 24.0           # the grid's own column pitch, the only ruler in shot


def content(cells, tags):
    """The box around everything that has to stay in frame, in pixels.

    The cells with their hole rims, and the tags. Losing a cell costs one step
    of the sequence; losing a tag costs the whole placement, so both count.
    """
    if not cells:
        return None
    x0 = min(c["x"] - c["rx"] for c in cells)
    x1 = max(c["x"] + c["rx"] for c in cells)
    y0 = min(c["y"] - c["ry"] for c in cells)
    y1 = max(c["y"] + c["ry"] for c in cells)
    for corner in (tags or {}).values():
        x0, x1 = min(x0, corner[:, 0].min()), max(x1, corner[:, 0].max())
        y0, y1 = min(y0, corner[:, 1].min()), max(y1, corner[:, 1].max())
    return float(x0), float(y0), float(x1), float(y1)


def scales(cells, cols, rows):
    """Pixels per mm, sideways at the board's edge and along the rows.

    Two numbers rather than one because barrel distortion squeezes the ends of
    the long axis far harder than the short one, so the same margin in pixels
    is worth quite different amounts of board travel in each direction.
    """
    pitch = np.array([[abs(cells[r * cols + k + 1]["x"] - cells[r * cols + k]["x"])
                       for k in range(cols - 1)] for r in range(rows)])
    across = np.median(pitch, axis=0)
    rowsep = np.median([abs(cells[(r + 1) * cols + k]["y"] - cells[r * cols + k]["y"])
                        for r in range(rows - 1) for k in range(cols)])
    return float(across.min() / COL_MM), float(rowsep / 35.0), float(across.max() / COL_MM)


def margins(frame_shape, cells, tags, cols, rows):
    """How far the board may slide in each direction, in millimetres.

    The useful unit. Pixels of clearance mean nothing until you know what a
    pixel is worth where the clearance is, and at the ends of the long axis a
    pixel is worth twice what it is in the middle.
    """
    box = content(cells, tags)
    if box is None:
        return None
    h, w = frame_shape[:2]
    x0, y0, x1, y1 = box
    edge, vert, middle = scales(cells, cols, rows)
    return {
        "box": box,
        "left": x0 / edge, "right": (w - x1) / edge,
        "up": y0 / vert, "down": (h - y1) / vert,
        # How far the camera itself must move to centre the picture. Measured
        # at the optical axis, where the scale is greatest, because that is
        # roughly where the camera is.
        "shift_x": ((w - x1) - x0) / 2 / middle,
        "shift_y": ((h - y1) - y0) / 2 / vert,
        "off_x": ((w - x1) - x0) / 2, "off_y": ((h - y1) - y0) / 2,
    }


def verdict(m, want=8.0):
    """Whether this is good enough to bolt down, and what to say if not."""
    worst = min(m["left"], m["right"], m["up"], m["down"])
    if abs(m["off_x"]) > 40 or abs(m["off_y"]) > 40:
        return False, worst, "not centred yet"
    if worst < want:
        return False, worst, f"centred, but only {worst:.0f} mm of room"
    return True, worst, f"centred, {worst:.0f} mm of room all round"
