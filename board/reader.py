"""One frame in, sixty-four classified cells out.

This is the whole vision side behind a single call, and it exists so that the
instrument and the debugging window sit on the same thing. They used to not:
the server imported the viewer to get at this, which meant running the
instrument pulled in the debug window, and every change to one risked the
other.
"""

import cv2
import numpy as np

from . import colour, geometry


def read_board(frame, learn=False, sens=1.20):
    """Returns (cells, tags, note, drift), where cells is None if it failed.

    `drift` is how far the board has moved since the geometry was recorded,
    which is worth showing: it is the number that quietly grows until
    something stops working.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    tags = geometry.detect_tags(gray)
    cells, how = geometry.cells_from_tags(tags)

    if cells is None:
        reference, _ = geometry.load_reference()
        if reference is None:
            note = (f"no {geometry.REFERENCE_FILE} yet: run setup.py to record "
                    "where the grid is")
        else:
            note = f"tags seen {sorted(tags)}, need three of {sorted(reference)}"
        return None, tags, note, None

    seen, fit, H = how
    note = f"placed from tags {seen}, agreeing to {fit:.1f} px"

    # Every frame that clearly shows a hole pins that cell a little more
    # firmly, so simply using the thing is what fills the map in. No single
    # frame shows all 64: a hole vanishes when the sheet beside it happens to
    # reflect a strip back at the same brightness as the room behind it.
    if learn:
        grew = geometry.accumulate_reference(gray, cells, H, sens)
        if grew:
            cells, _ = geometry.cells_from_tags(tags)
            note += f"   pinned {grew[1]}/{geometry.COLS * geometry.ROWS}"
    else:
        _, rows = geometry.load_reference()
        if rows is not None:
            note += (f"   pinned {int((rows[:, 4] > 1).sum())}"
                     f"/{geometry.COLS * geometry.ROWS}")

    reference, _ = geometry.load_reference()
    drift = max(np.linalg.norm(tags[i].mean(0) - reference[i].mean(0))
                for i in set(tags) & set(reference))

    for cell in cells:
        sample = colour.sample_cell(frame, gray, cell["x"], cell["y"],
                                    cell["rx"], cell["ry"])
        if sample is None:
            continue
        cell.update(sample)
        cell["bgr"] = tuple(int(v) for v in cv2.cvtColor(
            np.uint8([[[sample["L"] * 255 / 100,
                        sample["a"] + 128, sample["b"] + 128]]]),
            cv2.COLOR_LAB2BGR)[0][0])

    prototypes = colour.load_prototypes()
    if prototypes:
        colour.classify(cells, prototypes)
    return cells, tags, note, drift
