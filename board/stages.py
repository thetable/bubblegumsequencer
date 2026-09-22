"""The vision pipeline, one picture per step, for showing people.

Each stage exists because of something visibly wrong in the one before it, so
walking through them in order is the explanation. The captions say what the
step does and, where it is interesting, why the obvious cheaper thing fails.

Computed on request rather than in the playing loop: it is four times the work
of just reading the board, and nobody is watching it while the music is on.
"""

import base64

import cv2
import numpy as np

from . import colour, geometry

WIDE = 900          # what the browser gets; the source is 1920
QUALITY = 72


def _jpeg(image):
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    scale = WIDE / image.shape[1]
    small = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, QUALITY])
    return base64.b64encode(buf).decode() if ok else ""


def _mirror(image):
    """Every stage is shown the way round the player sees the board."""
    return cv2.flip(image, 1)


def render(frame):
    """Every stage, as {title, caption, jpeg} in the order they happen."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    out = []

    def stage(title, caption, image):
        out.append({"title": title, "caption": caption,
                    "jpeg": _jpeg(_mirror(image))})

    stage("What the camera sees",
          "A 130 degree lens under a scratched black sheet, 200 mm away. The "
          "rows bow, the corners fall dark, and the light strips are in shot.",
          frame)

    # 1. Flat field ------------------------------------------------------
    f = gray.astype(np.float32)
    background = cv2.GaussianBlur(f, (0, 0), geometry.BACKGROUND_SIGMA)
    ratio = f / (background + 1e-6)
    stage("Flatten the lighting",
          "Divide the picture by a blurred copy of itself. Vignetting and "
          "uneven strips cancel out, and what is left is how much brighter "
          "each thing is than its own surroundings.",
          np.clip(ratio * 110, 0, 255).astype(np.uint8))

    # 2. Holes -----------------------------------------------------------
    tags = geometry.detect_tags(gray)
    cells, how = geometry.cells_from_tags(tags)
    pitch = geometry.row_pitch(cells) if cells else 100.0
    blobs = []
    if cells:
        points = np.float32([[c["x"], c["y"]] for c in cells])
        blobs = geometry.find_holes(
            gray, cv2.convexHull(points).reshape(-1, 2), pitch, 1.20)
    found = frame.copy()
    for b in blobs:
        cv2.circle(found, (int(b["x"]), int(b["y"])), int(0.3 * pitch),
                   (90, 255, 90), 2)
    stage("Find the holes",
          f"Round blobs of about the right size: {len(blobs)} of 64 here. "
          "Never all of them. A hole disappears entirely when the sheet "
          "beside it reflects a strip back at the same brightness as the "
          "room shining through it.",
          found)

    # 3. Tags ------------------------------------------------------------
    marked = frame.copy()
    for tag_id, corner in sorted(tags.items()):
        cv2.polylines(marked, [np.int32(corner)], True, (255, 80, 255), 3)
        cv2.putText(marked, str(tag_id), tuple(np.int32(corner[0]) + [8, -10]),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 80, 255), 2, cv2.LINE_AA)
    stage("Find the tags",
          f"Four AprilTags on the underside of the sheet; {len(tags)} visible. "
          "Three is enough to know where the board is, so it can be shoved "
          "around mid-song and the grid follows.",
          marked)

    if not cells:
        return out

    # 4. The grid --------------------------------------------------------
    placed = frame.copy()
    for c in cells:
        cv2.ellipse(placed, (int(c["x"]), int(c["y"])), (c["rx"], c["ry"]),
                    0, 0, 360, (90, 255, 90), 2)
    edge = abs(cells[1]["x"] - cells[0]["x"])
    middle = abs(cells[8]["x"] - cells[7]["x"])
    stage("Place all 64",
          f"A polynomial warp fitted to the holes, cubic across the columns. "
          f"The pitch runs {edge:.0f} px at the edge to {middle:.0f} px in the "
          "middle, and a four-corner homography cannot bend like that. It was "
          "wrong by up to 94 px.",
          placed)

    # 5. Sampling --------------------------------------------------------
    for c in cells:
        sample = colour.sample_cell(frame, gray, c["x"], c["y"], c["rx"], c["ry"])
        if sample:
            c.update(sample)
            c["bgr"] = tuple(int(v) for v in cv2.cvtColor(
                np.uint8([[[sample["L"] * 255 / 100,
                            sample["a"] + 128, sample["b"] + 128]]]),
                cv2.COLOR_LAB2BGR)[0][0])
    sampled = (frame * 0.25).astype(np.uint8)
    for c in cells:
        if "bgr" in c:
            cv2.ellipse(sampled, (int(c["x"]), int(c["y"])), (c["rx"], c["ry"]),
                        0, 0, 360, c["bgr"], -1)
    stage("Read each one",
          "An ellipse, not a circle: seen from 49 degrees off-axis a round "
          "hole is an ellipse, and a circle would throw away most of it. "
          "Median colour, brightest 5% dropped, because one glint moves a "
          "mean anywhere.",
          sampled)

    # 6. Classification --------------------------------------------------
    prototypes = colour.load_prototypes()
    named = (frame * 0.25).astype(np.uint8)
    if prototypes:
        colour.classify(cells, prototypes)
        for c in cells:
            if c.get("colour") in (None, "empty"):
                cv2.ellipse(named, (int(c["x"]), int(c["y"])),
                            (c["rx"], c["ry"]), 0, 0, 360, (60, 60, 60), 2)
                continue
            cv2.ellipse(named, (int(c["x"]), int(c["y"])), (c["rx"], c["ry"]),
                        0, 0, 360, c["bgr"], -1)
            cv2.ellipse(named, (int(c["x"]), int(c["y"])), (c["rx"], c["ry"]),
                        0, 0, 360, (255, 255, 255), 2)
    held = sum(1 for c in cells if c.get("colour") not in (None, "empty"))
    stage("Name the colours",
          f"{held} balls. Nearest taught colour, measured in that colour's own "
          "spread rather than raw units. Empty is deliberately not a class to "
          "recognise: it is whatever no taught colour explains, because empty "
          "is the ceiling and nobody controls the ceiling. Where that is not "
          "enough, and in afternoon sun it is not, the cell has to be domed "
          "as well: a ball is a sphere lit from below and is brighter in the "
          "middle than at the rim. That is the same thing your eye uses when "
          "the colours stop being separable.",
          named)
    return out


def contact_sheet(frame):
    """All the stages as one image, for a slide."""
    shots = render(frame)
    tiles = []
    for s in shots:
        raw = base64.b64decode(s["jpeg"])
        tile = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        cv2.rectangle(tile, (0, 0), (tile.shape[1], 34), (0, 0, 0), -1)
        cv2.putText(tile, s["title"], (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(tile)
    while len(tiles) % 2:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i:i + 2]) for i in range(0, len(tiles), 2)]
    return np.vstack(rows)
