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

    def stage(title, caption, detail, image):
        """caption says what this step is for; detail says how it works."""
        out.append({"title": title, "caption": caption, "detail": detail,
                    "jpeg": _jpeg(_mirror(image))})

    stage("What the camera sees",
          "A 130 degree lens, 200 mm under a scratched black sheet, looking "
          "up. Everything that follows is undoing something you can already "
          "see going wrong here.",
          "The rows bow outwards: barrel distortion puts about 67 px between "
          "neighbouring holes at the edge of the frame and 135 px in the "
          "middle. The corners fall dark, the light strips are in shot, and a "
          "hole is not reliably brighter than the sheet around it, because "
          "both depend on where in the frame they happen to fall. No single "
          "brightness will separate them.",
          frame)

    # 1. Flat field ------------------------------------------------------
    f = gray.astype(np.float32)
    background = cv2.GaussianBlur(f, (0, 0), geometry.BACKGROUND_SIGMA)
    ratio = f / (background + 1e-6)
    stage("Flatten the lighting",
          "Divide the picture by a blurred copy of itself. That turns \u201chow "
          "bright is this\u201d into \u201chow much brighter is this than its own "
          "surroundings\u201d, which is the only version of the question with "
          "the same answer everywhere on the board.",
          f"The blur is wide, {geometry.BACKGROUND_SIGMA} px, far wider than a "
          "hole, so it captures the lighting and not the things being looked "
          "for. Dividing then cancels anything that varies smoothly: the "
          "lens darkening towards its corners, the strips being nearer one "
          "end than the other, the room falling away at the edges. A hole in "
          "the dark corner and a hole in the bright middle can differ "
          "threefold in raw brightness and come out the same here. Without "
          "this step there is no one threshold that finds both.",
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
          f"Keep everything at least 20% brighter than its surroundings, then "
          f"keep only what is hole-shaped. {len(blobs)} of 64 here, and never "
          "all 64 in any one frame.",
          "The flattened ratio is thresholded at 1.20, opened to drop speckle "
          "and closed to fill pinholes, and what remains is split into "
          "connected blobs. A blob survives if it is roughly round, width "
          "over height between 0.45 and 2.2, and close to the median blob "
          "area, between 0.35 and 2.5 times it. The median is the trick: "
          "every hole is the same physical size, so the picture itself "
          "supplies the scale and there is no size to type in. A hole still "
          "vanishes when the sheet beside it reflects a strip back at exactly "
          "the brightness of the room coming through it, which is why this "
          "step is not allowed to be the one that places the grid.",
          found)

    # 3. Tags ------------------------------------------------------------
    marked = frame.copy()
    for tag_id, corner in sorted(tags.items()):
        cv2.polylines(marked, [np.int32(corner)], True, (255, 80, 255), 3)
        cv2.putText(marked, str(tag_id), tuple(np.int32(corner[0]) + [8, -10]),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 80, 255), 2, cv2.LINE_AA)
    stage("Find the tags",
          f"Four AprilTags stuck to the underside of the sheet, {len(tags)} "
          "visible. They are not here to find holes. They are here to say "
          "where the board is right now.",
          "Each tag carries a 36h11 code, so it says which of the four it is "
          "rather than merely being a marker, and it offers four corners that "
          "can be located to well under a pixel. Three visible tags are "
          "enough to pin the board down, which matters because one of the "
          "four is usually the marginal one. With all four the fit is "
          "over-determined, so the tags can be checked against each other and "
          "their disagreement becomes a number worth watching. That is what "
          "lets the board be shoved about mid-song and have the grid follow.",
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
    _, rows = geometry.load_reference()
    pinned = 0 if rows is None else int((rows[:, 4] > 1).sum())
    stage("Place all 64",
          "The grid is not found in this frame. It was measured once, stored, "
          "and is now carried onto wherever the tags say the board is, which "
          "is how cells whose holes nothing can see still get placed.",
          "A homography maps the stored tag corners onto the ones seen now, "
          "and all 64 stored positions ride through it. The stored grid "
          "itself was fitted over detected holes as a polynomial, cubic "
          f"across the columns and quadratic down the rows: the pitch runs "
          f"{edge:.0f} px at the edge to {middle:.0f} px in the middle, and a "
          "four-corner homography cannot bend like that. Trying it that way "
          "put cells up to 94 px out. The store also improves as the thing is "
          "used: any cell whose hole is clearly visible has its position "
          f"averaged into the record, so the map fills in over runs. {pinned} "
          "of 64 rest on real sightings so far.",
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
          "One colour per cell: the median inside an ellipse, with the "
          "brightest 5% thrown away first.",
          "An ellipse rather than a circle because that is the shape a round "
          "hole presents to a camera 49 degrees off its axis. A circle would "
          "have to shrink to the narrow dimension and so discard most of the "
          "aperture on the outer columns, which are precisely the cells with "
          "the least light to spare. Median rather than mean, because a "
          "single glint drags a mean anywhere it likes. Dropping the "
          "brightest 5% first removes the specular highlight off the shell, "
          "which is the one part of a ball that is the colour of the lamp "
          "instead of the colour of the ball.",
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
          f"{held} balls. Each cell takes the nearest taught colour, measured "
          "in that colour's own spread, and only if it is close enough. "
          "Nothing at all is taught about what empty looks like.",
          "Empty is whatever no taught colour explains, because empty is the "
          "ceiling and nobody controls the ceiling. Measuring in spreads "
          "rather than raw units is what lets a tight cluster sit near a "
          f"loose one without the two being confused. Past "
          f"{colour.MAX_DISTANCE} spreads a cell is disowned. Where colour "
          "alone does not settle it, the cell must also be domed, brighter in "
          "its middle than at its rim, because a ball is a sphere lit from "
          "below and an empty hole is a flat view of the room. In afternoon "
          "sun that shape test is the only thing left separating a green ball "
          "from a lit hole, and it is the same cue your eye reaches for.",
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
