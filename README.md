# Bubblegum Sequencer

A 4 x 16 grid of holes in a black sheet. Drop coloured gumballs in; a camera
underneath reads them and a browser plays them as a step sequencer. Columns
are sixteenth notes, rows only let you stack several balls on the same one.

A rebuild of the Bubblegum Sequencer (Hesse & McDiarmid, CHI 2008), this time
with no hand-aligned grid and no hand-tuned colour thresholds: the board
carries AprilTags and the geometry is measured, and every colour is taught
from the balls themselves.

## What you need

- A UVC webcam, roughly 200 mm below the sheet, looking up
- LED strips inside the enclosure, lighting the sheet from below
- Python with `opencv-python`, `numpy` and `pillow`
- macOS only, for the exposure control: [uvc-util](https://github.com/jtfrey/uvc-util)

```sh
git clone --depth 1 https://github.com/jtfrey/uvc-util
clang -fno-objc-arc -O2 -Wno-everything -framework Foundation \
      -framework IOKit -framework CoreFoundation \
      uvc-util/src/*.m -o uvc-util
```

## Setting it up

Once per rig, in this order.

**1. Focus the lens.** Live preview with a sharpness trace.

```sh
python focus_check.py
```

**2. Print and stick on the tags.** Four AprilTags, 25 mm, two above the grid
and two below, on the underside of the sheet facing the camera. Print at 100%
and check the ruler on the sheet before cutting.

```sh
python make_tags.py          # writes tags.pdf
```

**3. Pin the exposure.** This matters more than anything else here. Left on
auto, the camera meters a mostly-black board, opens right up, and the room
coming through the empty holes ends up as bright as the balls while the balls
themselves blow out. Put some balls on the board first.

```sh
python cell_probe.py --dial-exposure
```

**4. Record where the grid sits.** Room lights on, so the empty holes are
visible. This is the only time you click anything: the four corner holes, in
the order it asks.

```sh
python cell_probe.py --reclick
```

After this the tags carry the geometry, and the board can be moved.

**5. Teach the colours.** Guided, one colour at a time. Include balls near the
edges of the board, where the light is weakest.

```sh
python live_view.py --teach
```

## Playing it

```sh
python server.py
```

Then open http://127.0.0.1:8099 and press space. Each colour is a voice,
synthesised by default and replaceable by an upload or a recording from your
microphone.

## Looking at what it sees

```sh
python live_view.py          # the grid, live, with each ball labelled
python cell_probe.py         # one frame, to cells.csv and two diagnostic plots
```

`live_view.py` keys: `space` freeze, `s` save the frame, `c` cycle what the
cells are filled with. `--learn` pins each cell's position more firmly as
holes become visible, which fills in ones that no single frame can show.

## The parts

| | |
|---|---|
| `cell_probe.py` | camera, exposure, finding the holes, the geometry, teaching colours |
| `live_view.py` | the same loop with a window on it, and the guided teach mode |
| `pattern.py` | what is on the board, as opposed to what the last frame showed |
| `server.py` | serves the instrument and streams the pattern to it |
| `app/` | the instrument: clock, voices, samples |
| `make_tags.py` | the printable tag sheet |
| `focus_check.py` | setting the lens, once |

Calibration lives in `corners.json`, `warp.json`, `exposure.json`,
`tag_reference.json` and `prototypes.json`. None of it is in the repo: it
describes one rig at one moment, and it is rewritten as the tools run.
