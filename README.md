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

**1. Focus the lens**, if it has never been set or has moved.

```sh
python focus_check.py
```

**2. Print and stick on the tags.** Four AprilTags, 25 mm, two above the grid
and two below, on the underside of the sheet facing the camera. Print at 100%
and check the ruler on the sheet before cutting.

```sh
python make_tags.py          # writes tags.pdf
```

**3. Everything else**, as one guided walk: exposure, then geometry, then
colours. It tells you what to do at each step.

```sh
python setup.py
```

The order is not arbitrary. Exposure first, because geometry and colour are
both read off a correctly exposed frame; left on auto the camera meters a
mostly-black board, opens right up, and the room coming through the empty
holes ends up as bright as the balls while the balls themselves blow out.
Geometry second, because teaching a colour means sampling cells, which means
knowing where they are.

Individual steps, when only one thing has changed:

```sh
python setup.py --exposure
python setup.py --geometry     # --reclick to redo the corner clicks
python setup.py --colours
```

## Playing it

```sh
python play.py
```

Then open http://127.0.0.1:8099 and press space. Ctrl-c in the terminal
stops it. Each colour is a voice, synthesised by default and replaceable by
an upload or a recording from your microphone.

It checks the rig before serving anything and refuses to start if the camera
is missing or the grid has never been located, saying which. Once it is
running, the browser shows the state instead: `live`, `settling`, or
`cannot see the board`.

## Looking at what it sees

```sh
python probe.py              # the grid, live, with each ball labelled
python probe.py --once       # one frame, to cells.csv and two plots
```

Keys: `space` freeze, `s` save the frame, `c` cycle what the cells are filled
with. `--learn` pins each cell's position more firmly as holes become
visible, which fills in ones that no single frame can show.

## The parts

Three commands, on one library.

| | |
|---|---|
| `play.py` | the instrument: serves the sequencer and streams the board to it |
| `setup.py` | the guided walk: exposure, geometry, colours |
| `probe.py` | a window on what the camera sees, for when something is wrong |
| `make_tags.py` | the printable tag sheet |
| `focus_check.py` | setting the lens |

| | |
|---|---|
| `board/camera.py` | opening it, and pinning the exposure through uvc-util |
| `board/geometry.py` | finding the holes, the warp, the tags, the reference |
| `board/colour.py` | sampling a cell, teaching a colour, classifying |
| `board/pattern.py` | what is on the board, not what the last frame showed |
| `board/reader.py` | one frame in, sixty-four classified cells out |
| `board/view.py` | drawing the board on a frame |
| `app/` | the instrument itself: clock, voices, samples |

Calibration lives in `camera.json`, `corners.json`, `warp.json`,
`tag_reference.json` and `prototypes.json`. None of it is in the repo: it
describes one rig at one moment, and it is rewritten as the tools run.

`camera.json` records the camera by **name**, not just by index, and both
commands check it before trusting what they opened. An index is a property
of what happened to be plugged in when the machine booted: unplug the board's
camera and index 0 silently becomes the laptop's own, which opens perfectly
happily and shows you your face. `--index` and `--uvc-index` still override
it. The two are different numberings, so they can disagree: OpenCV counts the
cameras AVFoundation offers, uvc-util counts the ones on the USB bus.
