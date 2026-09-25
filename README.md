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
- macOS, for the exposure control, which goes through
  [uvc-util](https://github.com/jtfrey/uvc-util). `setup.py` builds it for you.
- [uv](https://docs.astral.sh/uv/), which brings its own Python

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
```

That is the whole install. `uv run` builds the environment from `uv.lock` the
first time it is asked for one, about half a minute, and finds it already
there every time after. The versions are pinned rather than floored, so a
second Mac gets the ones this was built against and not whatever shipped
since.

## Setting it up

Once per rig, in this order.

**1. Focus the lens**, if it has never been set or has moved.

```sh
uv run focus_check.py
```

**1b. Aim the camera**, once, when the box is first put together.

```sh
uv run setup.py --centre
```

Aim it at the middle of the sensor, not at the middle of the board. The lens
axis on this module images at pixel 841 of 1920, so a camera centred under the
board sits 119 px off-centre in the frame: 8 px of clearance at one end and 234
at the other. The window shows two crosses; slide the camera until they meet.

**2. Print and stick on the tags.** Four AprilTags, 25 mm, two above the grid
and two below, on the underside of the sheet facing the camera. Print at 100%
and check the ruler on the sheet before cutting.

```sh
uv run make_tags.py          # writes captures/tags.pdf
```

**3. Everything else**, as one guided walk: tools, then exposure, then
geometry, then colours. It tells you what to do at each step.

```sh
uv run setup.py
```

The order is not arbitrary. Tools first, because the only thing in that step
is uvc-util and everything after it is measured off a frame whose exposure
uvc-util sets; if it is missing, setup offers to clone and compile it.
Exposure second, because geometry and colour are both read off a correctly
exposed frame; left on auto the camera meters a mostly-black board, opens
right up, and the room coming through the empty holes ends up as bright as
the balls while the balls themselves blow out. Geometry third, because
teaching a colour means sampling cells, which means knowing where they are.

Individual steps, when only one thing has changed:

```sh
uv run setup.py --tools
uv run setup.py --exposure
uv run setup.py --geometry     # --reclick to redo the corner clicks
uv run setup.py --colours
```

## Playing it

Double-click **Bubblegum.command** in Finder, which builds the environment if
it has to, starts the server and opens the browser. Or, from a terminal:

```sh
uv run play.py
```

Then open http://127.0.0.1:8099 and press space. Ctrl-c in the terminal
stops it. Each colour is a voice, synthesised by default and replaceable by
an upload or a recording from your microphone.

It checks the rig before serving anything and refuses to start if the camera
is missing or the grid has never been located, saying which. Once it is
running, the browser shows the state instead: `live`, `settling`, or
`cannot see the board`.

## Explaining it to people

The webapp has a second view, behind **Show the pipeline**, that walks through
what the vision side does to a frame: the raw picture, the lighting flattened,
the holes found, the tags, the grid placed, each cell sampled, each colour
named. Left and right arrows step through, and it stays live, so moving a ball
moves every stage.

Each step is there because of something visibly wrong in the one before it,
which is what makes it worth showing in that order.

```sh
uv run probe.py --sheet pipeline.png    # all the stages as one image
```

## Looking at what it sees

```sh
uv run probe.py              # the grid, live, with each ball labelled
uv run probe.py --once       # one frame, to captures/: a csv and two plots
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
| `board/framing.py` | where the board sits in the frame, and the room around it |
| `board/files.py` | where the tools put what they produce |
| `Bubblegum.command` | the same as `play.py`, for people who use Finder |

| | |
|---|---|
| `board/camera.py` | opening it, and pinning the exposure through uvc-util |
| `board/geometry.py` | finding the holes, the warp, the tags, the reference |
| `board/colour.py` | sampling a cell, teaching a colour, classifying |
| `board/pattern.py` | what is on the board, not what the last frame showed |
| `board/reader.py` | one frame in, sixty-four classified cells out |
| `board/view.py` | drawing the board on a frame |
| `app/` | the instrument itself: clock, voices, samples |

Everything the tools produce goes in `captures/`: frames saved with `s`, the
plots and the CSV from probe, the printable tag sheet. Gitignored, regenerated
on demand, and safe to empty whenever it gets big.

Calibration is the other thing outside the repo, and it is deliberately not in
`captures/`. `camera.json`, `corners.json`, `warp.json`, `tag_reference.json`
and `prototypes.json` describe one rig at one moment, they are read on every
run, and losing them costs a setup walk rather than a rerun.

`camera.json` records the camera by **name**, not just by index, and both
commands check it before trusting what they opened. An index is a property
of what happened to be plugged in when the machine booted: unplug the board's
camera and index 0 silently becomes the laptop's own, which opens perfectly
happily and shows you your face. `--index` and `--uvc-index` still override
it. The two are different numberings, so they can disagree: OpenCV counts the
cameras AVFoundation offers, uvc-util counts the ones on the USB bus.

## Moving it to another Mac

There is no download-and-play version of this, because half the instrument is
a physical board. What there is: a clone, an `install uv` line, and the same
guided walk you ran the first time.

```sh
git clone https://github.com/thetable/bubblegumsequencer
cd bubblegumsequencer
uv run setup.py            # builds uvc-util, then walks the rig
```

Roughly ten minutes, nearly all of it spent placing the board and teaching the
colours rather than waiting on software. The environment is 24 seconds and the
uvc-util build is a few more.

Two things do not travel and are not meant to. Calibration describes your
board under your lights, so it is measured again rather than copied. Camera
permission is granted per app by macOS, so the first run prompts once; a
double-clicked `Bubblegum.command` runs inside Terminal and inherits whatever
Terminal was already given.
