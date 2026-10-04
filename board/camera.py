"""The camera, and the one thing that mattered most: its exposure.

Left on automatic the camera meters a frame that is mostly black sheet, opens
right up, and two bad things follow at once. The room coming through the empty
holes ends up as bright as the balls, and the balls themselves blow out. Pinned
by hand, neither happens.

Setting it is where the machines part company. Under macOS OpenCV accepts the
property and silently ignores it, which cost an evening to discover, so there
it goes through uvc-util and IOKit. Windows and Linux do honour the property,
so there it is set on the capture itself and uvc-util is not wanted.

The number means different things in each case. uvc-util and V4L2 count in
hundred-microsecond steps; DirectShow counts in stops, as the log to base two
of the time in seconds. 157 and -6 are the same fifteen milliseconds. So the
value is stamped with the machine that produced it and only used there.
"""

import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile

import cv2
import numpy as np

CAMERA_FILE = "camera.json"

SYSTEM = sys.platform                      # "darwin", "win32", "linux"
ON_MAC = SYSTEM == "darwin"

# Which capture backend to ask for. Left to choose for itself OpenCV picks
# MSMF on Windows, which is slow to open and often refuses 1920 x 1080.
BACKENDS = {"darwin": cv2.CAP_AVFOUNDATION, "win32": cv2.CAP_DSHOW}
BACKEND = BACKENDS.get(SYSTEM, cv2.CAP_V4L2)

# What OpenCV wants in CAP_PROP_AUTO_EXPOSURE to mean automatic and manual.
# Another thing the two disagree about, and silently.
AUTO_MANUAL = {"win32": (0.75, 0.25)}.get(SYSTEM, (3.0, 1.0))

# The settings worth trying, in whatever unit this machine counts in. The
# hundred-microsecond ladder is spaced about half a stop a step; DirectShow
# cannot do better than a whole stop, since it only takes integers.
EXPOSURE_CANDIDATES = (
    [-13, -12, -11, -10, -9, -8, -7, -6, -5, -4] if SYSTEM == "win32"
    else [4, 6, 9, 13, 19, 27, 38, 55, 78, 110, 157, 220])

UVC_MANUAL, UVC_AUTO = 1, 8

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UVC_SOURCE = "https://github.com/jtfrey/uvc-util"


def uvc_path():
    """Where uvc-util is, or None.

    One answer for the whole program. It used to be worked out separately
    wherever it was needed, the two answers drifted apart in a refactor, and
    the one that was wrong failed by returning None rather than by
    complaining: the exposure quietly went back to automatic, which is the
    single thing this module exists to prevent.
    """
    if not ON_MAC:
        return None            # an IOKit binary; there is no other build
    found = shutil.which("uvc-util")
    if found:
        return found
    beside = os.path.join(ROOT, "uvc-util")
    return beside if os.path.exists(beside) else None


def _last_line(text):
    lines = [l for l in (text or "").strip().splitlines() if l.strip()]
    return lines[-1] if lines else "no output"


def build_uvc_util():
    """Clone and compile uvc-util into the repo root. Returns the path or None.

    Not vendored: it is somebody else's code and a Mac-only binary. Building
    it is two commands, which is two commands too many to leave in a README
    for a person to copy at the one moment they are least in the mood.
    """
    missing = [t for t in ("git", "clang") if shutil.which(t) is None]
    if missing:
        print(f"  {' and '.join(missing)} not installed.")
        print("  Install the command line tools:  xcode-select --install")
        return None

    target = os.path.join(ROOT, "uvc-util")
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "uvc-util")
        print(f"  cloning {UVC_SOURCE}")
        done = subprocess.run(["git", "clone", "--depth", "1", UVC_SOURCE, src],
                              capture_output=True, text=True)
        if done.returncode:
            print(f"  clone failed: {_last_line(done.stderr)}")
            return None

        sources = sorted(glob.glob(os.path.join(src, "src", "*.m")))
        if not sources:
            print("  the clone has no src/*.m: upstream has moved things about.")
            print(f"  Have a look at {UVC_SOURCE} and build it by hand.")
            return None

        print(f"  compiling {len(sources)} files")
        done = subprocess.run(
            ["clang", "-fno-objc-arc", "-O2", "-Wno-everything",
             "-framework", "Foundation", "-framework", "IOKit",
             "-framework", "CoreFoundation", *sources, "-o", target],
            capture_output=True, text=True)
        if done.returncode:
            print(f"  compile failed: {_last_line(done.stderr)}")
            return None

    print(f"  built {target}")
    return target


def uvc_util(index, *args):
    """Talk to the camera's UVC controls through uvc-util.

    OpenCV cannot set exposure on this camera under macOS: the property is
    accepted and silently ignored. uvc-util walks the USB bus with IOKit and
    sets the control on the device itself, which the capture then inherits.

    The binary is not in the repo; setup.py offers to build it.
    """
    exe = uvc_path()
    if exe is None:
        return None
    try:
        done = subprocess.run([exe, "-I", str(index), *args],
                              capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout.strip() if done.returncode == 0 else None


def settings():
    """Which camera this rig uses, and how it is set.

    The device name rather than the index, because an index is a property of
    what happened to be plugged in when the machine booted, while the name is
    a property of the rig. Unplug the board's camera and index 0 silently
    becomes the laptop's own; the name never does that.

    Two indices are kept alongside it because they are two different
    numberings and can disagree: OpenCV counts the cameras AVFoundation
    offers, uvc-util counts the ones on the USB bus.
    """
    if os.path.exists(CAMERA_FILE):
        with open(CAMERA_FILE) as fh:
            return json.load(fh)
    return {}


def remember(**fields):
    kept = settings()
    kept.update({k: v for k, v in fields.items() if v is not None})
    with open(CAMERA_FILE, "w") as fh:
        json.dump(kept, fh, indent=1)
    return kept


def resolve(args):
    """Fill in whatever the command line left out, from camera.json.

    Returns the expected device name, or None if the rig has never said.
    """
    kept = settings()
    if getattr(args, "index", None) is None:
        args.index = kept.get("index", 0)
    if getattr(args, "uvc_index", None) is None:
        args.uvc_index = kept.get("uvc_index", 0)
    return kept.get("device")


def uvc_devices():
    """The UVC cameras uvc-util can see, as {index: name}.

    Worth asking, because OpenCV will happily open whatever camera is at the
    index you gave it. With the board's camera unplugged that is the laptop's
    own, and everything downstream then reports cheerfully on a picture of
    your face.
    """
    exe = uvc_path()
    if exe is None:
        return None                      # cannot tell, which is not the same
    try:
        done = subprocess.run([exe, "-d"], capture_output=True, text=True,
                              timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    found = {}
    for line in done.stdout.splitlines():
        bits = line.split()
        if bits and bits[0].isdigit():
            found[int(bits[0])] = bits[-1]
    return found


def clipped_fraction(frame, corners=None):
    """How much of the board is pinned at 255.

    Restricted to the board, because the strips themselves are in shot and a
    light source is clipped by definition. Measuring the whole frame chased a
    floor it could never reach and drove the exposure four stops too dark.
    """
    blown = frame.max(axis=2) >= 250
    if corners is None:
        return float(blown.mean())
    board = cv2.fillPoly(np.zeros(blown.shape, np.uint8),
                         np.int32([corners]), 255) > 0
    return float(blown[board].mean()) if board.any() else float(blown.mean())


class quiet:
    """Hold the OS quiet while the camera opens.

    macOS logs a deprecation notice about Continuity Cameras straight to fd 2
    from inside AVFoundation, every single run. It is not ours to fix and it
    says nothing useful, so fd 2 is parked on /dev/null for the one call that
    provokes it and put straight back. Anything we actually want to report
    about the camera, we report ourselves.
    """

    def __enter__(self):
        self.saved = os.dup(2)
        self.null = os.open(os.devnull, os.O_WRONLY)
        os.dup2(self.null, 2)

    def __exit__(self, *_):
        os.dup2(self.saved, 2)
        os.close(self.null)
        os.close(self.saved)


def open_camera(args):
    with quiet():
        cap = cv2.VideoCapture(args.index, BACKEND)
    if not cap.isOpened():
        print(f"Could not open camera index {args.index}.")
        if ON_MAC:
            print("On macOS the terminal app needs camera permission under "
                  "Privacy & Security.")
        elif SYSTEM == "win32":
            print("On Windows, check Settings > Privacy > Camera, and that "
                  "nothing else already has it open.")
        return None
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    return cap


def rotation():
    """Half a turn, or none. How the camera ended up bolted in."""
    return 180 if int(settings().get("rotate", 0)) == 180 else 0


def orient(frame):
    """The frame the way the rig is actually mounted.

    Done here, at the moment of capture, rather than by turning the pattern
    round at the far end. Everything downstream then works on a picture taken
    by a camera the right way up: the warp, the tags, the sample ellipses, the
    pipeline pictures and the debug window all stay as they were, and there is
    one fact about the rig instead of a correction in every tool.

    Half a turn reverses both axes at once, which is why it is one switch and
    not two. It is not a vertical mirror; no mounting produces one of those.
    """
    if frame is None or rotation() != 180:
        return frame
    return cv2.rotate(frame, cv2.ROTATE_180)


def grab(cap, n=8):
    """Throw away a few frames so the sensor has settled before we look."""
    ok, frame = False, None
    for _ in range(n):
        ok, frame = cap.read()
    return orient(frame) if ok else None


def set_exposure(cap, index, value):
    """Fix the exposure, or hand it back to the camera when value is None.

    Takes the capture as well as the index because off macOS the control lives
    on the open capture rather than on the device.
    """
    if ON_MAC:
        if value is None:
            return uvc_util(index, "-s", f"auto-exposure-mode={UVC_AUTO}") is not None
        ok = uvc_util(index, "-s", f"auto-exposure-mode={UVC_MANUAL}") is not None
        return ok and uvc_util(index, "-s",
                               f"exposure-time-abs={int(value)}") is not None
    if cap is None:
        return False
    auto, manual = AUTO_MANUAL
    if value is None:
        return bool(cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, auto))
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, manual)
    return bool(cap.set(cv2.CAP_PROP_EXPOSURE, float(value)))


def exposure_reachable(cap, index):
    """Whether this machine can actually set the exposure at all."""
    if ON_MAC:
        return uvc_util(index, "-o", "exposure-time-abs") is not None
    return cap is not None and cap.isOpened()


def apply_pinned(cap, index):
    """Set the exposure this rig was dialled to, if it was dialled here.

    A value carried from another machine would be read in the wrong unit: 38
    means under four milliseconds through uvc-util and sixteen seconds through
    DirectShow. Leaving it on automatic is bad; setting it to sixteen seconds
    is worse, so a value from elsewhere is declined out loud.
    """
    kept = settings()
    value = kept.get("exposure")
    if value is None:
        return None
    where = kept.get("exposure_on", "darwin")
    if where != SYSTEM:
        print(f"  Exposure {value} was dialled on {where}, and this is "
              f"{SYSTEM}, where that number means something else.")
        print("  Leaving it automatic. Run:  python setup.py --exposure")
        return None
    set_exposure(cap, index, value)
    return value


def dial_exposure(cap, index, target, corners):
    """Sweep the exposure and keep the brightest setting that does not clip.

    Balls are the brightest thing on the board, so they are what blows out
    first, and a blown pixel has lost its colour for good. Better a slightly
    dark picture than a white one.
    """
    if not exposure_reachable(cap, index):
        print("  The exposure cannot be set on this machine.")
        if ON_MAC:
            print("  uvc-util is not answering. It should sit next to this "
                  "script, or on the PATH.")
        return None

    set_exposure(cap, index, None)     # start from the camera's own choice
    base = grab(cap)
    if base is None:
        return None
    print(f"  auto exposure gives {clipped_fraction(base, corners) * 100:.2f}% clipped. "
          f"Sweeping for something under {target * 100:.2f}%.")

    results = []
    for value in EXPOSURE_CANDIDATES:
        if not set_exposure(cap, index, value):
            continue
        frame = grab(cap, 6)
        if frame is None:
            continue
        c = clipped_fraction(frame, corners)
        results.append((value, c, cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean()))
        print(f"    exposure {value:>6}  ->  {c * 100:6.2f}% clipped, "
              f"mean {results[-1][2]:5.1f}")

    if not results:
        return None
    spread = max(c for _, c, _ in results) - min(c for _, c, _ in results)
    if spread < 0.002:
        print("\n  Nothing changed across the whole range, so the camera is "
              "ignoring the setting.")
        print("  Check `uvc-util -I 0 -o exposure-time-abs` by hand." if ON_MAC
              else "  Some webcams only accept exposure while auto is off, and "
                   "some refuse it entirely.")
        return None

    usable = [r for r in results if r[1] <= target]
    if not usable:
        value, clip, _ = min(results, key=lambda r: r[1])
        print(f"\n  Nothing got under {target * 100:.2f}%. Best was exposure "
              f"{value} at {clip * 100:.2f}%. Take a strip out as well.")
        return value
    # The brightest of the clean settings: longest exposure means least noise,
    # and noise is what is left once clipping is dealt with.
    value, clip, _ = max(usable, key=lambda r: r[0])
    print(f"\n  Using exposure {value}: {clip * 100:.2f}% clipped.")
    return value


def single_frame(args):
    """One settled frame, exposure pinned if it has been dialled."""
    cap = open_camera(args)
    if cap is None:
        return None
    apply_pinned(cap, getattr(args, "uvc_index", 0))
    frame = grab(cap, 10)
    cap.release()
    return frame
