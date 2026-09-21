"""The camera, and the one thing that mattered most: its exposure.

Left on automatic the camera meters a frame that is mostly black sheet, opens
right up, and two bad things follow at once. The room coming through the empty
holes ends up as bright as the balls, and the balls themselves blow out. Pinned
by hand, neither happens.

OpenCV cannot set exposure on this camera under macOS. It accepts the property
and silently ignores it, which cost an evening to discover, so everything here
goes through uvc-util instead.
"""

import glob
import json
import os
import shutil
import subprocess
import tempfile

import cv2
import numpy as np

CAMERA_FILE = "camera.json"
EXPOSURE_FILE = "exposure.json"      # older, read if camera.json is absent

# The camera's own exposure-time-abs control runs 1 to 5000 and auto picks about
# 157, which blows the balls out. These are the settings worth trying, spaced
# geometrically so each step is roughly half a stop.
EXPOSURE_CANDIDATES = [4, 6, 9, 13, 19, 27, 38, 55, 78, 110, 157, 220]

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
    if os.path.exists(EXPOSURE_FILE):
        with open(EXPOSURE_FILE) as fh:
            return {"exposure": json.load(fh)["exposure"]}
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
        cap = cv2.VideoCapture(args.index, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        print(f"Could not open camera index {args.index}.")
        print("On macOS the terminal app needs camera permission under "
              "Privacy & Security.")
        return None
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    return cap


def grab(cap, n=8):
    """Throw away a few frames so the sensor has settled before we look."""
    ok, frame = False, None
    for _ in range(n):
        ok, frame = cap.read()
    return frame if ok else None


def set_exposure(index, value):
    """Fix the exposure, or hand it back to the camera when value is None."""
    if value is None:
        return uvc_util(index, "-s", f"auto-exposure-mode={UVC_AUTO}") is not None
    ok = uvc_util(index, "-s", f"auto-exposure-mode={UVC_MANUAL}") is not None
    return ok and uvc_util(index, "-s", f"exposure-time-abs={int(value)}") is not None


def dial_exposure(cap, index, target, corners):
    """Sweep the exposure and keep the brightest setting that does not clip.

    Balls are the brightest thing on the board, so they are what blows out
    first, and a blown pixel has lost its colour for good. Better a slightly
    dark picture than a white one.
    """
    if uvc_util(index, "-o", "exposure-time-abs") is None:
        print("  uvc-util is not answering, so the exposure cannot be set.")
        print("  It should sit next to this script, or on the PATH.")
        return None

    set_exposure(index, None)          # start from the camera's own choice
    base = grab(cap)
    if base is None:
        return None
    print(f"  auto exposure gives {clipped_fraction(base, corners) * 100:.2f}% clipped. "
          f"Sweeping for something under {target * 100:.2f}%.")

    results = []
    for value in EXPOSURE_CANDIDATES:
        if not set_exposure(index, value):
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
        print("\n  Nothing changed across the whole range, which should not "
              "happen now that the control is being set on the device.")
        print("  Check `uvc-util -I 0 -o exposure-time-abs` by hand.")
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
    pinned = settings().get("exposure")
    if pinned is not None:
        set_exposure(getattr(args, "uvc_index", 0), pinned)
    frame = grab(cap, 10)
    cap.release()
    return frame
