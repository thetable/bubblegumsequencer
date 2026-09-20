"""The camera, and the one thing that mattered most: its exposure.

Left on automatic the camera meters a frame that is mostly black sheet, opens
right up, and two bad things follow at once. The room coming through the empty
holes ends up as bright as the balls, and the balls themselves blow out. Pinned
by hand, neither happens.

OpenCV cannot set exposure on this camera under macOS. It accepts the property
and silently ignores it, which cost an evening to discover, so everything here
goes through uvc-util instead.
"""

import os
import shutil
import subprocess

import cv2
import numpy as np

EXPOSURE_FILE = "exposure.json"

# The camera's own exposure-time-abs control runs 1 to 5000 and auto picks about
# 157, which blows the balls out. These are the settings worth trying, spaced
# geometrically so each step is roughly half a stop.
EXPOSURE_CANDIDATES = [4, 6, 9, 13, 19, 27, 38, 55, 78, 110, 157, 220]

UVC_MANUAL, UVC_AUTO = 1, 8


def uvc_util(index, *args):
    """Talk to the camera's UVC controls through uvc-util.

    OpenCV cannot set exposure on this camera under macOS: the property is
    accepted and silently ignored. uvc-util walks the USB bus with IOKit and
    sets the control on the device itself, which the capture then inherits.

    The binary sits next to this file and is not in the repo. To rebuild it:

        git clone --depth 1 https://github.com/jtfrey/uvc-util
        clang -fno-objc-arc -O2 -Wno-everything -framework Foundation \
              -framework IOKit -framework CoreFoundation \
              uvc-util/src/*.m -o uvc-util
    """
    exe = shutil.which("uvc-util") or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "uvc-util")
    if not os.path.exists(exe):
        return None
    try:
        done = subprocess.run([exe, "-I", str(index), *args],
                              capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout.strip() if done.returncode == 0 else None


def uvc_devices():
    """The UVC cameras uvc-util can see, as {index: name}.

    Worth asking, because OpenCV will happily open whatever camera is at the
    index you gave it. With the board's camera unplugged that is the laptop's
    own, and everything downstream then reports cheerfully on a picture of
    your face.
    """
    exe = shutil.which("uvc-util") or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), os.pardir, "uvc-util")
    if not os.path.exists(exe):
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
    import json
    cap = open_camera(args)
    if cap is None:
        return None
    if os.path.exists(EXPOSURE_FILE):
        with open(EXPOSURE_FILE) as fh:
            set_exposure(getattr(args, "uvc_index", 0), json.load(fh)["exposure"])
    frame = grab(cap, 10)
    cap.release()
    return frame
