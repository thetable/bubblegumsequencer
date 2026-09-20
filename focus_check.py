#!/usr/bin/env python3
"""Bubblegum sequencer, step 0: camera check and focus aid.

Live preview with a focus score, so the M12 lens can be set by hand.
The lens has no scale, so the number is the only reliable guide: turn it
slowly and watch for the peak.

Keys:
    q   quit
    s   save a full-resolution frame
    r   reset the peak-so-far
    a   toggle the auto-exposure request (see note below)
Mouse:
    click anywhere to move the focus probe
"""

import argparse
import datetime
import sys
import time

import cv2
import numpy as np

PROBE = 96  # side of the square focus probe, in source pixels


def find_cameras(max_index=4):
    """Report which capture indices actually open."""
    found = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i, cv2.CAP_AVFOUNDATION)
        if cap.isOpened():
            ok, frame = cap.read()
            if ok and frame is not None:
                h, w = frame.shape[:2]
                found.append((i, w, h))
        cap.release()
    return found


def open_camera(index, width, height, fps):
    cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        return None
    # MJPEG first: without it macOS often negotiates YUY2 and drops to a crawl
    # at 1080p.
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    return cap


def focus_score(gray_patch):
    """Variance of the Laplacian. Higher is sharper.

    The absolute value is meaningless; only its behaviour as you turn the
    lens matters. It also scales with scene contrast and exposure, so keep
    the scene and the light still while focusing.
    """
    return float(cv2.Laplacian(gray_patch, cv2.CV_64F).var())


def draw_overlay(view, scale, probe_xy, score, peak, stats, fps, shape):
    x, y = probe_xy
    half = PROBE // 2
    p0 = (int((x - half) * scale), int((y - half) * scale))
    p1 = (int((x + half) * scale), int((y + half) * scale))
    cv2.rectangle(view, p0, p1, (0, 255, 0), 2)

    lo, mean, hi, clipped = stats
    ratio = score / peak if peak > 0 else 0.0
    lines = [
        f"focus {score:8.0f}   peak {peak:8.0f}   {ratio * 100:5.1f}% of peak",
        f"exposure  min {lo:3d}  mean {mean:5.1f}  max {hi:3d}  clipped {clipped:4.2f}%",
        f"{shape[1]}x{shape[0]} @ {fps:4.1f} fps",
        "click = move probe   s = save   r = reset peak   a = auto-exposure   q = quit",
    ]
    for i, text in enumerate(lines):
        origin = (12, 28 + i * 26)
        cv2.putText(view, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(view, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 0), 1, cv2.LINE_AA)

    # A crude bar so the peak is visible at a glance while you turn the lens.
    bar_w = int(360 * min(ratio, 1.0))
    cv2.rectangle(view, (12, 130), (372, 150), (60, 60, 60), -1)
    cv2.rectangle(view, (12, 130), (12 + bar_w, 150), (0, 255, 0), -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int, default=None,
                    help="capture index; omit to list what is available")
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--preview-width", type=int, default=1280)
    args = ap.parse_args()

    if args.index is None:
        cams = find_cameras()
        if not cams:
            print("No cameras opened. On macOS this is usually a permissions "
                  "problem: System Settings > Privacy & Security > Camera, "
                  "and enable your terminal app.")
            return 1
        print("Cameras that opened:")
        for i, w, h in cams:
            print(f"  index {i}: {w}x{h}")
        print("\nRe-run with --index N. The built-in FaceTime camera is "
              "usually 0, so the USB module is probably the other one.")
        return 0

    cap = open_camera(args.index, args.width, args.height, args.fps)
    if cap is None:
        print(f"Could not open index {args.index}.")
        return 1

    ok, frame = cap.read()
    if not ok or frame is None:
        print("Opened the device but got no frame.")
        return 1

    h, w = frame.shape[:2]
    print(f"Negotiated {w}x{h}. Asked for {args.width}x{args.height}.")
    if (w, h) != (args.width, args.height):
        print("The camera did not give the requested size. Not fatal, but "
              "check the listed modes if you expected 1080p.")

    probe = [w // 2, h // 2]
    peak = 0.0
    auto_exposure = True
    scale = min(1.0, args.preview_width / w)

    window = "bubblegum focus"
    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)

    def on_mouse(event, mx, my, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN:
            probe[0] = int(np.clip(mx / scale, PROBE, w - PROBE))
            probe[1] = int(np.clip(my / scale, PROBE, h - PROBE))

    cv2.setMouseCallback(window, on_mouse)

    last = time.time()
    fps_smoothed = 0.0

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            print("Dropped frame, stopping.")
            break

        now = time.time()
        dt = now - last
        last = now
        if dt > 0:
            fps_smoothed = 0.9 * fps_smoothed + 0.1 * (1.0 / dt)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        half = PROBE // 2
        x, y = probe
        patch = gray[y - half:y + half, x - half:x + half]
        score = focus_score(patch)
        peak = max(peak, score)

        clipped = 100.0 * float(np.count_nonzero(gray >= 254)) / gray.size
        stats = (int(gray.min()), float(gray.mean()), int(gray.max()), clipped)

        view = cv2.resize(frame, None, fx=scale, fy=scale) if scale < 1.0 else frame.copy()
        draw_overlay(view, scale, probe, score, peak, stats, fps_smoothed, frame.shape)
        cv2.imshow(window, view)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("r"):
            peak = 0.0
        if key == ord("s"):
            name = datetime.datetime.now().strftime("frame_%Y%m%d_%H%M%S.png")
            cv2.imwrite(name, frame)
            print(f"saved {name}")
        if key == ord("a"):
            auto_exposure = not auto_exposure
            # macOS UVC support for these is patchy. If nothing changes on
            # screen, the camera is ignoring the request and you are stuck
            # with whatever it decides.
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3 if auto_exposure else 1)
            print(f"requested auto-exposure = {auto_exposure}")

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
