#!/usr/bin/env python3
"""The instrument. Reads the board and serves the sequencer to a browser.

Assumes the rig is set up. If it is not, setup.py does that and probe.py
shows you what the camera can see.

Python says what is on the board, the browser plays it, and the two never
wait for each other: vision can be slow and careful while the clock stays
solid. Server-sent events rather than a WebSocket, which is what the original
plan assumed: nothing travels back from the browser, and one-way needs no
handshake, no framing and no dependency.

Usage:
    python play.py                      # then open the printed address
    python play.py --image frame.png    # replay one frame, for working offline
"""

import argparse
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2

from board import camera, colour, geometry, pattern
from board.reader import read_board

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.join(HERE, "app")
TYPES = {".html": "text/html", ".js": "text/javascript", ".css": "text/css"}

STALE = 2.0     # seconds without a frame before we stop believing the pattern


class Board:
    """The current pattern, and whatever the vision loop last had to say."""

    def __init__(self, columns=geometry.COLS, rows=geometry.ROWS):
        self.columns, self.rows = columns, rows
        self.stabiliser = pattern.Stabiliser(columns * rows)
        self.lock = threading.Lock()
        self.version = 0
        self.note = "starting"
        self.last_frame = time.monotonic()

    def snapshot(self, flip=True):
        """The pattern as the player sees it.

        The camera is under the sheet, so its left is the player's right. The
        browser gets player order, because a sequencer that runs backwards
        against the board in front of you is unusable.
        """
        with self.lock:
            grid = [self.stabiliser.pattern[r * self.columns:(r + 1) * self.columns]
                    for r in range(self.rows)]
            if flip:
                grid = [list(reversed(row)) for row in grid]
            # Frames stopping is the failure that does not announce itself:
            # the pattern is still there and still correct-looking, just from
            # whenever the camera last worked. A machine going to sleep is
            # enough to cause it.
            silent = time.monotonic() - self.last_frame
            stale = silent > STALE
            return {"version": self.version,
                    "note": (f"no frame for {silent:.0f}s: {self.note}"
                             if stale else self.note),
                    "columns": self.columns, "rows": self.rows,
                    "grid": grid,
                    "blind": self.stabiliser.blind or stale,
                    "settling": self.stabiliser.blocked}

    def offer(self, cells, note):
        with self.lock:
            self.last_frame = time.monotonic()
            changed = self.stabiliser.update(
                pattern.readings_from(cells, cells is not None))
            self.note = note
            if changed:
                self.version += 1
            return changed


def start_camera(args):
    cap = camera.open_camera(args)
    if cap is None:
        return None
    if os.path.exists(camera.EXPOSURE_FILE):
        with open(camera.EXPOSURE_FILE) as fh:
            camera.set_exposure(args.uvc_index, json.load(fh)["exposure"])
    return cap


def preflight(args):
    """Everything that has to be true before serving. Returns the camera.

    Checked here rather than in the loop so that a rig which is not ready
    says so and stops, instead of starting a server that will never have
    anything to show. Once it is running, problems are the browser's to
    report; before it starts, they are the terminal's.
    """
    if not os.path.exists(geometry.REFERENCE_FILE):
        print(f"No {geometry.REFERENCE_FILE}: the grid has never been located.")
        print("Run:  python setup.py")
        return None
    if not os.path.exists(colour.PROTOTYPE_FILE):
        print(f"No {colour.PROTOTYPE_FILE}: no colours have been taught.")
        print("Run:  python setup.py --colours")
        return None
    if args.image:
        print(f"  replaying {args.image}, no camera")
        return "replay"

    cap = start_camera(args)
    if cap is None:
        print("Run probe.py to see what the camera can see, or setup.py to")
        print("set the rig up again.")
        return None

    taught = sorted(k for k in colour.load_prototypes() if k != "empty")
    exposure = "auto"
    if os.path.exists(camera.EXPOSURE_FILE):
        with open(camera.EXPOSURE_FILE) as fh:
            exposure = json.load(fh)["exposure"]
    print(f"  camera   ready, exposure pinned at {exposure}")
    print(f"  colours  {', '.join(taught)}")
    return cap


def watch(board, cap, args, stop):
    """Read the board forever, or replay one frame when there is no camera."""
    if args.image:
        frame = cv2.imread(args.image)
        while not stop.is_set():
            cells, _, note, _ = read_board(frame)
            board.offer(cells, note)
            time.sleep(1 / 30)
        return

    misses = 0
    while not stop.is_set():
        ok, frame = cap.read()
        if not ok:
            # A camera that has gone away reads false forever. Say so, rather
            # than quietly holding yesterday's pattern, and keep trying to get
            # it back.
            misses += 1
            board.offer(None, "camera returned no frame")
            if misses in (1, 25):
                print("  camera is not delivering frames")
            time.sleep(0.2)
            if misses % 25 == 0:
                cap.release()
                again = start_camera(args)
                if again is not None:
                    cap, misses = again, 0
                    print("  camera is back")
            continue
        misses = 0
        cells, _, note, _ = read_board(frame)
        for i, was, now_is in board.offer(cells, note):
            print(f"  cell {i}: {was} -> {now_is}")
    cap.release()


def handler_for(board, args, stop):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass          # the vision loop is the interesting output, not this

        def do_GET(self):
            if self.path.startswith("/events"):
                return self.stream()
            name = "index.html" if self.path in ("/", "") else self.path.lstrip("/")
            whole = os.path.join(APP, os.path.basename(name))
            if not os.path.exists(whole):
                self.send_error(404)
                return
            body = open(whole, "rb").read()
            self.send_response(200)
            self.send_header("Content-Type",
                             TYPES.get(os.path.splitext(whole)[1], "text/plain"))
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def stream(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            last = None
            try:
                while not stop.is_set():
                    now = board.snapshot(not args.no_flip)
                    # Only the pattern is worth waking the browser for. The
                    # note changes every frame and nobody reads it that fast.
                    key = (now["version"], now["blind"], now["settling"])
                    if key != last:
                        self.wfile.write(f"data: {json.dumps(now)}\n\n".encode())
                        self.wfile.flush()
                        last = key
                    time.sleep(0.05)
            except (BrokenPipeError, ConnectionResetError):
                pass
    return Handler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--image", type=str, default=None,
                    help="replay one saved frame instead of opening the camera")
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--uvc-index", type=int, default=0)
    ap.add_argument("--port", type=int, default=8099)
    ap.add_argument("--no-flip", action="store_true",
                    help="send the camera's column order instead of the "
                         "player's, which are mirror images of each other")
    args = ap.parse_args()

    cap = preflight(args)
    if cap is None:
        return 1

    stop = threading.Event()
    try:
        http = ThreadingHTTPServer(("127.0.0.1", args.port),
                                   handler_for(board := Board(), args, stop))
    except OSError as why:
        print(f"Cannot listen on port {args.port}: {why}")
        print("Something else is using it, most likely an older play.py.")
        print(f"Find it with:  lsof -nP -iTCP:{args.port} -sTCP:LISTEN")
        if cap != "replay":
            cap.release()
        return 1

    # Without this, server_close waits for the event streams to finish, and
    # an event stream by definition does not. Ctrl-c would hang.
    http.daemon_threads = True
    http.block_on_close = False

    threading.Thread(target=watch, args=(board, cap, args, stop),
                     daemon=True).start()

    print(f"\n  playing at  http://127.0.0.1:{args.port}")
    print("  ctrl-c to stop\n")
    try:
        http.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        stop.set()
        http.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
