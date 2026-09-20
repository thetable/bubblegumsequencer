#!/usr/bin/env python3
"""The bridge: Python says what is on the board, the browser plays it.

Runs the vision loop in a thread and serves two things over plain HTTP: the
instrument, and a stream of the pattern as it changes.

Server-sent events rather than a WebSocket, which is what HANDOVER.md section
6 assumed. The split it asks for is the point and is unchanged, but nothing
needs to travel from the browser back to the board, and one-way is what SSE
is for: no handshake, no framing, no dependency, and `EventSource` is three
lines in the browser. If the instrument ever needs to talk back, this is the
piece to swap.

Usage:
    python server.py                      # camera, then open the printed URL
    python server.py --image frame.png    # replay one frame, for working offline
"""

import argparse
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2

import cell_probe as cp
import live_view as lv
import pattern as pat

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.join(HERE, "app")
TYPES = {".html": "text/html", ".js": "text/javascript", ".css": "text/css"}


STALE = 2.0     # seconds without a frame before we stop believing the pattern


class Board:
    """The current pattern, and whatever the vision loop last had to say."""

    def __init__(self, columns=cp.COLS, rows=cp.ROWS):
        self.columns, self.rows = columns, rows
        self.stabiliser = pat.Stabiliser(columns * rows)
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
            # Frames stopping is the failure that does not announce itself: the
            # pattern is still there and still correct-looking, just from
            # whenever the camera last worked. A machine going to sleep is
            # enough to cause it, so silence past a couple of seconds counts as
            # not seeing the board.
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
            changed = self.stabiliser.update(pat.readings_from(cells, cells is not None))
            self.note = note
            if changed:
                self.version += 1
            return changed


def watch(board, args, stop):
    """Read the board forever, or replay one frame when there is no camera."""
    if args.image:
        frame = cv2.imread(args.image)
        while not stop.is_set():
            cells, _, note, _ = lv.read_board(frame)
            board.offer(cells, note)
            time.sleep(1 / 30)
        return

    cap = start_camera(args)
    if cap is None:
        stop.set()
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
        if misses:
            misses = 0
        cells, _, note, _ = lv.read_board(frame)
        for i, was, now_is in board.offer(cells, note):
            print(f"  cell {i}: {was} -> {now_is}")
    cap.release()


def start_camera(args):
    cap = cp.open_camera(args)
    if cap is None:
        return None
    if os.path.exists(cp.EXPOSURE_FILE):
        with open(cp.EXPOSURE_FILE) as fh:
            cp.set_exposure(args.uvc_index, json.load(fh)["exposure"])
    return cap


def handler_for(board, args):
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
                while True:
                    now = board.snapshot(not args.no_flip)
                    # Only the pattern is worth waking the browser for; the
                    # note changes every frame and nobody is reading it that
                    # fast. A heartbeat keeps the connection from idling out.
                    key = (now["version"], now["blind"], now["settling"])
                    if key != last:
                        self.wfile.write(
                            f"data: {json.dumps(now)}\n\n".encode())
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

    board = Board()
    stop = threading.Event()
    eyes = threading.Thread(target=watch, args=(board, args, stop), daemon=True)
    eyes.start()

    http = ThreadingHTTPServer(("127.0.0.1", args.port), handler_for(board, args))
    print(f"\n  the instrument is at  http://127.0.0.1:{args.port}\n")
    try:
        http.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
    return 0


if __name__ == "__main__":
    sys.exit(main())
