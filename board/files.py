"""Where the tools put what they produce.

Everything here is output: frames saved with `s`, the plots and the CSV from
probe, the printable tag sheet. It is regenerated on demand, it runs to tens
of megabytes within a few sessions, and none of it describes the project.

Kept apart from the calibration, which also sits outside the repo but is not
clutter: `warp.json` and its neighbours are the rig's state, they are read on
every run, and losing them costs a setup walk rather than a rerun.
"""

import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAPTURES = os.path.join(ROOT, "captures")


def capture(name):
    """A path inside captures/, with the folder made if it is not there yet.

    Absolute paths and anything the user typed with a directory in it are
    returned untouched: a `--sheet` aimed at the desktop should land on the
    desktop, not be quietly relocated.
    """
    if os.path.isabs(name) or os.path.dirname(name):
        return name
    os.makedirs(CAPTURES, exist_ok=True)
    return os.path.join(CAPTURES, name)


def shown(path):
    """The path as it is worth printing: relative to the repo, if it is inside."""
    try:
        here = os.path.relpath(path, ROOT)
    except ValueError:
        return path
    return path if here.startswith(os.pardir) else here
