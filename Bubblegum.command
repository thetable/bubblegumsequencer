#!/bin/sh
#
# Double-click this in Finder to play. Everything else here is a terminal
# command; this is the one that does not ask you to open a terminal first.
#
# A .command rather than a .app on purpose. A .app would have to be signed
# and notarised to hold on to its camera permission across rebuilds, whereas
# this runs inside Terminal, which was granted the camera the first time you
# ever ran play.py and has kept it since.

cd "$(dirname "$0")" || exit 1

# Finder starts this with a bare PATH: no login shell has run, so nothing has
# read your profile. uv has to be looked for where its installers put it.
PATH="$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"
export PATH

stay() {
    printf '\nPress return to close this window. '
    read -r _
}

if ! command -v uv >/dev/null 2>&1; then
    echo "uv is missing. It is what builds the Python environment."
    echo
    echo "    curl -LsSf https://astral.sh/uv/install.sh | sh"
    echo
    echo "Then double-click this again."
    stay
    exit 1
fi

# The first run downloads Python and the libraries, about half a minute on a
# new machine. Every run after that finds them already there and is instant.
if ! uv sync --quiet; then
    echo
    echo "Could not build the Python environment."
    stay
    exit 1
fi

# Anything wrong with the rig is reported by play.py itself, and it stops
# rather than serving. Hold the window open so that it can be read: a Finder
# launch has nowhere else to put it.
uv run play.py --open || stay
