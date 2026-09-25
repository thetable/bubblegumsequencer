#!/bin/sh
#
# Push the site to backin.de/gumball.
#
#   ./website/deploy.sh             upload whatever has changed
#   ./website/deploy.sh --dry-run   show what it would upload, touch nothing
#
# FTPS rather than SFTP, because the server accepts SSH keys only and there
# is no key on it yet. See the note at the bottom of this file.
#
# Nothing is ever deleted on the server, deliberately. That directory holds
# more than this site: the 2008 archive under /2008, the course paper and the
# De:Bug PDFs, and the original images folder. This page links to those, and
# so do fifteen years of other people's blogs. `mirror -R` without --delete
# only ever adds and updates, so all of it survives untouched.

set -e

HOST=web2.deinprovider.net
USER=web550
REMOTE=/htdocs/backin.de/gumball
LOCAL=$(cd "$(dirname "$0")" && pwd)

# Where the password lives in 1Password. Referenced by item ID rather than
# title, so renaming the entry in 1Password does not break the deploy.
#   Private / "backin.de ftp" / password
OP_REF=${OP_REF:-"op://Private/42lv5gll6jfbpotzrdaalrolau/password"}

DRY=""
case "$1" in
  --dry-run|-n) DRY="--dry-run" ;;
  "") ;;
  *) echo "usage: $(basename "$0") [--dry-run]" >&2; exit 2 ;;
esac

if ! command -v lftp >/dev/null 2>&1; then
  echo "lftp is not installed.  brew install lftp" >&2
  exit 1
fi

# The password comes from 1Password if it can, and is typed if it cannot.
# It is passed through the environment rather than on the command line, so it
# never appears in `ps` and never touches the disk. Nothing here prints it.
# The password comes from 1Password if it can, and is typed if it cannot.
# It goes through the environment rather than the command line, so it never
# shows up in `ps` and never touches the disk. Nothing here prints it.
#
# Gated on op read actually succeeding rather than on `op whoami`, which
# reports "not signed in" in shells where reading a secret works perfectly
# well. Try the thing you need; do not ask whether it will work.
AUTH="-u $USER"
if command -v op >/dev/null 2>&1 \
   && LFTP_PASSWORD=$(op read "$OP_REF" 2>/dev/null) \
   && [ -n "$LFTP_PASSWORD" ]; then
  export LFTP_PASSWORD
  AUTH="-u $USER --env-password"
  echo "  auth  1Password"
else
  unset LFTP_PASSWORD
  echo "  auth  typed"
  command -v op >/dev/null 2>&1 && \
    echo "        ($OP_REF did not resolve; op item list | grep -i backin)"
fi

echo "  from  $LOCAL"
echo "  to    ftps://$HOST$REMOTE"
[ -n "$DRY" ] && echo "  (dry run, nothing will be written)"
echo

# --only-newer keeps repeat runs quick; the exclusions keep the deploy to the
# site itself rather than the tooling that builds it.
# One lftp session: show the target first, then mirror into it. If the path
# is wrong the listing fails and nothing is uploaded.
lftp $AUTH "ftp://$HOST" -e "
set ftp:ssl-force true;
set ftp:ssl-protect-data true;
set ssl:verify-certificate yes;
set net:max-retries 2;
set net:timeout 20;
set cmd:fail-exit yes;
echo '  --- what is there now ---';
cls -l '$REMOTE';
echo '  --- uploading ---';
mirror -R --only-newer --verbose $DRY \
  --exclude-glob deploy.sh \
  --exclude-glob .DS_Store \
  '$LOCAL' '$REMOTE';
quit"

echo
[ -n "$DRY" ] && echo "Dry run finished. Re-run without --dry-run to upload." \
             || echo "Done.  http://backin.de/gumball/"

# ---------------------------------------------------------------------------
# To move to SFTP, which would be better: the box runs OpenSSH on port 22 and
# already refuses passwords, so it only needs your public key in the account's
# authorized_keys. Ask the provider to add ~/.ssh/id_ed25519.pub, or look for
# an SSH key field in their control panel. After that this whole script
# collapses to:
#
#   rsync -av --exclude deploy.sh ./website/ web550@web2.deinprovider.net:/htdocs/backin.de/gumball/
#
# rsync sends only the changed parts of changed files, needs no password once
# the key is in place, and still deletes nothing unless you ask it to.
