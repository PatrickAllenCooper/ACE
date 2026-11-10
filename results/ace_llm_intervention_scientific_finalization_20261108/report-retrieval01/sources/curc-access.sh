#!/usr/bin/env bash
# Shared transport only. No job submissions, credentials or project mutations.
set -euo pipefail

CURC_HOST=paco0228@login.rc.colorado.edu
CURC_CONTROL=${CURC_SOCKET:-/tmp/defab-curc.sock}

check_master() {
  ssh -S "$CURC_CONTROL" -o BatchMode=yes -o ConnectTimeout=10 \
    -O check "$CURC_HOST"
}

case "${1:-}" in
  check)
    check_master
    ;;
  connect)
    if check_master 2>/dev/null; then
      printf 'Shared CURC master is already available at %s\n' "$CURC_CONTROL"
      exit 0
    fi
    # A heredoc can occupy stdin, so OpenSSH should read the user's /dev/tty.
    if ! ( : </dev/tty ) 2>/dev/null; then
      printf 'Run connect in your local interactive terminal for password and Duo.\n' >&2
      exit 2
    fi
    ssh -M -S "$CURC_CONTROL" -o ControlPersist=8h \
      -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
      -o ConnectTimeout=20 -o NumberOfPasswordPrompts=1 -fN "$CURC_HOST"
    check_master
    ;;
  run)
    shift
    if [[ $# -eq 0 ]]; then
      printf 'Usage: curc-access.sh run <quoted remote command>\n' >&2
      exit 2
    fi
    if ! check_master >/dev/null 2>&1; then
      printf 'No shared CURC master. Ask Patrick to connect and approve Duo.\n' >&2
      exit 3
    fi
    exec ssh -S "$CURC_CONTROL" -o ControlMaster=no -o BatchMode=yes \
      -o PasswordAuthentication=no -o KbdInteractiveAuthentication=no \
      -o PubkeyAuthentication=no -o ConnectTimeout=10 \
      -o ServerAliveInterval=30 -o ServerAliveCountMax=3 "$CURC_HOST" "$@"
    ;;
  *)
    printf 'Usage: curc-access.sh {check|connect|run <remote command>}\n' >&2
    exit 2
    ;;
esac
