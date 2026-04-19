#!/usr/bin/env bash
# SnapVLA sync helper.
#
# Pulls the latest commit of the current branch on both the Windows inference
# host and the Raspberry Pi edge. Run from the Mac after `git push`.
#
# Usage:
#   scripts/sync.sh            # pull on both hosts
#   scripts/sync.sh pi         # pull on the Pi only
#   scripts/sync.sh windows    # pull on the Windows box only
#
# Requires:
#   - sshpass installed (brew install hudochenkov/sshpass/sshpass)
#   - SnapVLA checked out at matching paths on both hosts.

set -euo pipefail

readonly PASS='2004040316syZ#'

readonly PI_USER='yanxin'
readonly PI_HOST='raspberrypi.local'
readonly PI_PATH='~/vla_experiments'

readonly WIN_USER='29838'
readonly WIN_HOST='192.168.88.12'
readonly WIN_PATH='C:\Users\29838\Documents\Researches\vla_experiments'

branch="$(git rev-parse --abbrev-ref HEAD)"

pull_pi() {
    echo "[pi] pulling $branch on $PI_HOST..."
    sshpass -p "$PASS" ssh -o StrictHostKeyChecking=no "$PI_USER@$PI_HOST" \
        "cd $PI_PATH && git fetch origin && git checkout $branch && git pull --ff-only origin $branch"
}

pull_windows() {
    echo "[windows] pulling $branch on $WIN_HOST..."
    sshpass -p "$PASS" ssh -o StrictHostKeyChecking=no "$WIN_USER@$WIN_HOST" \
        "cd \"$WIN_PATH\" && git fetch origin && git checkout $branch && git pull --ff-only origin $branch"
}

target="${1:-all}"
case "$target" in
    pi)       pull_pi ;;
    windows)  pull_windows ;;
    all)      pull_pi; pull_windows ;;
    *)        echo "Unknown target: $target (want: pi|windows|all)" >&2; exit 1 ;;
esac

echo "done."
