#!/usr/bin/env bash
# =============================================================================
# stop_sglang.sh — Stop the Inkling SGLang multi-node stack
#                  (head + worker containers, optional SSHFS unmount).
#
# Usage:
#   ./stop_sglang.sh              # stop containers
#   ./stop_sglang.sh --unmount    # also unmount head HF cache on the worker
# =============================================================================

set -euo pipefail

WORKER_IP="${WORKER_IP:-10.0.0.2}"
WORKER_USER="${WORKER_USER:-zurih}"
SSH_KEY_NAME="id_ed25519_shared"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/${SSH_KEY_NAME}}"

HEAD_CONTAINER="inkling-sglang-head"
WORKER_CONTAINER="inkling-sglang-worker"
WORKER_MOUNT_POINT="/mnt/head-hf-cache"

UNMOUNT=true
for arg in "$@"; do
    case "$arg" in
        --unmount|-u) UNMOUNT=true ;;
        --no-unmount|-n) UNMOUNT=false ;;
        -h|--help)
            echo "Usage: $0 [--no-unmount]"
            echo "  Stop head + worker SGLang containers and unmount SSHFS."
            echo "  --no-unmount  Skip unmounting ${WORKER_MOUNT_POINT} on the worker."
            exit 0
            ;;
        *)
            echo "Unknown option: $arg" >&2
            echo "Usage: $0 [--no-unmount]" >&2
            exit 1
            ;;
    esac
done

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'

info()  { echo -e "${BLUE}[INFO]${NC}  $*"; }
ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*" >&2; }

worker_ssh() {
    ssh -i "$SSH_KEY" \
        -o StrictHostKeyChecking=no \
        -o IdentitiesOnly=yes \
        -o ConnectTimeout=8 \
        "${WORKER_USER}@${WORKER_IP}" -- "$@"
}

worker_docker() {
    local q=() a
    for a in "$@"; do
        q+=("$(printf '%q' "$a")")
    done
    ssh -i "$SSH_KEY" \
        -o StrictHostKeyChecking=no \
        -o IdentitiesOnly=yes \
        -o ConnectTimeout=8 \
        "${WORKER_USER}@${WORKER_IP}" \
        "docker ${q[*]}"
}

worker_nsenter() {
    worker_docker run --rm --privileged --pid=host --network host \
        alpine:3.20 \
        nsenter -t 1 -m -u -i -n -- \
        sh -c "$*"
}

echo -e "${BOLD}Stopping SGLang stack…${NC}"
echo ""

# ── Head ─────────────────────────────────────────────────────────────────────
if docker ps -a --format '{{.Names}}' 2>/dev/null | grep -qx "${HEAD_CONTAINER}"; then
    info "Removing head container (${HEAD_CONTAINER})..."
    docker rm -f "${HEAD_CONTAINER}" >/dev/null 2>&1 || true
    ok "Head container removed"
else
    warn "Head container '${HEAD_CONTAINER}' not found"
fi

# ── Worker ───────────────────────────────────────────────────────────────────
if [ ! -f "$SSH_KEY" ]; then
    error "Shared SSH key not found: ${SSH_KEY} — cannot reach worker"
    error "Head was stopped; stop the worker manually if needed:"
    echo "  ssh ${WORKER_USER}@${WORKER_IP} docker rm -f ${WORKER_CONTAINER}"
    exit 1
fi

if worker_ssh "echo ping-ok" 2>/dev/null | grep -q ping-ok; then
    if worker_docker ps -a --format '{{.Names}}' 2>/dev/null | grep -qx "${WORKER_CONTAINER}"; then
        info "Removing worker container (${WORKER_CONTAINER})..."
        worker_docker rm -f "${WORKER_CONTAINER}" >/dev/null 2>&1 || true
        ok "Worker container removed"
    else
        warn "Worker container '${WORKER_CONTAINER}' not found"
    fi

    if [ "$UNMOUNT" = true ]; then
        info "Unmounting ${WORKER_MOUNT_POINT} on worker..."
        worker_ssh "fusermount -uz ${WORKER_MOUNT_POINT} 2>/dev/null || umount -l ${WORKER_MOUNT_POINT} 2>/dev/null || true" || true
        worker_nsenter "
            umount -l ${WORKER_MOUNT_POINT} 2>/dev/null || true
            umount -f ${WORKER_MOUNT_POINT} 2>/dev/null || true
            fusermount -uz ${WORKER_MOUNT_POINT} 2>/dev/null || true
            if ! mountpoint -q ${WORKER_MOUNT_POINT} 2>/dev/null; then
                rm -rf ${WORKER_MOUNT_POINT} 2>/dev/null || true
                mkdir -p ${WORKER_MOUNT_POINT}
                chown ${WORKER_USER}:${WORKER_USER} ${WORKER_MOUNT_POINT} 2>/dev/null || true
            fi
        " >/dev/null 2>&1 || true
        if worker_ssh "ls ${WORKER_MOUNT_POINT} >/dev/null 2>&1" \
            && ! worker_ssh "mountpoint -q ${WORKER_MOUNT_POINT}" 2>/dev/null; then
            ok "SSHFS unmounted (clean dir at ${WORKER_MOUNT_POINT})"
        elif worker_ssh "mountpoint -q ${WORKER_MOUNT_POINT}" 2>/dev/null; then
            warn "Mount may still be active at ${WORKER_MOUNT_POINT}"
        else
            ok "Mount cleared"
        fi
    fi
else
    error "Cannot SSH to worker ${WORKER_USER}@${WORKER_IP}"
    error "Head was stopped; stop the worker manually if needed:"
    echo "  ssh -i ${SSH_KEY} ${WORKER_USER}@${WORKER_IP} docker rm -f ${WORKER_CONTAINER}"
    exit 1
fi

echo ""
ok "SGLang stack stopped"
echo "  Restart with: ./start_sglang.sh"
if [ "$UNMOUNT" != true ]; then
    echo "  (HF cache SSHFS left mounted; use ./stop_sglang.sh to unmount, or ./stop_sglang.sh --no-unmount next time)"
fi
echo ""
