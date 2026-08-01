#!/usr/bin/env bash
# =============================================================================
# start_sglang.sh — Serve Inkling-Small-NVFP4 with SGLang + DSpark
#                   on dual DGX Spark (GB10) over CX7, port 30000.
#
# Wrapper around the drowzeys champion runtime (scripts/nvfp4-kv-boot.sh /
# scripts/inkling-sglang-launch.sh) from:
#   github.com/drowzeys/keys-1M-CTX-Inkling-Small-NVFP4-Dspark-NVFP4-KV-Cache-
#   SGlang-SM121-optimized-on-Two-DGX-Sparks
# The server args/env/entrypoint below match that repo verbatim; this script
# only adds the two-node plumbing (SSH, image sync, SSHFS model cache,
# health wait).
#
# Requires the KV-quant GB10 image. If missing on the head it is pulled and
# tagged automatically (override source with UPSTREAM_IMAGE); the worker is
# synced from the head via docker save/load.
#
# Usage:
#   ./start_sglang.sh              # multi-node TP=2 (default)
#   NODES=1 ./start_sglang.sh      # single-node diagnosis
#   ./stop_sglang.sh
#
# Champion runtime (env-overridable):
#   triton + fp32 reduce, marlin MoE, flashinfer_trtllm FP4 GEMM, page=1,
#   mem=0.85, ctx=1048576, max-running=16, KV fp4_mx_block16, DSpark block=7,
#   decode graphs {1..8,10,12,14,16}, prefill/piecewise graphs off
# =============================================================================

set -euo pipefail

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
MODEL="thinkingmachines/Inkling-Small-NVFP4"
DRAFT_MODEL="RadixArk/Inkling-Small-DSpark-Preview"

HEAD_IP="10.0.0.1"
WORKER_IP="10.0.0.2"
HEAD_USER="${HEAD_USER:-$(whoami)}"
WORKER_USER="${WORKER_USER:-zurih}"

SSH_KEY_NAME="id_ed25519_shared"
SSH_KEY="$HOME/.ssh/${SSH_KEY_NAME}"
WORKER_SSH_KEY="\$HOME/.ssh/${SSH_KEY_NAME}"

SGLANG_PORT="${SGLANG_PORT:-8888}"
DIST_PORT="${DIST_PORT:-25000}"
# SGLang multi-node rendezvous over CX7
HEAD_CX7_IP="10.0.22.1"
WORKER_CX7_IP="10.0.22.2"
HEAD_CX7_IF="enp1s0f1np1"
WORKER_CX7_IF="enp1s0f0np0"
HEAD_CX7_IB="rocep1s0f1"
WORKER_CX7_IB="rocep1s0f0"

# KV-quant GB10 image (drowzeys champion; fp4 KV patches baked in).
SGLANG_IMAGE="${SGLANG_IMAGE:-local/sglang-inkling:gb10-kvquant}"

HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
WORKER_MOUNT_POINT="/mnt/head-hf-cache"

HEAD_CONTAINER="inkling-sglang-head"
WORKER_CONTAINER="inkling-sglang-worker"

# 1 = single node on head only; 2 = dual Spark (default)
NODES="${NODES:-2}"

# drowzeys champion runtime (nvfp4-kv-boot.sh), env-overridable
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.85}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-16}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-1048576}"
# fp4_mx_block16: the triton-lane fp4 recipe (NOT nvfp4 — that selects the
# flashinfer/trtllm packing the triton lane cannot read). Empty = omit flag.
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp4_mx_block16}"
PAGE_SIZE="${PAGE_SIZE:-1}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-triton}"
# bf16 accumulation across KV splits perturbs logits → fewer draft matches (~+11% accept)
TRITON_REDUCE_FP32="${TRITON_REDUCE_FP32:-1}"
FP4_GEMM_BACKEND="${FP4_GEMM_BACKEND:-flashinfer_trtllm}"
MOE_RUNNER_BACKEND="${MOE_RUNNER_BACKEND:-marlin}"
DSPARK_BLOCK_SIZE="${DSPARK_BLOCK_SIZE:-7}"
ENABLE_DECODE_GRAPHS="${ENABLE_DECODE_GRAPHS:-1}"
# Explicit decode-graph list (drowzeys champion); --cuda-graph-max-bs does NOT
# filter and the default list OOMs the pool.
CUDA_GRAPH_BS="${CUDA_GRAPH_BS:-1 2 3 4 5 6 7 8 10 12 14 16}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-inkling-small}"
# Draft is 64K-adapted; cap its context independently of CTX (baked patch reads this).
INKLING_DRAFT_CTX_CAP="${INKLING_DRAFT_CTX_CAP:-65536}"
# Leave unset: compact crashes Inkling's sconv JIT (drowzeys README, Knobs).
SGLANG_RAGGED_VERIFY_MODE="${SGLANG_RAGGED_VERIFY_MODE:-}"
INKLING_TORCH_CONV_COMMIT="${INKLING_TORCH_CONV_COMMIT:-1}"
INKLING_COMMIT_STEP_BIAS="${INKLING_COMMIT_STEP_BIAS:-1}"
INKLING_NOOP_CONV_COMMIT="${INKLING_NOOP_CONV_COMMIT:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "$LOG_DIR"
HEAD_LOG="${LOG_DIR}/sglang-head.log"
WORKER_LOG="${LOG_DIR}/sglang-worker.log"

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'
info()   { echo -e "${BLUE}[INFO]${NC}  $*"; }
ok()     { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()   { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()  { echo -e "${RED}[ERROR]${NC} $*" >&2; }
header() { echo -e "\n${BOLD}${CYAN}━━━ $* ━━━${NC}\n"; }

# ─────────────────────────────────────────────────────────────────────────────
# Remote helpers
# ─────────────────────────────────────────────────────────────────────────────
worker_ssh() {
    ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no -o IdentitiesOnly=yes \
        -o ConnectTimeout=8 "${WORKER_USER}@${WORKER_IP}" -- "$@"
}

worker_docker() {
    local q=() a
    for a in "$@"; do q+=("$(printf '%q' "$a")"); done
    ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no -o IdentitiesOnly=yes \
        -o ConnectTimeout=8 "${WORKER_USER}@${WORKER_IP}" "docker ${q[*]}"
}

worker_nsenter() {
    worker_docker run --rm --privileged --pid=host --network host \
        alpine:3.20 nsenter -t 1 -m -u -i -n -- sh -c "$*"
}

# ─────────────────────────────────────────────────────────────────────────────
# Pre-flight
# ─────────────────────────────────────────────────────────────────────────────
header "SGLang + DSpark — Pre-flight"

if [ ! -f "$SSH_KEY" ]; then
    error "Shared SSH key missing: $SSH_KEY"
    exit 1
fi
ok "SSH key present"

if ! command -v docker &>/dev/null; then
    error "Docker not found on head"
    exit 1
fi
ok "Docker on head"

if [ "$NODES" -ge 2 ]; then
    if ! worker_ssh "echo ping-ok" 2>/dev/null | grep -q ping-ok; then
        error "Cannot SSH to worker ${WORKER_USER}@${WORKER_IP}"
        exit 1
    fi
    ok "SSH to worker ${WORKER_IP}"
    if ! worker_docker info &>/dev/null; then
        error "Docker over SSH to worker failed"
        exit 1
    fi
    ok "Docker on worker"
fi

HEAD_GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || true)
ok "Head GPU: ${HEAD_GPU:-unknown}"
if [ "$NODES" -ge 2 ]; then
    WORKER_GPU=$(worker_ssh "nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1" || true)
    ok "Worker GPU: ${WORKER_GPU:-unknown}"
fi

mkdir -p "$HF_CACHE"

# Remove leftover containers
for c in "$HEAD_CONTAINER"; do
    if docker ps -a --format '{{.Names}}' | grep -qx "$c"; then
        warn "Removing leftover $c on head"
        docker rm -f "$c" >/dev/null 2>&1 || true
    fi
done
if [ "$NODES" -ge 2 ]; then
    if worker_docker ps -a --format '{{.Names}}' 2>/dev/null | grep -qx "$WORKER_CONTAINER"; then
        warn "Removing leftover $WORKER_CONTAINER on worker"
        worker_docker rm -f "$WORKER_CONTAINER" >/dev/null 2>&1 || true
    fi
fi

# ─────────────────────────────────────────────────────────────────────────────
# Image
# ─────────────────────────────────────────────────────────────────────────────
header "Docker image: ${SGLANG_IMAGE}"

# Upstream prebuilt KV-quant image (drowzeys). Only ~473 MB of new layers if the
# digest-pinned lmsysorg/sglang base is already present.
UPSTREAM_IMAGE="${UPSTREAM_IMAGE:-ghcr.io/drowzeys/inkling-sglang-gb10:kvquant}"

if docker image inspect "$SGLANG_IMAGE" &>/dev/null; then
    ok "Image present on head"
else
    if [ "$SGLANG_IMAGE" = "local/sglang-inkling:gb10-kvquant" ]; then
        info "Image not found on head — pulling ${UPSTREAM_IMAGE}..."
        if ! docker pull "${UPSTREAM_IMAGE}"; then
            error "Pull failed: ${UPSTREAM_IMAGE}"
            error "Alternatively bake it: KVQUANT=1 ./scripts/bake-image.sh (from the drowzeys repo)"
            exit 1
        fi
        docker tag "${UPSTREAM_IMAGE}" "${SGLANG_IMAGE}"
        ok "Pulled and tagged as ${SGLANG_IMAGE}"
    else
        info "Image not found on head — pulling ${SGLANG_IMAGE}..."
        if ! docker pull "${SGLANG_IMAGE}"; then
            error "Pull failed: ${SGLANG_IMAGE}"
            error "Set SGLANG_IMAGE to a reachable tag, or use the default and let it pull ${UPSTREAM_IMAGE}."
            exit 1
        fi
        ok "Image pulled on head"
    fi
fi

if [ "$NODES" -ge 2 ]; then
    if worker_docker image inspect "$SGLANG_IMAGE" &>/dev/null; then
        ok "Image present on worker"
    else
        info "Syncing image to worker via docker save/load..."
        IMAGE_TAR="/tmp/sglang-image-sync-$$.tar"
        docker save "$SGLANG_IMAGE" -o "$IMAGE_TAR"
        scp -i "$SSH_KEY" -o StrictHostKeyChecking=no -o IdentitiesOnly=yes \
            "$IMAGE_TAR" "${WORKER_USER}@${WORKER_IP}:/tmp/sglang-image-sync.tar"
        worker_ssh "docker load -i /tmp/sglang-image-sync.tar && rm -f /tmp/sglang-image-sync.tar"
        rm -f "$IMAGE_TAR"
        ok "Image on worker"
    fi
fi

# ─────────────────────────────────────────────────────────────────────────────
# Model cache (head disk; worker via SSHFS when multi-node)
# ─────────────────────────────────────────────────────────────────────────────
header "Model cache"

# Prefer already-downloaded snapshots under HF_CACHE
MODEL_REPO_DIRNAME="models--thinkingmachines--Inkling-Small-NVFP4"
DRAFT_REPO_DIRNAME="models--RadixArk--Inkling-Small-DSpark-Preview"
resolve_snapshot() {
    local dirname="$1" d hash snap
    for d in "$HF_CACHE/hub/$dirname" "$HF_CACHE/$dirname"; do
        [ -d "$d" ] || continue
        if [ -f "$d/refs/main" ]; then
            hash=$(tr -d ' \n' <"$d/refs/main")
            snap="$d/snapshots/$hash"
            if [ -f "$snap/config.json" ]; then
                echo "$snap"
                return 0
            fi
        fi
        for snap in "$d"/snapshots/*/; do
            if [ -f "${snap}config.json" ]; then
                echo "${snap%/}"
                return 0
            fi
        done
    done
    return 1
}

# Map a host snapshot path under HF_CACHE into the container mount.
container_path() {
    local p="$1" mapped
    mapped="${p/#$HF_CACHE//root/.cache/huggingface}"
    if [ "$mapped" = "$p" ]; then
        error "Path is not under HF_CACHE; cannot map into container."
        error "  path=$p"
        error "  HF_CACHE=$HF_CACHE"
        exit 1
    fi
    echo "$mapped"
}

MODEL_PATH=$(resolve_snapshot "$MODEL_REPO_DIRNAME" || true)
if [ -n "${MODEL_PATH:-}" ]; then
    ok "Model cached: $MODEL_PATH"
    # IMPORTANT: weights live at $HF_CACHE/models--... (not $HF_CACHE/hub/...).
    # HF repo-id lookup defaults to $HF_HOME/hub/models--... and would miss them.
    # Pass the absolute snapshot path, remapped into the container mount.
    CONTAINER_MODEL=$(container_path "$MODEL_PATH")
    info "Container model path: $CONTAINER_MODEL"
else
    warn "Model not found under HF_CACHE — SGLang will try HF id ${MODEL}"
    warn "Expected e.g. $HF_CACHE/models--thinkingmachines--Inkling-Small-NVFP4/snapshots/<hash>"
    CONTAINER_MODEL="$MODEL"
fi

# Draft: champion runs from a local dir with HF offline; use the local snapshot
# when present, else fall back to the repo id (downloaded into HF cache on first run).
DRAFT_PATH=$(resolve_snapshot "$DRAFT_REPO_DIRNAME" || true)
if [ -n "${DRAFT_PATH:-}" ]; then
    ok "Draft cached: $DRAFT_PATH"
    CONTAINER_DRAFT=$(container_path "$DRAFT_PATH")
    info "Container draft path: $CONTAINER_DRAFT"
else
    warn "Draft not found under HF_CACHE — SGLang will try HF id ${DRAFT_MODEL}"
    CONTAINER_DRAFT="$DRAFT_MODEL"
fi

if [ "$NODES" -ge 2 ]; then
    # Ensure worker can SSH back for SSHFS
    if [ -f "${SSH_KEY}.pub" ]; then
        PUB=$(cat "${SSH_KEY}.pub")
        mkdir -p "$HOME/.ssh"; touch "$HOME/.ssh/authorized_keys"; chmod 600 "$HOME/.ssh/authorized_keys"
        grep -qF "$PUB" "$HOME/.ssh/authorized_keys" 2>/dev/null || echo "$PUB" >>"$HOME/.ssh/authorized_keys"
    fi

    force_unmount_worker_cache() {
        worker_ssh "fusermount -uz ${WORKER_MOUNT_POINT} 2>/dev/null || umount -l ${WORKER_MOUNT_POINT} 2>/dev/null || true" || true
        worker_nsenter "
            umount -l ${WORKER_MOUNT_POINT} 2>/dev/null || true
            umount -f ${WORKER_MOUNT_POINT} 2>/dev/null || true
            fusermount -uz ${WORKER_MOUNT_POINT} 2>/dev/null || true
            if ! mountpoint -q ${WORKER_MOUNT_POINT} 2>/dev/null; then
                rm -rf ${WORKER_MOUNT_POINT} 2>/dev/null || true
            fi
            mkdir -p ${WORKER_MOUNT_POINT}
            chown ${WORKER_USER}:${WORKER_USER} ${WORKER_MOUNT_POINT}
            chmod 755 ${WORKER_MOUNT_POINT}
        " >/dev/null 2>&1 || true
        sleep 1
    }

    if ! worker_ssh "command -v sshfs >/dev/null"; then
        error "sshfs missing on worker"
        exit 1
    fi

    # enable user_allow_other if needed
    if ! worker_ssh "grep -qE '^[[:space:]]*user_allow_other' /etc/fuse.conf 2>/dev/null"; then
        info "Enabling user_allow_other on worker..."
        worker_nsenter "grep -qE '^[[:space:]]*user_allow_other' /etc/fuse.conf || echo user_allow_other >> /etc/fuse.conf" || true
    fi

    if worker_ssh "mountpoint -q ${WORKER_MOUNT_POINT} && ls ${WORKER_MOUNT_POINT} >/dev/null 2>&1"; then
        ok "SSHFS already mounted at ${WORKER_MOUNT_POINT}"
    else
        force_unmount_worker_cache
        info "Mounting head HF cache on worker over CX7..."
        if ! worker_ssh "sshfs -o StrictHostKeyChecking=no,IdentitiesOnly=yes,IdentityFile=${WORKER_SSH_KEY},allow_other,default_permissions,reconnect,ServerAliveInterval=15,ServerAliveCountMax=3,cache=yes,kernel_cache \
            ${HEAD_USER}@${HEAD_CX7_IP}:${HF_CACHE} ${WORKER_MOUNT_POINT}" 2>&1; then
            warn "CX7 mount failed; trying management IP..."
            worker_ssh "sshfs -o StrictHostKeyChecking=no,IdentitiesOnly=yes,IdentityFile=${WORKER_SSH_KEY},allow_other,default_permissions,reconnect,ServerAliveInterval=15 \
                ${HEAD_USER}@${HEAD_IP}:${HF_CACHE} ${WORKER_MOUNT_POINT}" 2>&1
        fi
        sleep 1
        if ! worker_ssh "ls ${WORKER_MOUNT_POINT} >/dev/null 2>&1"; then
            error "SSHFS mount failed on worker"
            exit 1
        fi
        ok "Worker mounts head HF cache at ${WORKER_MOUNT_POINT}"
    fi

    # Hard-check: if we have a local snapshot, worker must see the same config
    if [ -n "${MODEL_PATH:-}" ]; then
        SNAPSHOT_REL="${MODEL_PATH#"${HF_CACHE}/"}"
        WORKER_CFG="${WORKER_MOUNT_POINT}/${SNAPSHOT_REL}/config.json"
        if ! worker_ssh "test -f '${WORKER_CFG}'"; then
            error "Worker cannot see model via SSHFS: ${WORKER_CFG}"
            worker_ssh "ls -la ${WORKER_MOUNT_POINT} 2>&1 | head -20" || true
            exit 1
        fi
        ok "Worker sees model config via head storage (SSHFS)"
        # Docker must be able to bind-mount and read the path too
        info "Verifying Docker on worker can read SSHFS model path..."
        if ! worker_docker run --rm \
            -v "${WORKER_MOUNT_POINT}:/root/.cache/huggingface" \
            "${SGLANG_IMAGE}" \
            test -f "${CONTAINER_MODEL}/config.json" 2>/dev/null; then
            error "Docker on worker cannot read model at ${CONTAINER_MODEL}"
            error "Often means FUSE lack allow_other / dead mount."
            exit 1
        fi
        ok "Docker on worker can bind-mount head model weights"
    fi
fi

# ─────────────────────────────────────────────────────────────────────────────
# Launch args — drowzeys champion invocation (scripts/nvfp4-kv-boot.sh /
# scripts/inkling-sglang-launch.sh), verbatim, plus multi-node when NODES=2.
# ─────────────────────────────────────────────────────────────────────────────

if [ "$NODES" -ge 2 ]; then
    TP_SIZE=2
    NNODES=2
else
    TP_SIZE=1
    NNODES=1
fi

# Champion server args (GB10). ATTENTION/FP4/MOE/KV already defaulted above.
SGLANG_COMMON_ARGS=(
    --model-path "${CONTAINER_MODEL}"
    --trust-remote-code
    --served-model-name "${SERVED_MODEL_NAME}"
    --tp-size "${TP_SIZE}"
    --context-length "${CONTEXT_LENGTH}"
    --quantization modelopt_fp4
    --attention-backend "${ATTENTION_BACKEND}"
    # page-size 1: page 128 corrupts the triton DSpark verify path
    --page-size "${PAGE_SIZE}"
    --fp4-gemm-backend "${FP4_GEMM_BACKEND}"
    --moe-runner-backend "${MOE_RUNNER_BACKEND}"
    --mamba-radix-cache-strategy extra_buffer
    --mem-fraction-static "${MEM_FRACTION_STATIC}"
    --swa-full-tokens-ratio 0.1
    --mamba-full-memory-ratio 0.1
    --max-running-requests "${MAX_RUNNING_REQUESTS}"
    --chunked-prefill-size 8192
    --reasoning-parser inkling
    --tool-call-parser inkling
    --skip-server-warmup
    --disable-flashinfer-autotune
    --stream-interval 32
    # DSpark: block-size 7 (measured optimum; 15 is worse); draft unquantized
    --speculative-algorithm DSPARK
    --speculative-draft-model-path "${CONTAINER_DRAFT}"
    --speculative-draft-model-quantization unquant
    --speculative-dspark-block-size "${DSPARK_BLOCK_SIZE}"
    # triton backend cannot replay EXTEND; sm_121 piecewise compiler hard-fails
    --disable-prefill-cuda-graph
    --disable-piecewise-cuda-graph
)
# bf16 accumulation across KV splits perturbs logits → fewer draft matches
if [ "${TRITON_REDUCE_FP32}" = "1" ]; then
    SGLANG_COMMON_ARGS+=(--triton-attention-reduce-in-fp32)
fi
if [ -n "${KV_CACHE_DTYPE}" ]; then
    SGLANG_COMMON_ARGS+=(--kv-cache-dtype "${KV_CACHE_DTYPE}")
fi
if [ "${ENABLE_DECODE_GRAPHS}" = "1" ]; then
    # shellcheck disable=SC2086
    SGLANG_COMMON_ARGS+=(--cuda-graph-bs ${CUDA_GRAPH_BS})
else
    SGLANG_COMMON_ARGS+=(--cuda-graph-backend-decode=disabled)
fi

# Champion container env (NCCL over CX7 RoCEv2 + conv-commit fix + arch pins).
NCCL_ENV=(
    -e NCCL_IB_DISABLE=0
    -e NCCL_IB_ROCE_VERSION_NUM=2
    -e NCCL_IB_GID_INDEX=3
    -e NCCL_NET=IB
    -e NCCL_NET_PLUGIN=none
    -e NCCL_NVLS_ENABLE=0
    -e NCCL_CUMEM_ENABLE=0
    -e NCCL_CROSS_NIC=0
    -e NCCL_IGNORE_CPU_AFFINITY=1
    -e NCCL_DEBUG=WARN
    -e SGLANG_ENABLE_UNIFIED_RADIX_TREE=1
    -e TORCH_CUDA_ARCH_LIST=12.1a
    -e FLASHINFER_CUDA_ARCH_LIST=12.1a
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    # DSpark conv-state commit fix (baked inkling.py; required multi-node)
    -e INKLING_TORCH_CONV_COMMIT="${INKLING_TORCH_CONV_COMMIT:-1}"
    -e INKLING_COMMIT_STEP_BIAS="${INKLING_COMMIT_STEP_BIAS:-1}"
    -e INKLING_NOOP_CONV_COMMIT="${INKLING_NOOP_CONV_COMMIT:-0}"
    -e INKLING_DRAFT_CTX_CAP="${INKLING_DRAFT_CTX_CAP}"
    -e HF_HOME=/root/.cache/huggingface
    -e HF_HUB_CACHE=/root/.cache/huggingface
    -e HUGGINGFACE_HUB_CACHE=/root/.cache/huggingface
    -e TRANSFORMERS_CACHE=/root/.cache/huggingface
    -e PYTHONWARNINGS=ignore::SyntaxWarning
)
# Champion runs fully offline from local weights — only when BOTH snapshots are local.
if [ -n "${MODEL_PATH:-}" ] && [ -n "${DRAFT_PATH:-}" ]; then
    NCCL_ENV+=(-e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1)
fi
# Only forward RAGGED when explicitly set (empty = upstream default; compact crashes Inkling).
if [ -n "${SGLANG_RAGGED_VERIFY_MODE}" ]; then
    NCCL_ENV+=(-e "SGLANG_RAGGED_VERIFY_MODE=${SGLANG_RAGGED_VERIFY_MODE}")
fi

DOCKER_BASE=(
    --network host
    --gpus all
    --shm-size 16g
    --ipc host
    --privileged
    --device /dev/infiniband
    --cap-add IPC_LOCK
    --ulimit memlock=-1
    --ulimit stack=67108864
    --entrypoint python3
    "${NCCL_ENV[@]}"
)

# Champion entrypoint (repo verbatim): the kvquant image ships ENTRYPOINT "true",
# so --entrypoint python3 is mandatory, then -m sglang.launch_server as argv.
SGLANG_ENTRY=(-m sglang.launch_server)

header "Launch SGLang (DSpark champion) — NODES=${NNODES} TP=${TP_SIZE} image=${SGLANG_IMAGE}"

: >"$HEAD_LOG"
: >"$WORKER_LOG"

# ── Worker (rank 1) first when multi-node ──
if [ "$NNODES" -ge 2 ]; then
    info "Starting worker (node-rank 1, multi-node peer)..."
    info "  volume: ${WORKER_MOUNT_POINT} → /root/.cache/huggingface"
    info "  image (baked patches): ${SGLANG_IMAGE}"
    worker_docker run -d \
        --name "${WORKER_CONTAINER}" \
        "${DOCKER_BASE[@]}" \
        -e NCCL_SOCKET_IFNAME="${WORKER_CX7_IF}" \
        -e GLOO_SOCKET_IFNAME="${WORKER_CX7_IF}" \
        -e TP_SOCKET_IFNAME="${WORKER_CX7_IF}" \
        -e NCCL_IB_HCA="${WORKER_CX7_IB}" \
        -v "${WORKER_MOUNT_POINT}:/root/.cache/huggingface" \
        "${SGLANG_IMAGE}" \
        "${SGLANG_ENTRY[@]}" \
        "${SGLANG_COMMON_ARGS[@]}" \
        --nnodes "${NNODES}" \
        --node-rank 1 \
        --dist-init-addr "${HEAD_CX7_IP}:${DIST_PORT}" \
        --host 0.0.0.0 \
        --port "${SGLANG_PORT}"

    sleep 2
    if ! worker_docker ps --format '{{.Names}}' | grep -qx "${WORKER_CONTAINER}"; then
        error "Worker container exited immediately"
        worker_docker logs "${WORKER_CONTAINER}" 2>&1 | tee -a "$WORKER_LOG" | tail -80
        exit 1
    fi
    ok "Worker started"
    worker_docker logs -f "${WORKER_CONTAINER}" >>"$WORKER_LOG" 2>&1 &
    WORKER_LOG_PID=$!
    # Give worker a moment to bind before head starts rendezvous
    sleep 5
fi

# ── Head (rank 0 + API) ──
info "Starting head (node-rank 0, API on :${SGLANG_PORT})..."
info "  volume: ${HF_CACHE} → /root/.cache/huggingface"
info "  image (baked patches): ${SGLANG_IMAGE}"
info "  draft: ${CONTAINER_DRAFT}"

HEAD_EXTRA=()
if [ "$NNODES" -ge 2 ]; then
    HEAD_EXTRA+=(
        --nnodes "${NNODES}"
        --node-rank 0
        --dist-init-addr "${HEAD_CX7_IP}:${DIST_PORT}"
    )
fi

docker run -d \
    --name "${HEAD_CONTAINER}" \
    "${DOCKER_BASE[@]}" \
    -e NCCL_SOCKET_IFNAME="${HEAD_CX7_IF}" \
    -e GLOO_SOCKET_IFNAME="${HEAD_CX7_IF}" \
    -e TP_SOCKET_IFNAME="${HEAD_CX7_IF}" \
    -e NCCL_IB_HCA="${HEAD_CX7_IB}" \
    -v "${HF_CACHE}:/root/.cache/huggingface" \
    "${SGLANG_IMAGE}" \
    "${SGLANG_ENTRY[@]}" \
    "${SGLANG_COMMON_ARGS[@]}" \
    "${HEAD_EXTRA[@]}" \
    --host 0.0.0.0 \
    --port "${SGLANG_PORT}"

sleep 2
if ! docker ps --format '{{.Names}}' | grep -qx "${HEAD_CONTAINER}"; then
    error "Head container exited immediately"
    docker logs "${HEAD_CONTAINER}" 2>&1 | tee -a "$HEAD_LOG" | tail -80
    [ "$NNODES" -ge 2 ] && worker_docker logs "${WORKER_CONTAINER}" 2>&1 | tee -a "$WORKER_LOG" | tail -40 || true
    exit 1
fi
ok "Head started"
docker logs -f "${HEAD_CONTAINER}" >>"$HEAD_LOG" 2>&1 &
HEAD_LOG_PID=$!

# ─────────────────────────────────────────────────────────────────────────────
# Wait for health
# ─────────────────────────────────────────────────────────────────────────────
header "Waiting for SGLang API"

API_URL="http://${HEAD_IP}:${SGLANG_PORT}/v1/models"
# Also stream to console
docker logs -f "${HEAD_CONTAINER}" 2>&1 &
TAIL_PID=$!
cleanup_tail() { kill "$TAIL_PID" 2>/dev/null || true; kill "${HEAD_LOG_PID:-}" 2>/dev/null || true; kill "${WORKER_LOG_PID:-}" 2>/dev/null || true; }
trap cleanup_tail EXIT INT TERM

MAX_POLLS=240   # 20 min (draft model download + weight load)
POLL=0
READY=false
while [ $POLL -lt $MAX_POLLS ]; do
    POLL=$((POLL + 1))
    if curl -sf "$API_URL" >/dev/null 2>&1; then
        READY=true
        break
    fi
    if ! docker ps --format '{{.Names}}' | grep -qx "${HEAD_CONTAINER}"; then
        echo ""
        error "Head container died"
        docker logs "${HEAD_CONTAINER}" 2>&1 | tee -a "$HEAD_LOG" | tail -100
        [ "$NNODES" -ge 2 ] && worker_docker logs "${WORKER_CONTAINER}" 2>&1 | tee -a "$WORKER_LOG" | tail -60 || true
        exit 1
    fi
    if [ "$NNODES" -ge 2 ] && ! worker_docker ps --format '{{.Names}}' | grep -qx "${WORKER_CONTAINER}"; then
        echo ""
        error "Worker container died"
        worker_docker logs "${WORKER_CONTAINER}" 2>&1 | tee -a "$WORKER_LOG" | tail -100
        docker logs "${HEAD_CONTAINER}" 2>&1 | tee -a "$HEAD_LOG" | tail -40
        exit 1
    fi
    [ $((POLL % 2)) -eq 0 ] && echo -n "."
    sleep 5
done

cleanup_tail
trap - EXIT INT TERM

if [ "$READY" != true ]; then
    error "Timed out waiting for API at $API_URL"
    docker logs "${HEAD_CONTAINER}" 2>&1 | tee -a "$HEAD_LOG" | tail -80
    exit 1
fi

echo ""
echo -e "${BOLD}${GREEN}"
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║           SGLang + DSpark is up and running!                ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"
echo -e "  ${BOLD}Model:${NC}      ${MODEL}  (served as \"${SERVED_MODEL_NAME}\")"
echo -e "  ${BOLD}Draft:${NC}      ${DRAFT_MODEL}  (DSPARK)"
echo -e "  ${BOLD}Endpoint:${NC}   http://${HEAD_IP}:${SGLANG_PORT}/v1"
echo -e "  ${BOLD}Parallel:${NC}   TP=${TP_SIZE}  nnodes=${NNODES}"
echo -e "  ${BOLD}Attention:${NC}  ${ATTENTION_BACKEND}  (fp32 reduce: ${TRITON_REDUCE_FP32})"
echo -e "  ${BOLD}FP4/MoE:${NC}    ${FP4_GEMM_BACKEND} + ${MOE_RUNNER_BACKEND}"
echo -e "  ${BOLD}KV cache:${NC}   ${KV_CACHE_DTYPE:-auto}"
echo -e "  ${BOLD}CUDA graph:${NC} decode bs={${CUDA_GRAPH_BS// /,}} prefill=off (ENABLE_DECODE_GRAPHS=${ENABLE_DECODE_GRAPHS})"
echo -e "  ${BOLD}Image notes:${NC} drowzeys KV-quant GB10 image (patches baked in)"
echo -e "  ${BOLD}Context:${NC}    ${CONTEXT_LENGTH}"
echo -e "  ${BOLD}Concurrent:${NC} ${MAX_RUNNING_REQUESTS}"
echo -e "  ${BOLD}DSpark:${NC}     block=${DSPARK_BLOCK_SIZE}  draft=${DRAFT_MODEL}"
echo -e "  ${BOLD}mem-frac:${NC}   ${MEM_FRACTION_STATIC}  page-size=${PAGE_SIZE}"
echo -e "  ${BOLD}Image:${NC}      ${SGLANG_IMAGE}"
echo -e "  ${BOLD}Logs:${NC}       ${HEAD_LOG}"
[ "$NNODES" -ge 2 ] && echo -e "              ${WORKER_LOG}"
echo ""
echo -e "  ${BOLD}Quick test:${NC}"
echo "    curl -s http://${HEAD_IP}:${SGLANG_PORT}/v1/models | head"
echo "    curl -s http://${HEAD_IP}:${SGLANG_PORT}/v1/completions -H 'Content-Type: application/json' \\"
echo "      -d '{\"model\": \"${SERVED_MODEL_NAME}\", \"prompt\": \"The capital of France is\", \"max_tokens\": 12, \"temperature\": 0}'"
echo "    # expect byte-exact: \" Paris. The capital of Germany is Berlin. The capital of\""
echo ""
echo -e "  ${BOLD}Stop:${NC}  ./stop_sglang.sh"
echo ""
