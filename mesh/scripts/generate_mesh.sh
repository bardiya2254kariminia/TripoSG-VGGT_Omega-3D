#!/usr/bin/env bash
# =============================================================================
# generate_mesh.sh — batch mesh generation (TripoSG mesh → MV-Adapter texture)
#
# Processes every image in INPUT_DIR and writes outputs to OUTPUT_DIR.
#
# Configurable via environment variables (all optional):
#
#   INPUT_DIR            Directory of input images   (default: <repo>/images)
#   OUTPUT_DIR           Directory for outputs       (default: <repo>/mesh/generated_meshes)
#   MESH_BACKBONE        "triposg" (default) or "hunyuan"
#   TEXTURE_BACKEND      "mvadapter" (default), "hunyuan", or "none"
#   NO_BG_REMOVAL        Set to "1" to skip BEN2 background removal
#
#   # Hunyuan mesh backbone options (only used when MESH_BACKBONE=hunyuan):
#   HUNYUAN_MODEL        Hunyuan3D-2 HF model ID     (default: tencent/Hunyuan3D-2)
#
#   # TripoSG mesh backbone options (only used when MESH_BACKBONE=triposg):
#   TRIPOSG_MODEL_PATH   Local TripoSG weights dir   (auto-downloaded if unset)
#   TRIPOSG_STEPS        Denoising steps             (default: 50)
#   TRIPOSG_GUIDANCE     Guidance scale              (default: 7.5)
#   TRIPOSG_SEED         RNG seed                    (default: 42)
#
#   # MV-Adapter texturing options (only used when TEXTURE_BACKEND=mvadapter):
#   MVADAPTER_VARIANT    "sdxl" (default) or "sd21"
#   MVADAPTER_STEPS      Denoising steps             (default: 50)
#   MVADAPTER_GUIDANCE   Guidance scale              (default: 3.0)
#   MVADAPTER_SEED       RNG seed (-1=random)        (default: -1)
#   MVADAPTER_TEXT       Optional text prompt        (default: image-conditioned)
#   MVADAPTER_CHECKPOINTS  Directory with RealESRGAN/LaMa (auto-downloaded if unset)
#   MVADAPTER_REPO       Path to MV-Adapter repo     (auto-detected if unset)
#   MVADAPTER_SD21_BASE_MODEL  Local path or HF repo for SD2.1 base (auto-downloaded if unset)
#
# Usage:
#   bash mesh/scripts/generate_mesh.sh
#
#   # Default: TripoSG mesh → MVAdapter texturing
#   INPUT_DIR=/data/images OUTPUT_DIR=/data/out bash mesh/scripts/generate_mesh.sh
#
#   # Hunyuan mesh → MVAdapter texturing:
#   MESH_BACKBONE=hunyuan bash mesh/scripts/generate_mesh.sh
#
#   # TripoSG mesh → no texturing (bare geometry only):
#   TEXTURE_BACKEND=none bash mesh/scripts/generate_mesh.sh
#
#   # Hunyuan mesh → Hunyuan paint:
#   MESH_BACKBONE=hunyuan TEXTURE_BACKEND=hunyuan bash mesh/scripts/generate_mesh.sh
#
#   # MVAdapter with text prompt:
#   MVADAPTER_TEXT="a rusty robot" bash mesh/scripts/generate_mesh.sh
#
#   # MVAdapter with SD2.1 (lower VRAM):
#   MVADAPTER_VARIANT=sd21 bash mesh/scripts/generate_mesh.sh
# =============================================================================
set -euo pipefail

CONDA_ENV_NAME="hanyuan"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# ── activate conda env ────────────────────────────────────────────────────────
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"

# ── configuration with defaults ──────────────────────────────────────────────
INPUT_DIR="${INPUT_DIR:-${REPO_ROOT}/images}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/mesh/generated_meshes}"
MESH_BACKBONE="${MESH_BACKBONE:-triposg}"
TEXTURE_BACKEND="${TEXTURE_BACKEND:-mvadapter}"
# MESH_BACKBONE="${MESH_BACKBONE:-hunyuan}"
# TEXTURE_BACKEND="${TEXTURE_BACKEND:-hunyuan}"
NO_BG_REMOVAL="${NO_BG_REMOVAL:-0}"

# Hunyuan mesh backbone
HUNYUAN_MODEL="${HUNYUAN_MODEL:-tencent/Hunyuan3D-2}"

# TripoSG mesh backbone
TRIPOSG_MODEL_PATH="${TRIPOSG_MODEL_PATH:-}"
TRIPOSG_STEPS="${TRIPOSG_STEPS:-50}"
TRIPOSG_GUIDANCE="${TRIPOSG_GUIDANCE:-7.5}"
TRIPOSG_SEED="${TRIPOSG_SEED:-42}"

# MV-Adapter texturing
MVADAPTER_VARIANT="${MVADAPTER_VARIANT:-sdxl}"
MVADAPTER_STEPS="${MVADAPTER_STEPS:-50}"
MVADAPTER_GUIDANCE="${MVADAPTER_GUIDANCE:-3.0}"
MVADAPTER_SEED="${MVADAPTER_SEED:--1}"
MVADAPTER_TEXT="${MVADAPTER_TEXT:-}"
MVADAPTER_CHECKPOINTS="${MVADAPTER_CHECKPOINTS:-}"
MVADAPTER_REPO="${MVADAPTER_REPO:-}"
MVADAPTER_SD21_BASE_MODEL="${MVADAPTER_SD21_BASE_MODEL:-}"

# ── validation ────────────────────────────────────────────────────────────────
cd "${REPO_ROOT}"
mkdir -p "${OUTPUT_DIR}"

if [ ! -d "${INPUT_DIR}" ]; then
    echo "[ERROR] Input directory not found: ${INPUT_DIR}" >&2
    exit 1
fi

# Validate backend combination
if [ "${TEXTURE_BACKEND}" = "hunyuan" ] && [ "${MESH_BACKBONE}" != "hunyuan" ]; then
    echo "[ERROR] texture_backend='hunyuan' requires mesh_backbone='hunyuan'" >&2
    echo "        Current: mesh_backbone='${MESH_BACKBONE}'" >&2
    exit 1
fi

shopt -s nullglob
images=("${INPUT_DIR}"/*)
if [ "${#images[@]}" -eq 0 ]; then
    echo "[ERROR] No files found in ${INPUT_DIR}" >&2
    exit 1
fi

echo "[INFO] ─────────────────────────────────────────────────────────────"
echo "[INFO] Repo root        : ${REPO_ROOT}"
echo "[INFO] Input dir        : ${INPUT_DIR}"
echo "[INFO] Output dir       : ${OUTPUT_DIR}"
echo "[INFO] Mesh backbone    : ${MESH_BACKBONE}"
echo "[INFO] Texture backend  : ${TEXTURE_BACKEND}"
echo "[INFO] No BG removal    : ${NO_BG_REMOVAL}"

if [ "${MESH_BACKBONE}" = "hunyuan" ]; then
    echo "[INFO] ── Hunyuan mesh backbone:"
    echo "[INFO]    Model          : ${HUNYUAN_MODEL}"
fi

if [ "${MESH_BACKBONE}" = "triposg" ]; then
    echo "[INFO] ── TripoSG mesh backbone:"
    echo "[INFO]    Model path     : ${TRIPOSG_MODEL_PATH:-<auto-download>}"
    echo "[INFO]    Steps          : ${TRIPOSG_STEPS}"
    echo "[INFO]    Guidance       : ${TRIPOSG_GUIDANCE}"
    echo "[INFO]    Seed           : ${TRIPOSG_SEED}"
fi

if [ "${TEXTURE_BACKEND}" = "mvadapter" ]; then
    echo "[INFO] ── MV-Adapter texturing:"
    echo "[INFO]    Variant        : ${MVADAPTER_VARIANT}"
    echo "[INFO]    Steps          : ${MVADAPTER_STEPS}"
    echo "[INFO]    Guidance       : ${MVADAPTER_GUIDANCE}"
    echo "[INFO]    Seed           : ${MVADAPTER_SEED}"
    if [ -n "${MVADAPTER_TEXT}" ]; then
        echo "[INFO]    Text           : '${MVADAPTER_TEXT}'"
    else
        echo "[INFO]    Mode           : image-conditioned (no text)"
    fi
fi

echo "[INFO] ─────────────────────────────────────────────────────────────"

# ── build common Python argument array ───────────────────────────────────────
COMMON_ARGS=(
    --mesh_backbone    "${MESH_BACKBONE}"
    --texture_backend  "${TEXTURE_BACKEND}"
)

# Hunyuan mesh backbone args
[ "${MESH_BACKBONE}" = "hunyuan" ] && COMMON_ARGS+=(--hunyuan_model "${HUNYUAN_MODEL}")

# TripoSG mesh backbone args
if [ "${MESH_BACKBONE}" = "triposg" ]; then
    COMMON_ARGS+=(
        --triposg_steps    "${TRIPOSG_STEPS}"
        --triposg_guidance "${TRIPOSG_GUIDANCE}"
        --triposg_seed     "${TRIPOSG_SEED}"
    )
    [ -n "${TRIPOSG_MODEL_PATH}" ] && COMMON_ARGS+=(--triposg_model_path "${TRIPOSG_MODEL_PATH}")
fi

# MV-Adapter texturing args
if [ "${TEXTURE_BACKEND}" = "mvadapter" ]; then
    COMMON_ARGS+=(
        --mvadapter_variant  "${MVADAPTER_VARIANT}"
        --mvadapter_steps    "${MVADAPTER_STEPS}"
        --mvadapter_guidance "${MVADAPTER_GUIDANCE}"
        --mvadapter_seed     "${MVADAPTER_SEED}"
    )
    [ -n "${MVADAPTER_TEXT}" ]        && COMMON_ARGS+=(--mvadapter_text "${MVADAPTER_TEXT}")
    [ -n "${MVADAPTER_CHECKPOINTS}" ] && COMMON_ARGS+=(--mvadapter_checkpoints "${MVADAPTER_CHECKPOINTS}")
    [ -n "${MVADAPTER_REPO}" ]        && COMMON_ARGS+=(--mvadapter_repo "${MVADAPTER_REPO}")
    [ -n "${MVADAPTER_SD21_BASE_MODEL}" ] && COMMON_ARGS+=(--mvadapter_sd21_base_model "${MVADAPTER_SD21_BASE_MODEL}")
fi

# Common flags
[ "${NO_BG_REMOVAL}" = "1" ] && COMMON_ARGS+=(--no_bg_removal)

# ── cap OpenMP / BLAS thread counts ──────────────────────────────────────────
# With many CPU cores the default "one thread per core" causes libgomp to spawn
# dozens of threads at once.  Each thread needs its own stack; when the process
# is also holding large GPU/CPU tensors this exhausts virtual-address space or
# hits the container's thread limit, giving:
#   "libgomp: Thread creation failed: Resource temporarily unavailable"
# followed by heap corruption and a segfault.  Capping to a small fixed count
# is safe for GPU-bound inference and eliminates the crash.
THREAD_CAP="${OMP_NUM_THREADS:-4}"
export OMP_NUM_THREADS="${THREAD_CAP}"
export MKL_NUM_THREADS="${THREAD_CAP}"
export OPENBLAS_NUM_THREADS="${THREAD_CAP}"
export NUMEXPR_NUM_THREADS="${THREAD_CAP}"
export VECLIB_MAXIMUM_THREADS="${THREAD_CAP}"

# ── process all images in one Python process (models loaded once) ─────────────
# inference.py --input_dir processes every image in INPUT_DIR sequentially
# while keeping BEN2 / TripoSG / MVAdapter loaded in VRAM between images.
# This is significantly faster than the old per-image subprocess loop when
# more than one image is present.
echo ""
echo "[INFO] ── Launching batch inference (models loaded once for all images)"

PYOPENGL_PLATFORM=egl python -u mesh/inference.py \
    --input_dir  "${INPUT_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    "${COMMON_ARGS[@]}"

echo ""
echo "[INFO] All images processed.  Results in: ${OUTPUT_DIR}"
