#!/usr/bin/env bash
# =============================================================================
# generate_mesh.sh — batch mesh generation (Hunyuan3D-2 shape + TripoSG texture)
#
# Processes every image in INPUT_DIR and writes outputs to OUTPUT_DIR.
#
# Configurable via environment variables (all optional):
#
#   INPUT_DIR            Directory of input images   (default: <repo>/images)
#   OUTPUT_DIR           Directory for outputs       (default: <repo>/mesh/generated_meshes)
#   HUNYUAN_MODEL        Hunyuan3D-2 HF model ID     (default: tencent/Hunyuan3D-2)
#   TEXTURE_BACKEND      "triposg" (default) or "hunyuan"
#   NO_TEXTURE           Set to "1" to skip texturizing and save bare mesh only
#   NO_BG_REMOVAL        Set to "1" to skip BEN2 background removal
#
#   # TripoSG knobs (only used when TEXTURE_BACKEND=triposg):
#   TRIPOSG_MODEL_PATH   Local TripoSG weights dir   (auto-downloaded if unset)
#   TRIPOSG_STEPS        Denoising steps             (default: 50)
#   TRIPOSG_GUIDANCE     Guidance scale              (default: 7.5)
#   TRIPOSG_SEED         RNG seed                    (default: 42)
#
# Usage:
#   bash mesh/scripts/generate_mesh.sh
#
#   # TripoSG texturizing (default):
#   INPUT_DIR=/data/images OUTPUT_DIR=/data/out bash mesh/scripts/generate_mesh.sh
#
#   # Hunyuan paint instead of TripoSG:
#   TEXTURE_BACKEND=hunyuan bash mesh/scripts/generate_mesh.sh
#
#   # Bare mesh only:
#   NO_TEXTURE=1 bash mesh/scripts/generate_mesh.sh
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
HUNYUAN_MODEL="${HUNYUAN_MODEL:-tencent/Hunyuan3D-2}"
TEXTURE_BACKEND="${TEXTURE_BACKEND:-triposg}"
NO_TEXTURE="${NO_TEXTURE:-0}"
NO_BG_REMOVAL="${NO_BG_REMOVAL:-0}"

TRIPOSG_MODEL_PATH="${TRIPOSG_MODEL_PATH:-}"
TRIPOSG_STEPS="${TRIPOSG_STEPS:-50}"
TRIPOSG_GUIDANCE="${TRIPOSG_GUIDANCE:-7.5}"
TRIPOSG_SEED="${TRIPOSG_SEED:-42}"

# ── validation ────────────────────────────────────────────────────────────────
cd "${REPO_ROOT}"
mkdir -p "${OUTPUT_DIR}"

if [ ! -d "${INPUT_DIR}" ]; then
    echo "[ERROR] Input directory not found: ${INPUT_DIR}" >&2
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
echo "[INFO] Hunyuan model    : ${HUNYUAN_MODEL}"
echo "[INFO] Texture backend  : ${TEXTURE_BACKEND}"
echo "[INFO] No texture       : ${NO_TEXTURE}"
echo "[INFO] No BG removal    : ${NO_BG_REMOVAL}"
if [ "${TEXTURE_BACKEND}" = "triposg" ]; then
    echo "[INFO] TripoSG path     : ${TRIPOSG_MODEL_PATH:-<auto-download>}"
    echo "[INFO] TripoSG steps    : ${TRIPOSG_STEPS}"
    echo "[INFO] TripoSG guidance : ${TRIPOSG_GUIDANCE}"
    echo "[INFO] TripoSG seed     : ${TRIPOSG_SEED}"
fi
echo "[INFO] ─────────────────────────────────────────────────────────────"

# ── build common Python argument array ───────────────────────────────────────
COMMON_ARGS=(
    --hunyuan_model  "${HUNYUAN_MODEL}"
    --texture_backend "${TEXTURE_BACKEND}"
    --triposg_steps  "${TRIPOSG_STEPS}"
    --triposg_guidance "${TRIPOSG_GUIDANCE}"
    --triposg_seed   "${TRIPOSG_SEED}"
)

[ "${NO_TEXTURE}"    = "1" ] && COMMON_ARGS+=(--no_texture)
[ "${NO_BG_REMOVAL}" = "1" ] && COMMON_ARGS+=(--no_bg_removal)
[ -n "${TRIPOSG_MODEL_PATH}" ] && COMMON_ARGS+=(--triposg_model_path "${TRIPOSG_MODEL_PATH}")

# ── process images ────────────────────────────────────────────────────────────
for img in "${images[@]}"; do
    [ -f "${img}" ] || continue
    name="$(basename "${img}")"
    name="${name%.*}"
    img_output_dir="${OUTPUT_DIR}/${name}"

    echo ""
    echo "[INFO] ── Processing: ${img}"
    echo "[INFO]    Output dir: ${img_output_dir}"

    PYOPENGL_PLATFORM=egl python -u mesh/inference.py \
        --image        "${img}" \
        --output_dir   "${img_output_dir}" \
        "${COMMON_ARGS[@]}"

    echo "[INFO] ── Done: ${name}"
done

echo ""
echo "[INFO] All images processed.  Results in: ${OUTPUT_DIR}"
