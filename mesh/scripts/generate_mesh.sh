#!/usr/bin/env bash
set -euo pipefail

CONDA_ENV_NAME="hanyuan"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Activate Conda environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"

INPUT_DIR="${INPUT_DIR:-${REPO_ROOT}/images}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/mesh/generated_meshes}"
MODEL="${MODEL:-tencent/Hunyuan3D-2}"

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

for img in "${images[@]}"; do
    [ -f "${img}" ] || continue
    name="$(basename "${img}")"
    name="${name%.*}"

    echo "[INFO] Processing ${img} → ${OUTPUT_DIR}/${name}"
    PYOPENGL_PLATFORM=egl python -u ./mesh/hanyuan_vggt-omega_mesh_inference.py \
        --image "${img}" \
        --output_dir "${OUTPUT_DIR}/${name}" \
        --model "${MODEL}"
done