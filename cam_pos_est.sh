#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python "${SCRIPT_DIR}/camera_pose_estimation.py" \
    --input_dir    "${SCRIPT_DIR}/mesh/generated_meshes/input" \
    --B_dir        "${SCRIPT_DIR}/mesh/generated_meshes/B" \
    --B_prime_dir  "${SCRIPT_DIR}/mesh/generated_meshes/B_prime" \
    --checkpoint_path  "${SCRIPT_DIR}/checkpoints/vggt-omega/vggt_omega_1b_512.pt" \
    --output_dir   "${SCRIPT_DIR}/rendered_images"
