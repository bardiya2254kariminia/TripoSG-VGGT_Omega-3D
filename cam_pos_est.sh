#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Folder mapping:
#   mesh/generated_meshes/input/   → --B              (canonical reference)
#   mesh/generated_meshes/B/       → --B_with_pose_A  (pose A)
#   mesh/generated_meshes/B_prime/ → --B_prime        (render target)

python "${SCRIPT_DIR}/camera_pose_estimation.py" \
    --B              "${SCRIPT_DIR}/mesh/generated_meshes/input" \
    --B_with_pose_A  "${SCRIPT_DIR}/mesh/generated_meshes/B" \
    --B_prime        "${SCRIPT_DIR}/mesh/generated_meshes/B_prime" \
    --checkpoint_path  "${SCRIPT_DIR}/checkpoints/vggt-omega/vggt_omega_1b_512.pt" \
    --output_dir   "${SCRIPT_DIR}/rendered_images"
