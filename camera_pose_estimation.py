"""
Camera pose estimation and mesh rendering.

Asset directories (from cam_pos_est.sh):
  --input_dir    → canonical B reference  (input.png, input_mesh_textured.glb)
  --B_dir        → B photographed from pose A  (B.png, B_mesh_textured.glb)
  --B_prime_dir  → target object B'  (B_prime.png, B_prime_mesh_textured.glb)

Usage (see cam_pos_est.sh):
    python camera_pose_estimation.py \\
        --input_dir    mesh/generated_meshes/input \\
        --B_dir        mesh/generated_meshes/B \\
        --B_prime_dir  mesh/generated_meshes/B_prime \\
        --output_dir   rendered_images

Output layout under --output_dir:
  B/         temp_views, temp_views2 for B_with_pose_A (from B_dir)
  input/     temp_views, temp_views2 for canonical B (from input_dir)
  B_prime/   temp_views, temp_views2 + final B_prime.png render
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from camera import (
    Omega_CameraPoseFinder,
    compute_new_pose_from_relative,
    load_vggt_omega,
)


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate camera poses with VGGT-Omega and transfer them across objects."
    )
    parser.add_argument(
        "--input_dir", required=True,
        help="Directory with input.png + input_mesh_textured.glb (canonical B reference).",
    )
    parser.add_argument(
        "--B_dir", required=True,
        help="Directory with B.png + B_mesh_textured.glb (B photographed from pose A).",
    )
    parser.add_argument(
        "--B_prime_dir", required=True,
        help="Directory with B_prime.png + B_prime_mesh_textured.glb (render target).",
    )
    parser.add_argument("--output_dir", default="rendered_images")
    parser.add_argument("--num_views", type=int, default=50)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument(
        "--checkpoint_path",
        default="/workspace/checkpoints/checkpoints_vggt-omega/vggt_omega_1b_512.pt",
    )
    return parser.parse_args()


# ── Asset resolution ───────────────────────────────────────────────────────────

ASSET_SPECS = {
    "input": ("input.png", "input_mesh_textured.glb"),
    "B": ("B.png", "B_mesh_textured.glb"),
    "B_prime": ("B_prime.png", "B_prime_mesh_textured.glb"),
}


def _resolve_assets(asset_dir: str, label: str) -> tuple[str, str]:
    if label not in ASSET_SPECS:
        raise ValueError(f"Unknown asset label: {label}")

    if not os.path.isdir(asset_dir):
        raise FileNotFoundError(f"Asset directory not found: {asset_dir}")

    image_name, mesh_name = ASSET_SPECS[label]
    image_path = os.path.join(asset_dir, image_name)
    mesh_path = os.path.join(asset_dir, mesh_name)

    missing = [p for p in (image_path, mesh_path) if not os.path.isfile(p)]
    if missing:
        raise FileNotFoundError(
            f"Missing required files in {asset_dir} ({label}):\n"
            + "\n".join(f"  - {p}" for p in missing)
        )

    return image_path, mesh_path


def _estimate_pose(
    vggt_model,
    output_dir: str,
    mesh_path: str,
    image_path: str,
    img_name: str,
    num_views: int,
    image_size: tuple[int, int],
    device: torch.device,
) -> tuple[Omega_CameraPoseFinder, torch.Tensor]:
    """Run view generation + pose estimation; writes intermediates under output_dir."""
    pose_finder = Omega_CameraPoseFinder(
        vggt_model, image_size, device, output_dir=output_dir
    )
    pose_finder.set_mesh(mesh_path)
    all_Rs, all_Ts = pose_finder.generate_initial_views(num_views, img_name)
    extrinsic, _ = pose_finder.get_vggt_initial_guess(
        image_path, all_Rs, all_Ts, img_name
    )
    return pose_finder, extrinsic


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_size = (args.image_size, args.image_size)

    image_B_with_pose_A_path, mesh_B_with_pose_A_path = _resolve_assets(args.B_dir, "B")
    image_B_path, mesh_B_path = _resolve_assets(args.input_dir, "input")
    image_B_prime_path, mesh_B_prime_path = _resolve_assets(args.B_prime_dir, "B_prime")

    out_B_dir = os.path.join(args.output_dir, "B")
    out_input_dir = os.path.join(args.output_dir, "input")
    out_B_prime_dir = os.path.join(args.output_dir, "B_prime")

    print("Using assets:")
    print(f"  B_with_pose_A (from B_dir):   {image_B_with_pose_A_path}  |  {mesh_B_with_pose_A_path}")
    print(f"    → output: {out_B_dir}")
    print(f"  B reference (from input_dir): {image_B_path}  |  {mesh_B_path}")
    print(f"    → output: {out_input_dir}")
    print(f"  B' (from B_prime_dir):        {image_B_prime_path}  |  {mesh_B_prime_path}")
    print(f"    → output: {out_B_prime_dir}")

    os.makedirs(args.output_dir, exist_ok=True)

    vggt_model = load_vggt_omega(args.checkpoint_path, device)

    print("\n── Pose estimation for B_with_pose_A ──")
    _, initial_extrinsic_B_with_pose_A = _estimate_pose(
        vggt_model, out_B_dir,
        mesh_B_with_pose_A_path, image_B_with_pose_A_path, "B_with_pose_A",
        args.num_views, image_size, device,
    )
    print(f"\nExtrinsic B_with_pose_A:\n{initial_extrinsic_B_with_pose_A}")

    print("\n── Pose estimation for B ──")
    _, initial_extrinsic_B = _estimate_pose(
        vggt_model, out_input_dir,
        mesh_B_path, image_B_path, "B",
        args.num_views, image_size, device,
    )
    print(f"\nExtrinsic B:\n{initial_extrinsic_B}")

    print("\n── Pose estimation for B' ──")
    pose_finder, initial_extrinsic_B_prime = _estimate_pose(
        vggt_model, out_B_prime_dir,
        mesh_B_prime_path, image_B_prime_path, "B_prime",
        args.num_views, image_size, device,
    )
    print(f"\nExtrinsic B':\n{initial_extrinsic_B_prime}")

    print("\n── Computing final pose for B' ──")
    final_B_prime_extrinsic = compute_new_pose_from_relative(
        initial_extrinsic_B_with_pose_A,
        initial_extrinsic_B,
        initial_extrinsic_B_prime,
    )

    R_B_prime: np.ndarray = final_B_prime_extrinsic[0:3, :3]
    T_B_prime: np.ndarray = final_B_prime_extrinsic[0:3, 3]
    print(f"\nFinal R for B':\n{R_B_prime}")
    print(f"Final T for B':\n{T_B_prime}")

    final_rendered_image = pose_finder.mesh_renderer.render(
        image_size[0], R_B_prime, T_B_prime
    )
    final_rendered_image = final_rendered_image[0, ..., :3].cpu().numpy()

    out_name = os.path.splitext(os.path.basename(image_B_prime_path))[0] + ".png"
    out_path = os.path.join(out_B_prime_dir, out_name)
    plt.imsave(out_path, final_rendered_image)
    print(f"\nFinal render saved → {out_path}")


if __name__ == "__main__":
    main()
