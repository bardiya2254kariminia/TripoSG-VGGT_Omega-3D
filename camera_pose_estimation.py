"""
Camera pose estimation and mesh rendering.

Folder mapping (see cam_pos_est.sh):
  mesh/generated_meshes/input/   → --B              (input.png, input_mesh_textured.glb)
  mesh/generated_meshes/B/       → --B_with_pose_A  (B.png, B_mesh_textured.glb)
  mesh/generated_meshes/B_prime/ → --B_prime        (B_prime.png, B_prime_mesh_textured.glb)

Pipeline (matches tetmp_codes/camera_fincer.py):
  1. Estimate camera for B_with_pose_A (pose A photo + its mesh)
  2. Estimate camera for B (reference photo + its mesh)
  3. Estimate camera for B_prime (reference photo + its mesh)
  4. final camera = V_B @ inv(V_A) @ V_B_prime  (reference formula)
     with MSE-refined poses; also saves a direct input-camera render.
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
from camera.models.hanyuan import Renderer


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate camera poses with VGGT-Omega and transfer them across objects."
    )
    parser.add_argument(
        "--B_with_pose_A", required=True, dest="B_with_pose_A_dir",
        help="Directory with B.png + B_mesh_textured.glb (pose A; typically mesh/generated_meshes/B).",
    )
    parser.add_argument(
        "--B", required=True, dest="B_dir",
        help="Directory with input.png + input_mesh_textured.glb (reference; typically mesh/generated_meshes/input).",
    )
    parser.add_argument(
        "--B_prime", required=True, dest="B_prime_dir",
        help="Directory with B_prime.png + B_prime_mesh_textured.glb (render target).",
    )
    parser.add_argument("--output_dir", default="rendered_images")
    parser.add_argument("--num_views", type=int, default=50)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument(
        "--dist", type=float, default=3.0,
        help="Camera distance used for context views and VGGT pose output.",
    )
    parser.add_argument(
        "--checkpoint_path",
        default="/workspace/checkpoints/checkpoints_vggt-omega/vggt_omega_1b_512.pt",
    )
    parser.add_argument(
        "--no_bg_removal",
        action="store_true",
        help="Skip BEN2 background removal; use original images as-is (reference default).",
    )
    return parser.parse_args()


# ── Asset resolution ───────────────────────────────────────────────────────────

ASSET_SPECS = {
    "B": ("input.png", "input_mesh_textured.glb"),
    "B_with_pose_A": ("B.png", "B_mesh_textured.glb"),
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


# ── BEN2 background erasure (optional; reference uses raw images) ───────────────

def _load_ben2_model(device: torch.device):
    from ben2 import AutoModel

    print("Loading BEN2 background removal model ...")
    model = AutoModel.from_pretrained("PramaLLC/BEN2")
    model.to(device).eval()
    return model


def _erase_background_white(ben_model, image_path: str, output_path: str) -> str:
    from PIL import Image

    image = Image.open(image_path).convert("RGB")
    print(f"[BG] Running BEN2 on {image_path} ...")
    foreground = ben_model.inference(image).convert("RGBA")

    white_bg = Image.new("RGBA", foreground.size, (255, 255, 255, 255))
    white_bg.paste(foreground, mask=foreground.split()[3])
    white_bg.convert("RGB").save(output_path)
    print(f"[BG] White-background image saved → {output_path}")
    return output_path


def _prepare_query_images(
    image_paths: dict[str, str],
    output_dirs: dict[str, str],
    device: torch.device,
    skip_bg_removal: bool = False,
) -> dict[str, str]:
    if skip_bg_removal:
        return dict(image_paths)

    ben_model = _load_ben2_model(device)
    prepared: dict[str, str] = {}

    for label, src_path in image_paths.items():
        out_dir = output_dirs[label]
        os.makedirs(out_dir, exist_ok=True)
        dst_path = os.path.join(out_dir, f"{label}_white_bg.png")
        prepared[label] = _erase_background_white(ben_model, src_path, dst_path)

    del ben_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return prepared


def _estimate_pose(
    vggt_model,
    output_dir: str,
    mesh_path: str,
    image_path: str,
    img_name: str,
    num_views: int,
    image_size: tuple[int, int],
    device: torch.device,
    dist: float,
) -> torch.Tensor:
    """Run view generation + pose estimation; writes temp_views / temp_views2."""
    pose_finder = Omega_CameraPoseFinder(
        vggt_model, image_size, device, output_dir=output_dir, dist=dist
    )
    pose_finder.set_mesh(mesh_path)
    all_Rs, all_Ts = pose_finder.generate_initial_views(num_views, img_name)
    extrinsic, _ = pose_finder.get_vggt_initial_guess(
        image_path, all_Rs, all_Ts, img_name
    )
    return extrinsic


def _y180_flip_rt(R: np.ndarray, T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """180° yaw correction for front/back mesh ambiguity."""
    R_y180 = np.diag([-1.0, 1.0, -1.0])
    R_flip = R @ R_y180
    cam_pos = (-R.T @ T.reshape(3, 1)).reshape(3)
    cam_flip = np.array([-cam_pos[0], cam_pos[1], -cam_pos[2]])
    T_flip = (-R_flip @ cam_flip.reshape(3, 1)).reshape(3)
    return R_flip, T_flip


def _side_view_score(img: np.ndarray) -> float:
    """Higher = more side-like (asymmetric silhouette), lower = back/front."""
    gray = img.mean(axis=-1)
    mask = gray < 0.98
    if mask.sum() < 50:
        return 0.0
    h, w = gray.shape
    left = float(mask[:, : w // 2].sum())
    right = float(mask[:, w // 2 :].sum())
    return abs(left - right) / (left + right + 1e-6)


def _pick_best_orientation(
    renderer: Renderer, image_size: int, R: np.ndarray, T: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Render with R,T and Y180-flipped variant; pick the more side-like view."""
    img_a = renderer.render(image_size, R, T)[0, ..., :3].cpu().numpy()
    R_b, T_b = _y180_flip_rt(R, T)
    img_b = renderer.render(image_size, R_b, T_b)[0, ..., :3].cpu().numpy()

    if _side_view_score(img_b) > _side_view_score(img_a):
        return R_b, T_b, img_b
    return R, T, img_a


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_size = (args.image_size, args.image_size)

    image_B_with_pose_A_path, _ = _resolve_assets(args.B_with_pose_A_dir, "B_with_pose_A")
    image_B_path, mesh_B_path = _resolve_assets(args.B_dir, "B")
    image_B_prime_path, mesh_B_prime_path = _resolve_assets(args.B_prime_dir, "B_prime")
    # Pose A and B must share one mesh so VGGT rotations are comparable.
    mesh_B_with_pose_A_path = mesh_B_path

    out_B_with_pose_A_dir = os.path.join(args.output_dir, "B_with_pose_A")
    out_B_dir = os.path.join(args.output_dir, "B")
    out_B_prime_dir = os.path.join(args.output_dir, "B_prime")
    os.makedirs(args.output_dir, exist_ok=True)

    query_images = _prepare_query_images(
        image_paths={
            "B_with_pose_A": image_B_with_pose_A_path,
            "B": image_B_path,
            "B_prime": image_B_prime_path,
        },
        output_dirs={
            "B_with_pose_A": out_B_with_pose_A_dir,
            "B": out_B_dir,
            "B_prime": out_B_prime_dir,
        },
        device=device,
        skip_bg_removal=args.no_bg_removal,
    )

    vggt_model = load_vggt_omega(args.checkpoint_path, device)

    # ── finding the camera parameters for B_with_pose_A (same mesh as B) ──
    print("\n── Pose estimation for B_with_pose_A ──")
    print(f"  (using reference mesh: {mesh_B_with_pose_A_path})")
    initial_extrinsic_B_with_pose_A = _estimate_pose(
        vggt_model, out_B_with_pose_A_dir,
        mesh_B_with_pose_A_path, query_images["B_with_pose_A"], "B_with_pose_A",
        args.num_views, image_size, device, args.dist,
    )
    print(f"\nVGGT Initial Extrinsic (4x4) for B_with_pose_A:\n{initial_extrinsic_B_with_pose_A}")

    # ── finding the camera parameters for B ──
    print("\n── Pose estimation for B ──")
    initial_extrinsic_B = _estimate_pose(
        vggt_model, out_B_dir,
        mesh_B_path, query_images["B"], "B",
        args.num_views, image_size, device, args.dist,
    )
    print(f"\nVGGT Initial Extrinsic (4x4) for B:\n{initial_extrinsic_B}")

    # ── finding the camera parameters for B_prime ──
    print("\n── Pose estimation for B_prime ──")
    initial_extrinsic_B_prime = _estimate_pose(
        vggt_model, out_B_prime_dir,
        mesh_B_prime_path, query_images["B_prime"], "B_prime",
        args.num_views, image_size, device, args.dist,
    )
    print(f"\nVGGT Initial Extrinsic (4x4) for B_prime:\n{initial_extrinsic_B_prime}")

    # ── final camera for B_prime ──
    print("\n── Final B_prime camera ──")
    final_relative = compute_new_pose_from_relative(
        initial_extrinsic_B_with_pose_A,
        initial_extrinsic_B,
        initial_extrinsic_B_prime,
    )

    # Direct input-camera pose (MSE-refined V_B) — matches input.png angle.
    direct_extrinsic = initial_extrinsic_B.cpu().numpy()
    R_direct = direct_extrinsic[0:3, 0:3]
    T_direct = direct_extrinsic[0:3, 3]

    R_rel = final_relative[0:3, 0:3]
    T_rel = final_relative[0:3, 3]
    print(f"\nDirect input camera R:\n{R_direct}")
    print(f"\nRelative-transfer R:\n{R_rel}")

    mesh_renderer = Renderer(mesh_B_prime_path, device)

    # Primary output: input.png camera on B′ mesh (+ auto Y180 fix if mesh faces backward).
    R_final, T_final, final_rendered_image = _pick_best_orientation(
        mesh_renderer, args.image_size, R_direct, T_direct
    )

    # Also save relative-transfer variant for comparison.
    rel_img = mesh_renderer.render(args.image_size, R_rel, T_rel)
    rel_img = rel_img[0, ..., :3].cpu().numpy()
    plt.imsave(os.path.join(out_B_prime_dir, "B_prime_relative.png"), rel_img)

    # Sanity: reference mesh with same direct camera (should match input.png side view).
    ref_renderer = Renderer(mesh_B_path, device)
    ref_check = ref_renderer.render(args.image_size, R_final, T_final)
    ref_check = ref_check[0, ..., :3].cpu().numpy()
    plt.imsave(os.path.join(out_B_dir, "reference_camera_check.png"), ref_check)

    out_name = os.path.splitext(os.path.basename(image_B_prime_path))[0] + ".png"
    out_path = os.path.join(out_B_prime_dir, out_name)
    plt.imsave(out_path, final_rendered_image)
    print(f"\nFinal render saved → {out_path}")


if __name__ == "__main__":
    main()
