"""
Camera pose estimation using VGGT-Omega.

Pose model backend:
  "vggt_omega"  — Facebook's VGGT-Omega (Omega_CameraPoseFinder)

Mesh-generation / rendering backends:
  "hanyuan"  — Hunyuan3D-2 + PyTorch3D (camera.models.hanyuan)
  "trellis"  — TRELLIS.2-4B + PyTorch3D (camera.models.trellis)

Usage:
    pose_finder = create_pose_finder(
        mesh_pth="mesh.glb",
        image_size=512,
        device="cuda",
        backend="hanyuan",           # or "trellis"
        checkpoint_path="/workspace/checkpoints/vggt-omega/vggt_omega_1b_512.pt",
    )

    # Or instantiate directly:
    pose_finder = Omega_CameraPoseFinder(
        mesh_pth="mesh.glb",
        image_size=512,
        device="cuda",
        checkpoint_path="path/to/checkpoint.pt",
        backend="hanyuan",
    )
"""

import contextlib
import gc
import os
import shutil
import sys
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

# renderers
from camera.models.hanyuan import HunyuanRenderer

# VGGT-Omega imports
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    omega_path = os.path.abspath(os.path.join(current_dir, "..", "vggt-omega"))
    if omega_path not in sys.path:
        sys.path.insert(0, omega_path)

    from vggt_omega.models import VGGTOmega
    from vggt_omega.utils.load_fn import load_and_preprocess_images as load_and_preprocess_images_omega
    from vggt_omega.utils.pose_enc import encoding_to_camera
    print("Successfully imported VGGT-Omega modules.")
except ImportError as e:
    print(f"FATAL: Could not import VGGT-Omega modules. Error: {e}")
    sys.exit(1)


# ── Renderer factory ──────────────────────────────────────────────────────────

def _build_renderer(backend: str, mesh_pth: str, device):
    """
    Instantiate the appropriate mesh renderer based on the backend name.

    Args:
        backend: "hanyuan".
        mesh_pth: Path to the .glb mesh file.
        device: torch device or device string.

    Returns:
        HunyuanRenderer instance.
    """
    return HunyuanRenderer(mesh_pth, device)


# ── Omega_CameraPoseFinder ────────────────────────────────────────────────────

class Omega_CameraPoseFinder:
    """
    Camera pose estimator backed by VGGT-Omega.

    Workflow:
      1. Render known views of the mesh at multiple radii / angles.
      2. Feed the known views + the target image to VGGT-Omega.
      3. Find the closest known view by rotation similarity to seed the
         camera-distance search.
      4. Refine the translation magnitude with a binary search over rendered MSE.
    """

    def __init__(
        self,
        mesh_pth: str,
        image_size,
        device,
        dist: float = 3.0,
        fov: float = 40,
        checkpoint_path: str = "/workspace/checkpoints/checkpoints_vggt-omega/vggt_omega_1b_512.pt",
        model_image_resolution: int = 512,
        enable_alignment: bool = False,
        backend: Literal["hanyuan", "trellis"] = "hanyuan",
    ):
        """
        Args:
            mesh_pth: Path to the .glb mesh file.
            image_size: Render resolution — int or (H, W) tuple.
            device: "cuda" or "cpu".
            dist: Default camera distance from origin.
            fov: Field of view in degrees (informational).
            checkpoint_path: Path to the VGGT-Omega checkpoint .pt file.
            model_image_resolution: Resolution images are resized to before
                                    feeding to VGGT-Omega (e.g. 512).
            enable_alignment: Enable VGGT-Omega's alignment module.
            backend: Renderer backend — "hanyuan" (HunyuanRenderer) or
                     "trellis" (TrellisRenderer).
        """
        self.original_image_size = (
            image_size if isinstance(image_size, (list, tuple)) else (image_size, image_size)
        )
        self.device = torch.device(device) if not isinstance(device, torch.device) else device
        self.dist = dist
        self.fov = fov
        self.checkpoint_path = checkpoint_path
        self.model_image_resolution = model_image_resolution
        self.enable_alignment = enable_alignment
        self.backend = backend

        self.mesh_renderer = _build_renderer(backend, mesh_pth, self.device)
        self.vggt_model = None
        self.known_image_paths = []
        self.known_radii = []

        self.view_path = "./temp_views"
        shutil.rmtree(self.view_path, ignore_errors=True)
        os.makedirs(self.view_path, exist_ok=True)

    # ── VGGT-Omega load / unload ──────────────────────────────────────────────

    def _load_vggt(self):
        if self.vggt_model is None:
            print("Loading VGGT-Omega model...")
            self.vggt_model = VGGTOmega(enable_alignment=self.enable_alignment).to(self.device)
            state_dict = torch.load(self.checkpoint_path, map_location="cpu")
            if isinstance(state_dict, dict) and "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            self.vggt_model.load_state_dict(state_dict)
            self.vggt_model.eval()
            print("VGGT-Omega model loaded.")

    def _unload_vggt(self):
        if self.vggt_model is not None:
            del self.vggt_model
            self.vggt_model = None
            gc.collect()
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            print("VGGT-Omega model unloaded.")

    # ── Camera matrix helper ──────────────────────────────────────────────────

    def get_opencv_camera_matrix(self, azimuth_deg, elevation_deg, distance_from_origin):
        azimuth_rad = np.deg2rad(azimuth_deg)
        elevation_rad = np.deg2rad(elevation_deg)

        cam_x = distance_from_origin * np.cos(elevation_rad) * np.sin(azimuth_rad)
        cam_y = distance_from_origin * np.sin(elevation_rad)
        cam_z = distance_from_origin * np.cos(elevation_rad) * np.cos(azimuth_rad)

        camera_position = np.array([cam_x, cam_y, cam_z])
        target_position = np.array([0.0, 0.0, 0.0])
        world_up_vector = np.array([0.0, 1.0, 0.0])

        forward_vec = target_position - camera_position
        forward_vec /= np.linalg.norm(forward_vec)

        right_vec = np.cross(world_up_vector, forward_vec)
        right_vec /= np.linalg.norm(right_vec)

        up_vec = np.cross(forward_vec, right_vec)  # noqa: F841

        R = np.stack([right_vec, -up_vec, forward_vec], axis=0)
        T = -R @ camera_position.reshape(3, 1)

        return R, T

    # ── Multi-radius view generation ──────────────────────────────────────────

    def generate_initial_views(self, num_views, img_name, radii=None):
        """
        Generate known views at multiple radii so VGGT-Omega has visual context
        at every plausible camera distance.
        """
        if radii is None:
            radii = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]

        print(f"Generating initial views across radii {radii} ...")
        self.known_image_paths = []
        self.known_radii = []

        azimuths = np.linspace(0, 360, int(np.sqrt(num_views * 2)), endpoint=False)
        elevations = np.linspace(-45, 45, int(np.sqrt(num_views / 2)), endpoint=True)

        num_of_views = 0
        all_Rs = []
        all_Ts = []

        os.makedirs("./temp_views", exist_ok=True)

        for radius in radii:
            for azimuth in azimuths:
                for elevation in elevations:
                    r, t = self.get_opencv_camera_matrix(azimuth, elevation, radius)
                    all_Rs.append(torch.from_numpy(r))
                    all_Ts.append(torch.from_numpy(t))

                    rendered_image = self.mesh_renderer.render(
                        image_size=self.original_image_size, R=r, T=t
                    )
                    rendered_image = rendered_image[0, ..., :3].cpu().numpy()

                    path = f"temp_views/{img_name}_{num_of_views}.png"
                    plt.imsave(path, rendered_image)
                    self.known_image_paths.append(path)
                    self.known_radii.append(radius)
                    num_of_views += 1

        print(
            f"Generated {num_of_views} views total "
            f"({len(radii)} radii × azimuth × elevation)."
        )
        return all_Rs, all_Ts

    # ── VGGT-Omega depth helper ───────────────────────────────────────────────

    def _get_omega_depth_estimate(self, predictions, target_index):
        """
        Use VGGT-Omega's predicted depth map as a coarse distance estimate.
        """
        try:
            depth_predictions = predictions["depth"]  # [B, S, H, W, 1]
            target_depth_map = depth_predictions[0, target_index]

            if target_depth_map.ndim == 3:
                target_depth_map = target_depth_map[..., 0]

            valid_depths = target_depth_map[target_depth_map > 0]
            if valid_depths.numel() == 0:
                print("VGGT-Omega depth map is empty — skipping.")
                return None

            estimate = valid_depths.median().item()
            print(f"VGGT-Omega depth estimate: {estimate:.3f}")
            return estimate

        except Exception as e:
            print(f"VGGT-Omega depth unavailable ({e}) — skipping.")
            return None

    # ── Render-and-compare helper ─────────────────────────────────────────────

    def _render_and_compare(self, target_img_np, R_np, dist):
        """
        Render the mesh from rotation R_np at the given distance and
        return the MSE against the target image (values in [0, 1]).
        """
        forward = -R_np[2, :]
        cam_pos_at_dist = forward * (-dist)
        T_new = -R_np @ cam_pos_at_dist.reshape(3, 1)

        rendered = self.mesh_renderer.render(
            image_size=self.original_image_size, R=R_np, T=T_new
        )
        rendered_np = rendered[0, ..., :3].cpu().numpy()

        h, w = rendered_np.shape[:2]
        if target_img_np.shape[:2] != (h, w):
            from PIL import Image as PILImage
            target_resized = np.array(
                PILImage.fromarray((target_img_np * 255).astype(np.uint8)).resize((w, h))
            ) / 255.0
        else:
            target_resized = target_img_np

        mse = np.mean((rendered_np - target_resized) ** 2)
        return mse

    # ── Binary search for best distance ──────────────────────────────────────

    def _binary_search_distance(self, target_img_np, R_np, low, high, iters=6):
        """
        Binary search over camera distance, probing MSE at midpoint ± delta
        and shrinking the range toward the lower-error side.
        """
        print(f"Binary search for distance in [{low:.2f}, {high:.2f}]")

        for i in range(iters):
            mid = (low + high) / 2.0
            delta = (high - low) * 0.1

            score_left = self._render_and_compare(target_img_np, R_np, mid - delta)
            score_right = self._render_and_compare(target_img_np, R_np, mid + delta)

            if score_left < score_right:
                high = mid
            else:
                low = mid

            print(
                f"  iter {i + 1}: range=[{low:.3f}, {high:.3f}]  "
                f"score_left={score_left:.4f}  score_right={score_right:.4f}"
            )

        best = (low + high) / 2.0
        print(f"Binary search result: {best:.3f}")
        return best

    # ── Main pose estimation ──────────────────────────────────────────────────

    def get_vggt_initial_guess(self, target_image_path, all_Rs, all_Ts, img_name):
        """
        Estimate the camera pose for target_image_path given pre-rendered
        known views stored in self.known_image_paths / self.known_radii.

        Args:
            target_image_path: Path to the query image whose pose we want.
            all_Rs: list of (3, 3) rotation tensors for each known view.
            all_Ts: list of (3, 1) translation tensors for each known view.
            img_name: Prefix for debug render filenames.

        Returns:
            (target_extrinsic, target_intrinsic) — 4×4 and 3×3 tensors.
        """
        self._load_vggt()

        all_image_paths = self.known_image_paths + [target_image_path]
        target_index = len(all_image_paths) - 1

        images = load_and_preprocess_images_omega(
            all_image_paths,
            image_resolution=self.model_image_resolution,
        ).to(self.device)

        print("Running VGGT-Omega for initial pose estimation...")
        with torch.inference_mode():
            if self.device.type == "cuda":
                cap_major = torch.cuda.get_device_capability(self.device)[0]
                amp_dtype = torch.bfloat16 if cap_major >= 8 else torch.float16
                amp_ctx = torch.cuda.amp.autocast(dtype=amp_dtype)
            else:
                amp_ctx = contextlib.nullcontext()

            with amp_ctx:
                predictions = self.vggt_model(images)

            extrinsic, intrinsic = encoding_to_camera(
                predictions["pose_enc"],
                predictions["images"].shape[-2:],
            )

            # Step 1: coarse depth from VGGT-Omega depth head
            omega_depth_estimate = self._get_omega_depth_estimate(predictions, target_index)

        # Step 2: find closest known view by rotation similarity
        target_pose = extrinsic[0][target_index]
        best_known_idx = None
        best_similarity = float("inf")

        for i in range(len(self.known_image_paths)):
            known_pose = extrinsic[0][i]
            diff = torch.norm(target_pose[:, :3] - known_pose[:, :3]).item()
            if diff < best_similarity:
                best_similarity = diff
                best_known_idx = i

        sampling_radius = self.known_radii[best_known_idx]
        print(
            f"Closest known view radius: {sampling_radius:.2f}  "
            f"(pose diff={best_similarity:.4f})"
        )

        # Step 3: combine depth estimate + sampling radius
        if omega_depth_estimate is not None:
            # weight sampling radius more (0.6) — VGGT-Omega depth can be noisy
            initial_dist_estimate = 0.6 * sampling_radius + 0.4 * omega_depth_estimate
            search_margin = 0.8
        else:
            initial_dist_estimate = sampling_radius
            search_margin = 1.2

        search_low = max(0.5, initial_dist_estimate - search_margin)
        search_high = initial_dist_estimate + search_margin
        print(
            f"Initial distance estimate: {initial_dist_estimate:.3f}  "
            f"search range: [{search_low:.2f}, {search_high:.2f}]"
        )

        # Step 4: align rotation using first known view as reference
        R_reference = all_Rs[0]
        M_reference = torch.eye(4, device=self.device)
        M_reference[0:3, :3] = R_reference

        M_vggt_target = torch.eye(4, device=self.device)
        M_vggt_target[0:3, :] = extrinsic[0][target_index]
        M_aligned_target = torch.matmul(M_vggt_target, M_reference)
        R_target_np = M_aligned_target[0:3, :3].cpu().numpy()

        # Step 5: binary search for best distance
        target_img_np = np.array(
            Image.open(target_image_path).convert("RGB").resize(self.original_image_size)
        ) / 255.0

        best_dist = self._binary_search_distance(
            target_img_np, R_target_np, search_low, search_high, iters=6
        )

        # Step 6: build all final extrinsics
        extrinsics_new = []
        for i in range(extrinsic.shape[1]):
            M_vggt = torch.eye(4, device=self.device)
            M_vggt[0:3, :] = extrinsic[0][i]

            M_aligned = torch.matmul(M_vggt, M_reference)

            if i == target_index:
                M_aligned[0:3, 3] = torch.tensor(
                    [0, 0, best_dist], dtype=torch.float32, device=self.device
                )
            else:
                radius_i = self.known_radii[i] if i < len(self.known_radii) else self.dist
                M_aligned[0:3, 3] = torch.tensor(
                    [0, 0, radius_i], dtype=torch.float32, device=self.device
                )

            extrinsics_new.append(M_aligned)

        extrinsics_new = torch.stack(extrinsics_new, dim=0).to(self.device)

        target_extrinsic = extrinsics_new[target_index]
        target_intrinsic = intrinsic[0, target_index]

        # debug renders
        os.makedirs("./temp_views2", exist_ok=True)
        for i in range(len(all_image_paths)):
            rendered_image = self.mesh_renderer.render(
                image_size=self.original_image_size,
                R=extrinsics_new[i][0:3, :3].cpu().numpy(),
                T=extrinsics_new[i][0:3, 3].cpu().numpy(),
            )
            rendered_image = rendered_image[0, ..., :3].cpu().numpy()
            plt.imsave(f"temp_views2/{img_name}_{i}.png", rendered_image)

        self._unload_vggt()

        return target_extrinsic, target_intrinsic


# ── Factory function ──────────────────────────────────────────────────────────

def create_pose_finder(
    mesh_pth: str,
    image_size,
    device,
    backend: Literal["hanyuan", "trellis"] = "hanyuan",
    dist: float = 3.0,
    fov: float = 40,
    checkpoint_path: str = "/workspace/checkpoints/checkpoints_vggt-omega/vggt_omega_1b_512.pt",
    model_image_resolution: int = 512,
    enable_alignment: bool = False,
) -> Omega_CameraPoseFinder:
    """
    Create an Omega_CameraPoseFinder instance.

    Args:
        mesh_pth: Path to the .glb mesh file.
        image_size: Render resolution — int or (H, W) tuple.
        device: "cuda" or "cpu".
        backend: Renderer backend:
                 - "hanyuan": Hunyuan3D + PyTorch3D
                 - "trellis": TRELLIS.2 + PyTorch3D
        dist: Default camera distance from origin.
        fov: Field of view in degrees.
        checkpoint_path: Path to VGGT-Omega .pt checkpoint file.
        model_image_resolution: Image resolution fed to VGGT-Omega.
        enable_alignment: Enable VGGT-Omega alignment module.

    Returns:
        Omega_CameraPoseFinder instance.

    Example::

        finder = create_pose_finder(
            mesh_pth="mesh.glb",
            image_size=512,
            device="cuda",
            backend="hanyuan",
            checkpoint_path="./checkpoints/vggt-omega/vggt_omega_1b_512.pt",
        )
    """
    return Omega_CameraPoseFinder(
        mesh_pth=mesh_pth,
        image_size=image_size,
        device=device,
        dist=dist,
        fov=fov,
        checkpoint_path=checkpoint_path,
        model_image_resolution=model_image_resolution,
        enable_alignment=enable_alignment,
        backend=backend,
    )


# ── Relative pose transfer ────────────────────────────────────────────────────

def compute_new_pose_from_relative(view_matrix_B_with_pose_A, view_matrix_B, view_matrix_B_prime):
    """
    Transfer a relative rotation from one object to another.

    Given:
        V_B_with_pose_A  — camera pose used to photograph horse B from angle A
        V_B              — camera pose used to photograph horse B from its reference angle
        V_B_prime        — camera pose used to photograph the jumping horse

    Returns:
        V_B_prime_new    — camera pose to render the jumping horse from angle B
    """
    V_B_with_pose_A = view_matrix_B_with_pose_A.cpu().numpy()
    V_B = view_matrix_B.cpu().numpy()
    V_B_prime = view_matrix_B_prime.cpu().numpy()

    try:
        V_B_with_pose_A_inv = np.linalg.inv(V_B_with_pose_A)
    except np.linalg.LinAlgError:
        print("Error: view_matrix_B_with_pose_A is not invertible.")
        return None

    delta_transform = V_B @ V_B_with_pose_A_inv
    V_B_prime_new = delta_transform @ V_B_prime

    return V_B_prime_new
