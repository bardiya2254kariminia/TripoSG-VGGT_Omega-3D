"""
Camera pose estimation using VGGT-Omega.

Usage:
    from camera import Omega_CameraPoseFinder, load_vggt_omega

    device = "cuda"
    vggt_model = load_vggt_omega(checkpoint_path, device)

    pose_finder = Omega_CameraPoseFinder(vggt_model, image_size=512, device=device, output_dir="out/obj")
    pose_finder.set_mesh("mesh.glb")
    all_Rs, all_Ts = pose_finder.generate_initial_views(50, "obj")
    extrinsic, intrinsic = pose_finder.get_vggt_initial_guess(
        "query.png", all_Rs, all_Ts, "obj"
    )
"""

import contextlib
import os
import shutil
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from camera.models.hanyuan import Renderer

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


# ── VGGT-Omega loader ─────────────────────────────────────────────────────────

def load_vggt_omega(
    checkpoint_path: str,
    device,
    enable_alignment: bool = False,
) -> VGGTOmega:
    """Load VGGT-Omega once and reuse across multiple meshes/images."""
    dev = torch.device(device) if not isinstance(device, torch.device) else device
    print("Loading VGGT-Omega model...")
    model = VGGTOmega(enable_alignment=enable_alignment).to(dev)
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    model.load_state_dict(state_dict)
    model.eval()
    print("VGGT-Omega model loaded.")
    return model


# ── Omega_CameraPoseFinder ────────────────────────────────────────────────────

class Omega_CameraPoseFinder:
    """
    Camera pose estimator backed by VGGT-Omega.

    Load VGGT-Omega once via :func:`load_vggt_omega`, pass it here, then call
    :meth:`set_mesh` before each object. The pose model stays in memory; only
    the mesh renderer is swapped per object.

    Workflow:
      1. :meth:`set_mesh` — attach a .glb for rendering.
      2. :meth:`generate_initial_views` — render known views at ``self.dist``.
      3. :meth:`get_vggt_initial_guess` — estimate pose for a query image.
    """

    def __init__(
        self,
        vggt_model: VGGTOmega,
        image_size,
        device,
        output_dir: str,
        dist: float = 3.0,
        fov: float = 40,
        model_image_resolution: int = 512,
    ):
        """
        Args:
            vggt_model: Pre-loaded VGGT-Omega model (from :func:`load_vggt_omega`).
            image_size: Render resolution — int or (H, W) tuple.
            device: "cuda" or "cpu".
            output_dir: Directory for intermediate renders (temp_views).
            dist: Camera distance from origin used for context views and pose output.
            fov: Field of view in degrees (informational).
            model_image_resolution: Resolution fed to VGGT-Omega (e.g. 512).
        """
        self.vggt_model = vggt_model
        self.original_image_size = (
            image_size if isinstance(image_size, (list, tuple)) else (image_size, image_size)
        )
        self.device = torch.device(device) if not isinstance(device, torch.device) else device
        self.output_dir = output_dir
        self.dist = dist
        self.fov = fov
        self.model_image_resolution = model_image_resolution

        self.mesh_renderer = None
        self.known_image_paths = []

        self.view_path = os.path.join(output_dir, "temp_views")
        os.makedirs(output_dir, exist_ok=True)
        shutil.rmtree(self.view_path, ignore_errors=True)
        os.makedirs(self.view_path, exist_ok=True)

    def set_mesh(self, mesh_pth: str) -> None:
        """Load (or swap to) the mesh used for view rendering and pose estimation."""
        self.mesh_renderer = Renderer(mesh_pth, self.device)
        self.known_image_paths = []

    def _require_mesh_renderer(self) -> None:
        if self.mesh_renderer is None:
            raise RuntimeError("Call set_mesh(mesh_pth) before rendering or pose estimation.")

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

    # ── View generation ───────────────────────────────────────────────────────

    def generate_initial_views(self, num_views, img_name):
        """Generate context views on a sphere at ``self.dist`` (reference logic)."""
        self._require_mesh_renderer()

        print(f"Generating {num_views} initial views for context ...")
        self.known_image_paths = []

        azimuths = np.linspace(0, 360, int(np.sqrt(num_views * 2)), endpoint=False)
        elevations = np.linspace(-45, 45, int(np.sqrt(num_views / 2)), endpoint=True)

        num_of_views = 0
        all_Rs = []
        all_Ts = []

        for azimuth in azimuths:
            for elevation in elevations:
                r, t = self.get_opencv_camera_matrix(azimuth, elevation, self.dist)
                all_Rs.append(torch.from_numpy(r))
                all_Ts.append(torch.from_numpy(t))

                rendered_image = self.mesh_renderer.render(
                    image_size=self.original_image_size, R=r, T=t
                )
                rendered_image = rendered_image[0, ..., :3].cpu().numpy()

                path = os.path.join(self.view_path, f"{img_name}_{num_of_views}.png")
                plt.imsave(path, rendered_image)
                self.known_image_paths.append(path)
                num_of_views += 1

        print(f"Generated {num_of_views} views.")
        return all_Rs, all_Ts

    # ── Main pose estimation ──────────────────────────────────────────────────

    def get_vggt_initial_guess(self, target_image_path, all_Rs, all_Ts, img_name):
        """
        Estimate the camera pose for *target_image_path* given pre-rendered
        known views in ``self.known_image_paths``.

        Aligns VGGT-Omega output to the renderer frame via ``all_Rs[0]`` and
        assigns a fixed translation ``[0, 0, self.dist]`` to every view, matching
        the reference CameraPoseFinder logic.
        """
        self._require_mesh_renderer()

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

        R_reference = all_Rs[0]
        M_reference = torch.eye(4, device=self.device)
        M_reference[0:3, :3] = R_reference

        extrinsics_new = []
        for i in range(extrinsic.shape[1]):
            M_vggt = torch.eye(4, device=self.device)
            M_vggt[0:3, :] = extrinsic[0][i]

            M_aligned = torch.matmul(M_vggt, M_reference)
            M_aligned[0:3, 3] = torch.tensor(
                [0, 0, self.dist], dtype=torch.float32, device=self.device
            )

            extrinsics_new.append(M_aligned)

        extrinsics_new = torch.stack(extrinsics_new, dim=0).to(self.device)

        target_extrinsic = extrinsics_new[target_index]
        target_intrinsic = intrinsic[0, target_index]

        return target_extrinsic, target_intrinsic


# Backward-compatible alias
CameraPoseFinder = Omega_CameraPoseFinder


# ── Relative pose transfer ────────────────────────────────────────────────────

def compute_new_pose_from_relative(view_matrix_B_with_pose_A, view_matrix_B, view_matrix_B_prime):
    """
    Transfer a relative camera transform from object B onto B′.

    Given:
        V_B_with_pose_A  — camera pose for B photographed from pose A
        V_B              — camera pose for the canonical B reference photo
        V_B_prime        — camera pose for the B′ reference photo

    Returns:
        V_B_prime_new    — camera pose to render B′ after applying (B ref ← pose A)
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
