"""
Camera pose estimation using VGGT / VGGT-Omega.

Pose model backends:
  "vggt"        — Facebook's VGGT-1B (CameraPoseFinder)
  "vggt_omega"  — Facebook's VGGT-Omega (Omega_CameraPoseFinder)

Mesh-generation / rendering backends:
  "hanyuan"  — Hunyuan3D-2 + PyTorch3D (camera.models.hanyuan)
  "trellis"  — TRELLIS.2-4B + PyTorch3D (camera.models.trellis)

Usage:
    # Option 1: Use factory function (recommended)
    pose_finder = create_pose_finder(
        mesh_pth="mesh.glb",
        image_size=512,
        device="cuda",
        pose_model="vggt",        # or "vggt_omega"
        backend="hanyuan"          # or "trellis"
    )

    # Option 2: Use classes directly
    pose_finder = CameraPoseFinder(mesh_pth, image_size, device, backend="trellis")
    pose_finder = Omega_CameraPoseFinder(mesh_pth, image_size, device, 
                                          checkpoint_path="path/to/checkpoint.pt",
                                          backend="trellis")
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


def _get_trellis_renderer():
    """Import TrellisRenderer only when the trellis backend is requested."""
    try:
        from camera.models.trellis import TrellisRenderer
    except ImportError as exc:
        raise ImportError(
            "TRELLIS backend requires trellis2. Install it with "
            "bash bash_scripts/trellis.sh, or use backend='hanyuan'."
        ) from exc
    return TrellisRenderer


def _build_renderer(backend: str, mesh_path: str, device):
    """Return the correct renderer instance for the chosen backend."""
    if backend == "hanyuan":
        return HunyuanRenderer(mesh_path, device)
    elif backend == "trellis":
        return _get_trellis_renderer()(mesh_path, device)
    else:
        raise ValueError(f"Unknown backend '{backend}'. Choose 'hanyuan' or 'trellis'.")


# VGGT imports 
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    vggt_path = os.path.abspath(os.path.join(current_dir, '..', 'vggt'))
    if vggt_path not in sys.path:
        sys.path.insert(0, vggt_path)

    # from vggt.models.vggt import VGGT
    # from vggt.utils.load_fn import load_and_preprocess_images
    # from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    print("Successfully imported VGGT modules.")
except ImportError as e:
    print(f"FATAL: Could not import VGGT modules. Error: {e}")
    sys.exit(1)

# VGGT-Omega imports 
try:
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


class CameraPoseFinder:
    def __init__(
        self,
        mesh_pth: str,
        image_size,
        device,
        dist: float = 3.0,
        fov: float = 40,
        backend: Literal["hanyuan", "trellis"] = "hanyuan",
    ):
        """
        Args:
            mesh_pth: Path to the .glb mesh file.
            image_size: Render resolution — int or (H, W) tuple.
            device: "cuda" or "cpu".
            dist: Default camera distance from origin.
            fov: Field of view in degrees (informational; used by callers).
            backend: Renderer backend — "hanyuan" (HunyuanRenderer) or
                     "trellis" (TrellisRenderer).
        """
        self.original_image_size = image_size if isinstance(image_size, (list, tuple)) else (image_size, image_size)
        self.device = device
        self.dist = dist
        self.fov = fov
        self.backend = backend
        self.mesh_renderer = _build_renderer(backend, mesh_pth, device)
        self.vggt_model = None
        self.known_image_paths = []
        self.known_radii = []

        self.view_path = "./temp_views"
        shutil.rmtree(self.view_path, ignore_errors=True)
        os.makedirs(self.view_path, exist_ok=True)
 
    # ── VGGT load / unload ────────────────────────────────────────────────
 
    def _load_vggt(self):
        if self.vggt_model is None:
            print("Loading VGGT model...")
            self.vggt_model = VGGT.from_pretrained("facebook/VGGT-1B").to(self.device)
            print("VGGT model loaded.")
 
    def _unload_vggt(self):
        if self.vggt_model is not None:
            del self.vggt_model
            self.vggt_model = None
            gc.collect()
            torch.cuda.empty_cache()
            print("VGGT model unloaded.")
 
    # ── Camera matrix helper ──────────────────────────────────────────────
 
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
 
        up_vec = np.cross(forward_vec, right_vec)
 
        R = np.stack([right_vec, -up_vec, forward_vec], axis=0)
        T = -R @ camera_position.reshape(3, 1)
 
        return R, T
 
    # ── Multi-radius view generation ──────────────────────────────────────
 
    def generate_initial_views(self, num_views, img_name,
                               radii=None):
        """
        Generate known views at multiple radii so VGGT has visual context
        at every plausible camera distance.
        """
        if radii is None:
            radii = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]
 
        print(f"Generating initial views across radii {radii} ...")
        self.known_image_paths = []
        self.known_radii = []
 
        azimuths   = np.linspace(0, 360, int(np.sqrt(num_views * 2)), endpoint=False)
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
                        image_size=self.original_image_size, R=r, T=t)
                    rendered_image = rendered_image[0, ..., :3].cpu().numpy()
 
                    path = f'temp_views/{img_name}_{num_of_views}.png'
                    plt.imsave(path, rendered_image)
                    self.known_image_paths.append(path)
                    self.known_radii.append(radius)
                    num_of_views += 1
 
        print(f"Generated {num_of_views} views total ({len(radii)} radii × azimuth × elevation).")
        return all_Rs, all_Ts
 
    # ── VGGT depth head ───────────────────────────────────────────────────
 
    def _get_vggt_depth_estimate(self, aggregated_tokens_list, target_index, images_batch):
        """
        Try to get a distance estimate from VGGT's depth head.
        Returns a float or None if the head is unavailable / returns empty.
        """
        try:
            depth_predictions = self.vggt_model.depth_head(aggregated_tokens_list)
            # expected shape: (1, num_images, H, W)
            target_depth_map = depth_predictions[0, target_index]  # (H, W)
 
            valid_depths = target_depth_map[target_depth_map > 0]
            if len(valid_depths) == 0:
                print("VGGT depth head returned empty map — skipping.")
                return None
 
            estimate = valid_depths.median().item()
            print(f"VGGT depth estimate: {estimate:.3f}")
            return estimate
 
        except Exception as e:
            print(f"VGGT depth head unavailable ({e}) — skipping.")
            return None
 
    # ── Render-and-compare helper ─────────────────────────────────────────
 
    def _render_and_compare(self, target_img_np, R_np, dist):
        """
        Render the mesh from rotation R_np at the given distance and
        return the MSE against the target image (values in [0, 1]).
        """
        # recompute T so the camera sits at `dist` along its current direction
        cam_pos = -R_np.T @ np.zeros((3, 1))          # world origin in cam space = 0
        cam_pos_world = -R_np.T @ np.zeros((3,))       # camera sits at origin direction
        # direction from mesh to camera (world space)
        forward = -R_np[2, :]                           # third row of R is forward axis
        cam_pos_at_dist = forward * (-dist)             # place camera behind the mesh
        T_new = -R_np @ cam_pos_at_dist.reshape(3, 1)
 
        rendered = self.mesh_renderer.render(
            image_size=self.original_image_size, R=R_np, T=T_new)
        rendered_np = rendered[0, ..., :3].cpu().numpy()
 
        # resize target to rendered size if needed
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
 
    # ── Binary search for best distance ───────────────────────────────────
 
    def _binary_search_distance(self, target_img_np, R_np, low, high, iters=6):
        """
        Binary search over camera distance.
        At each iteration, probe midpoint ± delta and shrink toward the
        side with lower MSE against the target image.
        """
        print(f"Binary search for distance in [{low:.2f}, {high:.2f}]")
 
        for i in range(iters):
            mid   = (low + high) / 2.0
            delta = (high - low) * 0.1   # 10 % of current range as probe offset
 
            score_left  = self._render_and_compare(target_img_np, R_np, mid - delta)
            score_right = self._render_and_compare(target_img_np, R_np, mid + delta)
 
            if score_left < score_right:
                high = mid   # best distance is in the lower half
            else:
                low  = mid   # best distance is in the upper half
 
            print(f"  iter {i + 1}: range=[{low:.3f}, {high:.3f}]  "
                  f"score_left={score_left:.4f}  score_right={score_right:.4f}")
 
        best = (low + high) / 2.0
        print(f"Binary search result: {best:.3f}")
        return best
 
    # ── Main pose estimation ──────────────────────────────────────────────
 
    def get_vggt_initial_guess(self, target_image_path, all_Rs, all_Ts, img_name):
        self._load_vggt()
 
        all_image_paths = self.known_image_paths + [target_image_path]
        target_index    = len(all_image_paths) - 1
 
        images = load_and_preprocess_images(all_image_paths).to(self.device)
 
        print("Running VGGT for initial pose estimation...")
        with torch.no_grad():
            dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
            with torch.cuda.amp.autocast(dtype=dtype):
                images_batch = images.unsqueeze(0)
                aggregated_tokens_list, _ = self.vggt_model.aggregator(images_batch)
 
            pose_enc = self.vggt_model.camera_head(aggregated_tokens_list)[-1]
            extrinsic, intrinsic = pose_encoding_to_extri_intri(
                pose_enc, images_batch.shape[-2:])
 
            # ── STEP 1: try VGGT depth head ───────────────────────────────
            vggt_depth_estimate = self._get_vggt_depth_estimate(
                aggregated_tokens_list, target_index, images_batch)
 
        # ── STEP 2: find closest known view by rotation similarity ────────
        target_pose = extrinsic[0][target_index]   # (3, 4)
        best_known_idx  = None
        best_similarity = float('inf')
 
        for i in range(len(self.known_image_paths)):
            known_pose = extrinsic[0][i]
            # compare only the rotation block (first 3 columns)
            diff = torch.norm(target_pose[:, :3] - known_pose[:, :3]).item()
            if diff < best_similarity:
                best_similarity = diff
                best_known_idx  = i
 
        sampling_radius = self.known_radii[best_known_idx]
        print(f"Closest known view radius: {sampling_radius:.2f}  "
              f"(pose diff={best_similarity:.4f})")
 
        # ── STEP 3: combine depth estimate + sampling radius ──────────────
        if vggt_depth_estimate is not None:
            # weight sampling radius more (0.6) because VGGT depth can be noisy
            initial_dist_estimate = 0.6 * sampling_radius + 0.4 * vggt_depth_estimate
            search_margin = 0.8
        else:
            initial_dist_estimate = sampling_radius
            search_margin = 1.2   # wider range since we have less information
 
        search_low  = max(0.5, initial_dist_estimate - search_margin)
        search_high = initial_dist_estimate + search_margin
        print(f"Initial distance estimate: {initial_dist_estimate:.3f}  "
              f"search range: [{search_low:.2f}, {search_high:.2f}]")
 
        # ── STEP 4: align rotation using first known view as reference ─────
        R_reference = all_Rs[0]
        M_reference = torch.eye(4, device=self.device)
        M_reference[0:3, :3] = R_reference
 
        # extract target's aligned rotation
        M_vggt_target = torch.eye(4, device=self.device)
        M_vggt_target[0:3, :] = extrinsic[0][target_index]
        M_aligned_target = torch.matmul(M_vggt_target, M_reference)
        R_target_np = M_aligned_target[0:3, :3].cpu().numpy()
 
        # ── STEP 5: binary search for best distance ────────────────────────
        target_img_np = np.array(
            Image.open(target_image_path).convert("RGB").resize(self.original_image_size)
        ) / 255.0
 
        best_dist = self._binary_search_distance(
            target_img_np, R_target_np, search_low, search_high, iters=6)
 
        # ── STEP 6: build all final extrinsics ─────────────────────────────
        extrinsics_new = []
        for i in range(extrinsic.shape[1]):
            M_vggt = torch.eye(4, device=self.device)
            M_vggt[0:3, :] = extrinsic[0][i]
 
            M_aligned = torch.matmul(M_vggt, M_reference)
 
            if i == target_index:
                # use our carefully estimated distance for the target
                M_aligned[0:3, 3] = torch.tensor([0, 0, best_dist],
                                                  dtype=torch.float32,
                                                  device=self.device)
            else:
                # for known views use their actual sampling radius
                radius_i = self.known_radii[i] if i < len(self.known_radii) else self.dist
                M_aligned[0:3, 3] = torch.tensor([0, 0, radius_i],
                                                  dtype=torch.float32,
                                                  device=self.device)
 
            extrinsics_new.append(M_aligned)
 
        extrinsics_new = torch.stack(extrinsics_new, dim=0).to(self.device)
 
        target_extrinsic = extrinsics_new[target_index]
        target_intrinsic = intrinsic[0, target_index]
 
        # ── debug renders ──────────────────────────────────────────────────
        os.makedirs("./temp_views2", exist_ok=True)
        for i in range(len(all_image_paths)):
            rendered_image = self.mesh_renderer.render(
                image_size=self.original_image_size,
                R=extrinsics_new[i][0:3, :3].cpu().numpy(),
                T=extrinsics_new[i][0:3, 3].cpu().numpy())
            rendered_image = rendered_image[0, ..., :3].cpu().numpy()
            plt.imsave(f'temp_views2/{img_name}_{i}.png', rendered_image)
 
        self._unload_vggt()
 
        return target_extrinsic, target_intrinsic
 
class Omega_CameraPoseFinder:
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

    # ── VGGT-Omega load / unload ──────────────────────────────────────────

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

    # ── Camera matrix helper ──────────────────────────────────────────────

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

        up_vec = np.cross(forward_vec, right_vec)

        R = np.stack([right_vec, -up_vec, forward_vec], axis=0)
        T = -R @ camera_position.reshape(3, 1)

        return R, T

    # ── Multi-radius view generation ─────────────────────────────────────

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

    # ── VGGT-Omega depth helper ──────────────────────────────────────────

    def _get_omega_depth_estimate(self, predictions, target_index):
        """
        Use VGGT-Omega's predicted depth map as a coarse distance estimate.
        """
        try:
            depth_predictions = predictions["depth"]  # [B, S, H, W, 1] in the demo-style API
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

    # ── Render-and-compare helper ────────────────────────────────────────

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

    # ── Binary search for best distance ──────────────────────────────────

    def _binary_search_distance(self, target_img_np, R_np, low, high, iters=6):
        """
        Binary search over camera distance.
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

    # ── Main pose estimation ─────────────────────────────────────────────

    def get_vggt_initial_guess(self, target_image_path, all_Rs, all_Ts, img_name):
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

            # Step 1: try Omega depth
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


# choosing between VGGT and VGGT-Omega
def create_pose_finder(
    mesh_pth: str,
    image_size,
    device,
    pose_model: Literal["vggt", "vggt_omega"] = "vggt",
    backend: Literal["hanyuan", "trellis"] = "hanyuan",
    dist: float = 3.0,
    fov: float = 40,
    # VGGT-Omega specific parameters (only used when pose_model="vggt_omega")
    checkpoint_path: str = "path/to/vggt_omega_1b_512.pt",
    model_image_resolution: int = 512,
    enable_alignment: bool = False,
):
    """
    Factory function to create the appropriate pose finder based on the pose model.

    Args:
        mesh_pth: Path to the .glb mesh file.
        image_size: Render resolution — int or (H, W) tuple.
        device: "cuda" or "cpu".
        pose_model: Which pose estimation model to use:
                    - "vggt": Facebook's VGGT-1B (faster, standard)
                    - "vggt_omega": Facebook's VGGT-Omega (advanced, needs checkpoint)
        backend: Renderer backend:
                 - "hanyuan": Hunyuan3D + PyTorch3D
                 - "trellis": TRELLIS.2 + PyTorch3D
        dist: Default camera distance from origin.
        fov: Field of view in degrees.
        checkpoint_path: Path to VGGT-Omega checkpoint (only for pose_model="vggt_omega").
        model_image_resolution: Image resolution for VGGT-Omega (only for pose_model="vggt_omega").
        enable_alignment: Enable VGGT-Omega alignment module (only for pose_model="vggt_omega").

    Returns:
        CameraPoseFinder or Omega_CameraPoseFinder instance.

    Examples:
        # Use VGGT with Hunyuan renderer
        finder = create_pose_finder(
            mesh_pth="mesh.glb",
            image_size=512,
            device="cuda",
            pose_model="vggt",
            backend="hanyuan"
        )

        # Use VGGT-Omega with TRELLIS renderer
        finder = create_pose_finder(
            mesh_pth="mesh.glb",
            image_size=512,
            device="cuda",
            pose_model="vggt_omega",
            backend="trellis",
            checkpoint_path="./checkpoints/vggt_omega_1b_512.pt"
        )
    """
    if pose_model == "vggt":
        return CameraPoseFinder(
            mesh_pth=mesh_pth,
            image_size=image_size,
            device=device,
            dist=dist,
            fov=fov,
            backend=backend,
        )
    elif pose_model == "vggt_omega":
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
    else:
        raise ValueError(
            f"Unknown pose_model '{pose_model}'. Choose 'vggt' or 'vggt_omega'."
        )

# Relative pose transfer
def compute_new_pose_from_relative(view_matrix_B_with_pose_A, view_matrix_B, view_matrix_B_prime):
    """
    Given:
        V_B_with_pose_A  — camera pose used to photograph horse B from angle A
        V_B              — camera pose used to photograph horse B from its reference angle
        V_B_prime        — camera pose used to photograph the jumping horse
 
    Returns:
        V_B_prime_new    — camera pose to render the jumping horse from angle B
    """
    V_B_with_pose_A = view_matrix_B_with_pose_A.cpu().numpy()
    V_B             = view_matrix_B.cpu().numpy()
    V_B_prime       = view_matrix_B_prime.cpu().numpy()
 
    try:
        V_B_with_pose_A_inv = np.linalg.inv(V_B_with_pose_A)
    except np.linalg.LinAlgError:
        print("Error: view_matrix_B_with_pose_A is not invertible.")
        return None
 
    # delta = relative transform from pose-A viewpoint to pose-B viewpoint
    delta_transform = V_B @ V_B_with_pose_A_inv
 
    # apply the same delta to B'
    V_B_prime_new = delta_transform @ V_B_prime
 
    return V_B_prime_new
