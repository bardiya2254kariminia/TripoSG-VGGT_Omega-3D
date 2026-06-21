"""
find_camera.py  —  two-stage camera pose finder for a single image + mesh.

Stage 1 – VGGT-Omega (edge-map world)
    Render 200 sphere views as Canny edge maps, feed them + the target image
    (also edge-mapped) to VGGT-Omega.  This reliably recovers the camera
    ELEVATION and DISTANCE (these change the silhouette shape a lot), but
    can be wrong on AZIMUTH (front/back/side look similar in edges).

Stage 2 – DINOv2 azimuth sweep (RGB world)
    Fix the VGGT elevation + distance.  Sweep azimuth 0–360° in coarse steps,
    render each in RGB, score against the real target photo with DINOv2.
    Then refine around the winner with fine steps.
    DINOv2 has no azimuth ambiguity — it clearly sees "face" vs "back" vs "side".

Usage
-----
    python find_camera.py --image /path/to/photo.png --mesh /path/to/mesh.glb

    # optional
    --output          output path  (default: ./rendered_<stem>.png)
    --size            render resolution (default 512)
    --dist            camera distance from origin (default 3.0)
    --num_views       sphere views for VGGT-Omega context (default 200)
    --sweep_az_step   coarse azimuth step in degrees (default 10)
    --sweep_el_range  ± elevation variation around VGGT estimate in degrees (default 15)
    --sweep_el_step   elevation step in the sweep (default 5)
    --fine_az_step    fine azimuth step in degrees (default 2)
    --fine_range      ± window for fine search (default 15)
    --checkpoint      VGGT-Omega checkpoint path (auto-detected if omitted)
"""

import argparse
import gc
import os
import sys
import shutil

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms

from pytorch3d.io import IO
from pytorch3d.io.experimental_gltf_io import MeshGlbFormat
from pytorch3d.renderer import (
    BlendParams, MeshRasterizer, MeshRenderer,
    PerspectiveCameras, PointLights, RasterizationSettings, SoftPhongShader,
)

# ── VGGT-Omega path ────────────────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_OMEGA_PATH = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "vggt-omega"))
if _OMEGA_PATH not in sys.path:
    sys.path.insert(0, _OMEGA_PATH)

try:
    from vggt_omega.models import VGGTOmega
    from vggt_omega.utils.load_fn import load_and_preprocess_images as _load_imgs
    from vggt_omega.utils.pose_enc import encoding_to_camera
except ImportError as e:
    sys.exit(f"FATAL: could not import VGGT-Omega — {e}")

_DEFAULT_CKPT = os.path.join(
    _SCRIPT_DIR, "checkpoints", "vggt-omega", "vggt_omega_1b_512.pt"
)


# ─────────────────────────────────────────────────────────────────────────────
# Edge-map helpers
# ─────────────────────────────────────────────────────────────────────────────

def _edge_from_rgba(rgba: np.ndarray, low=50, high=150) -> np.ndarray:
    rgb   = (rgba[..., :3] * 255).clip(0, 255).astype(np.uint8)
    mask  = (rgba[..., 3] > 0.1).astype(np.uint8)
    gray  = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, low, high)
    edges = (edges * mask).astype(np.float32) / 255.0
    return np.stack([edges, edges, edges], axis=-1)


def _edge_from_file(path: str, size: tuple, low=50, high=150) -> np.ndarray:
    img   = Image.open(path).convert("RGBA").resize(size)
    arr   = np.array(img)
    mask  = (arr[..., 3] > 128).astype(np.uint8)
    gray  = cv2.cvtColor(arr[..., :3], cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, low, high)
    edges = (edges * mask).astype(np.float32) / 255.0
    return np.stack([edges, edges, edges], axis=-1)


# ─────────────────────────────────────────────────────────────────────────────
# Camera geometry
# ─────────────────────────────────────────────────────────────────────────────

def _opencv_RT(az_deg: float, el_deg: float, dist: float):
    az, el = np.deg2rad(az_deg), np.deg2rad(el_deg)
    pos = np.array([
        dist * np.cos(el) * np.sin(az),
        dist * np.sin(el),
        dist * np.cos(el) * np.cos(az),
    ])
    fwd = -pos / np.linalg.norm(pos)
    up  = np.array([0.0, 1.0, 0.0])
    rgt = np.cross(up, fwd)
    if np.linalg.norm(rgt) < 1e-6:
        up  = np.array([1.0, 0.0, 0.0])
        rgt = np.cross(up, fwd)
    rgt /= np.linalg.norm(rgt)
    up2  = np.cross(fwd, rgt)
    R    = np.stack([rgt, -up2, fwd], axis=0)
    T    = (-R @ pos).flatten()
    return R, T


def _extract_el_dist(R_np: np.ndarray, T_np: np.ndarray):
    """Recover camera elevation (deg) and distance from an OpenCV [R|T]."""
    if R_np.ndim == 3:
        R_np = R_np[0]
    cam_pos = -R_np.T @ T_np.flatten()
    dist    = float(np.linalg.norm(cam_pos))
    el_deg  = float(np.rad2deg(np.arcsin(
        np.clip(cam_pos[1] / (dist + 1e-8), -1.0, 1.0)
    )))
    az_deg  = float(np.rad2deg(np.arctan2(cam_pos[0], cam_pos[2])))
    return az_deg, el_deg, dist


# ─────────────────────────────────────────────────────────────────────────────
# Renderer
# ─────────────────────────────────────────────────────────────────────────────

class Renderer:
    def __init__(self, mesh_path: str, device: str):
        self.device = device
        io = IO()
        io.register_meshes_format(MeshGlbFormat())
        self.mesh = io.load_mesh(mesh_path, include_textures=True).to(device)

    def render(self, image_size, R: np.ndarray, T: np.ndarray) -> np.ndarray:
        H, W = image_size if isinstance(image_size, tuple) else (image_size, image_size)
        R_t  = torch.from_numpy(R).float().to(self.device).unsqueeze(0)
        T_t  = torch.from_numpy(T).float().to(self.device).unsqueeze(0)

        # Compute camera world position: cam_pos = -R^T @ T
        cam_pos = -R.T @ T.reshape(3)
        
        # Place light to the right and slightly above the camera (camera-relative)
        # In OpenCV convention: right = R[0,:], up = -R[1,:]
        right_vec = R[0, :]
        up_vec = -R[1, :]
        
        # Light offset: 1.0 units right, 0.5 units up from camera
        light_pos = cam_pos + 1.0 * right_vec + 0.5 * up_vec
        light_pos = light_pos.tolist()

        R_p3d = R_t.clone().permute(0, 2, 1)
        T_p3d = T_t.clone()
        R_p3d[:, :, :2] *= -1
        T_p3d[:, :2]    *= -1

        fl = torch.tensor([[float(W), float(H)]], device=self.device)
        pp = torch.tensor([[W / 2.0, H / 2.0]], device=self.device)
        cameras = PerspectiveCameras(
            device=self.device, R=R_p3d, T=T_p3d,
            focal_length=fl, principal_point=pp,
            image_size=((H, W),), in_ndc=False,
        )
        raster = RasterizationSettings(image_size=(H, W), blur_radius=0.0, faces_per_pixel=1)
        lights = PointLights(device=self.device, location=[light_pos])
        shader = SoftPhongShader(
            device=self.device, cameras=cameras, lights=lights,
            blend_params=BlendParams(background_color=(1.0, 1.0, 1.0)),
        )
        return MeshRenderer(
            rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster),
            shader=shader,
        )(self.mesh)[0].cpu().numpy()   # (H, W, 4)


# ─────────────────────────────────────────────────────────────────────────────
# DINOv2 scorer
# ─────────────────────────────────────────────────────────────────────────────

class DinoScorer:
    def __init__(self, device: str):
        self.device = device
        print("  Loading DINOv2 ViT-S/14 …")
        self.model = torch.hub.load(
            'facebookresearch/dinov2', 'dinov2_vits14', verbose=False
        ).to(device).eval()
        self.tf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        print("  DINOv2 loaded.")

    @torch.inference_mode()
    def feat(self, pil_img: Image.Image) -> torch.Tensor:
        x = self.tf(pil_img).unsqueeze(0).to(self.device)
        return F.normalize(self.model(x), dim=-1)   # (1, D)

    @torch.inference_mode()
    def feat_batch(self, pil_imgs, batch=16) -> torch.Tensor:
        feats = []
        for i in range(0, len(pil_imgs), batch):
            xs = torch.stack([self.tf(im) for im in pil_imgs[i:i+batch]]).to(self.device)
            feats.append(F.normalize(self.model(xs), dim=-1))
        return torch.cat(feats, dim=0)   # (N, D)

    def unload(self):
        del self.model, self.tf
        gc.collect(); torch.cuda.empty_cache()


# ─────────────────────────────────────────────────────────────────────────────
# Main finder
# ─────────────────────────────────────────────────────────────────────────────

class CameraFinder:
    def __init__(self, mesh_path: str, image_size, device: str,
                 dist: float = 3.0, num_views: int = 200,
                 checkpoint: str = _DEFAULT_CKPT):
        self.device   = device
        self.dist     = dist
        self.size     = image_size if isinstance(image_size, tuple) else (image_size, image_size)
        self.renderer = Renderer(mesh_path, device)
        self.ckpt     = checkpoint
        self.num_views = num_views
        self._vggt    = None
        self._tmp     = os.path.join(_SCRIPT_DIR, "_temp_views")
        self._tmp2    = os.path.join(_SCRIPT_DIR, "_temp2_views")
        shutil.rmtree(self._tmp, ignore_errors=True)
        shutil.rmtree(self._tmp2, ignore_errors=True)
        os.makedirs(self._tmp)
        os.makedirs(self._tmp2)

    # ── VGGT-Omega helpers ────────────────────────────────────────────────────

    def _load_vggt(self):
        if self._vggt is None:
            print("  Loading VGGT-Omega …")
            m  = VGGTOmega(enable_alignment=False).to(self.device)
            sd = torch.load(self.ckpt, map_location="cpu")
            if isinstance(sd, dict) and "state_dict" in sd:
                sd = sd["state_dict"]
            m.load_state_dict(sd)
            m.eval()
            self._vggt = m
            print("  VGGT-Omega loaded.")

    def _unload_vggt(self):
        if self._vggt is not None:
            del self._vggt; self._vggt = None
            gc.collect(); torch.cuda.empty_cache()

    def _build_edge_views(self):
        """Render 200 sphere views as Canny edge maps for VGGT-Omega context."""
        n  = self.num_views
        azs = np.linspace(0, 360, int(np.sqrt(n * 2)), endpoint=False)
        els = np.linspace(-45, 45, int(np.sqrt(n / 2)), endpoint=True)
        paths, Rs, Ts = [], [], []
        idx = 0
        print(f"  Generating {int(np.sqrt(n*2))}×{int(np.sqrt(n/2))} edge-map views …")
        for az in azs:
            for el in els:
                R, T = _opencv_RT(az, el, self.dist)
                rgba = self.renderer.render(self.size, R, T)
                edge = _edge_from_rgba(rgba)
                p = os.path.join(self._tmp, f"edge_{idx}.png")
                plt.imsave(p, edge)
                paths.append(p)
                Rs.append(torch.from_numpy(R))
                Ts.append(T)
                idx += 1
        print(f"  Generated {idx} edge views.")
        return paths, Rs, Ts

    def _run_vggt(self, image_path: str, view_paths: list, Rs: list) -> tuple:
        """
        Run VGGT-Omega → return (az_deg, el_deg, dist) of the estimated camera.
        """
        self._load_vggt()

        target_edge = _edge_from_file(image_path, self.size)
        tgt_path    = os.path.join(self._tmp, "_target_edge.png")
        plt.imsave(tgt_path, target_edge)

        all_paths  = view_paths + [tgt_path]
        target_idx = len(all_paths) - 1

        images = _load_imgs(all_paths).to(self.device)

        print("  Running VGGT-Omega …")
        with torch.inference_mode():
            pred = self._vggt(images)

        pose_enc = pred["pose_enc"]
        extrinsic, _ = encoding_to_camera(pose_enc, images.shape[-2:])

        R0    = Rs[0].to(self.device)
        M_ref = torch.eye(4, device=self.device)
        M_ref[:3, :3] = R0

        ext_list = []
        for i in range(extrinsic.shape[1]):
            M = torch.eye(4, device=self.device)
            M[:3, :] = extrinsic[0][i]
            M_aligned = torch.matmul(M, M_ref)
            M_aligned[:3, 3] = torch.tensor([0.0, 0.0, self.dist], device=self.device)
            ext_list.append(M_aligned)

        ext    = torch.stack(ext_list)[target_idx]
        R_init = ext[:3, :3].cpu().numpy()
        T_init = ext[:3, 3].cpu().numpy()

        self._unload_vggt()

        az_vggt, el_vggt, dist_vggt = _extract_el_dist(R_init, T_init)
        print(f"  VGGT-Omega estimate → az={az_vggt:.1f}°  el={el_vggt:.1f}°  dist={dist_vggt:.2f}")
        return az_vggt, el_vggt, dist_vggt

    # ── DINOv2 azimuth sweep ──────────────────────────────────────────────────

    def _dino_best_on_grid(self, grid: list, target_feat: torch.Tensor,
                           dino: DinoScorer, label: str):
        """
        Render mesh at every (az, el) in grid, score with DINOv2, return best.
        Returns (best_az, best_el, best_score, best_R, best_T).
        """
        renders, params = [], []
        print(f"  [{label}] rendering {len(grid)} RGB views …", flush=True)
        for az, el in grid:
            R, T  = _opencv_RT(az % 360, el, self.dist)
            rgba  = self.renderer.render(self.size, R, T)
            rgb   = (rgba[..., :3] * 255).clip(0, 255).astype(np.uint8)
            renders.append(Image.fromarray(rgb))
            params.append((az % 360, el, R, T))

        print(f"  [{label}] scoring with DINOv2 …", flush=True)
        feats  = dino.feat_batch(renders)                      # (N, D)
        scores = (feats @ target_feat.T).squeeze(-1)           # (N,)
        
        # Save all renders with scores overlaid
        print(f"  [{label}] saving {len(renders)} views with scores to _temp2_views/ …", flush=True)
        for idx, (img, score, (az, el, _, _)) in enumerate(zip(renders, scores, params)):
            # Create a copy to draw on
            img_annotated = img.copy()
            draw = ImageDraw.Draw(img_annotated)
            
            # Try to use a decent font, fallback to default
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
            except:
                font = ImageFont.load_default()
            
            # Prepare text
            text = f"az={az:.1f}° el={el:.1f}°\nDINO={score.item():.4f}"
            
            # Draw text with background for readability
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            # Position at top-left with padding
            x, y = 10, 10
            draw.rectangle([x-5, y-5, x+text_width+5, y+text_height+5], fill=(0, 0, 0, 180))
            draw.text((x, y), text, fill=(255, 255, 0), font=font)
            
            # Save
            save_path = os.path.join(self._tmp2, f"{label}_{idx:04d}_az{az:.1f}_el{el:.1f}_s{score.item():.4f}.png")
            img_annotated.save(save_path)
        
        bi     = scores.argmax().item()
        az_b, el_b, R_b, T_b = params[bi]
        print(f"  [{label}] best  az={az_b:.1f}°  el={el_b:.1f}°  "
              f"DINOv2={scores[bi].item():.4f}")
        return az_b, el_b, scores[bi].item(), R_b, T_b

    # ── public API ────────────────────────────────────────────────────────────

    def find_and_render(self, image_path: str, output_path: str,
                        sweep_az_step: float = 10.0,
                        sweep_el_range: float = 15.0,
                        sweep_el_step: float = 5.0,
                        fine_az_step: float = 2.0,
                        fine_range: float = 15.0):

        # ── Stage 1: VGGT-Omega (edge-map world) → elevation + initial az ────
        print("\n" + "="*60)
        print("STAGE 1 — VGGT-Omega edge-map pose estimation")
        print("="*60)
        view_paths, Rs, Ts = self._build_edge_views()
        az_vggt, el_vggt, _ = self._run_vggt(image_path, view_paths, Rs)

        # ── Stage 2a: coarse azimuth sweep (RGB + DINOv2) ────────────────────
        print("\n" + "="*60)
        print("STAGE 2 — DINOv2 azimuth sweep (RGB world)")
        print("="*60)

        dino        = DinoScorer(self.device)
        target_feat = dino.feat(Image.open(image_path).convert("RGB"))

        # Full 0–360° azimuth sweep at VGGT elevation ± sweep_el_range
        az_steps = np.arange(0, 360, sweep_az_step)
        el_steps = np.arange(
            el_vggt - sweep_el_range,
            el_vggt + sweep_el_range + 0.01,
            sweep_el_step,
        )
        el_steps = np.clip(el_steps, -60, 60)
        coarse_grid = [(az, el) for az in az_steps for el in el_steps]

        print(f"  Coarse sweep: az every {sweep_az_step}° × "
              f"el [{el_steps[0]:.0f}°..{el_steps[-1]:.0f}°] = {len(coarse_grid)} candidates")
        az_c, el_c, score_c, R_c, T_c = self._dino_best_on_grid(
            coarse_grid, target_feat, dino, "coarse")

        # ── Stage 2b: fine azimuth refinement ────────────────────────────────
        az_fine = np.arange(az_c - fine_range, az_c + fine_range + 0.01, fine_az_step)
        el_fine = np.arange(el_c - fine_range / 2, el_c + fine_range / 2 + 0.01, fine_az_step)
        el_fine = np.clip(el_fine, -60, 60)
        fine_grid = [(az, el) for az in az_fine for el in el_fine]

        print(f"  Fine refinement: ±{fine_range}° at {fine_az_step}°/step "
              f"= {len(fine_grid)} candidates")
        az_f, el_f, score_f, R_f, T_f = self._dino_best_on_grid(
            fine_grid, target_feat, dino, "fine")

        dino.unload()

        # ── pick winner ───────────────────────────────────────────────────────
        if score_f >= score_c:
            R_best, T_best, az_best, el_best = R_f, T_f, az_f, el_f
            best_score = score_f
        else:
            R_best, T_best, az_best, el_best = R_c, T_c, az_c, el_c
            best_score = score_c

        print(f"\n{'='*60}")
        print(f"RESULT: az={az_best:.1f}°  el={el_best:.1f}°  DINOv2={best_score:.4f}")
        print(f"  R:\n{R_best}")
        print(f"  T: {T_best}")
        print(f"{'='*60}")

        rgba = self.renderer.render(self.size, R_best, T_best)
        plt.imsave(output_path, rgba[..., :3])
        print(f"\nSaved → {output_path}")
        print(f"Edge-map views saved in: {self._tmp}")
        print(f"DINOv2 scored views saved in: {self._tmp2}")

        # Keep temp files for inspection (comment out to auto-cleanup)
        # shutil.rmtree(self._tmp, ignore_errors=True)
        # shutil.rmtree(self._tmp2, ignore_errors=True)
        return R_best, T_best


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Two-stage camera finder: VGGT-Omega (edge) + DINOv2 azimuth sweep (RGB).")
    p.add_argument("--image",          required=True)
    p.add_argument("--mesh",           required=True)
    p.add_argument("--output",         default=None)
    p.add_argument("--size",           type=int,   default=512)
    p.add_argument("--dist",           type=float, default=3.0)
    p.add_argument("--num_views",      type=int,   default=200,
                   help="VGGT-Omega context views (default 200)")
    p.add_argument("--sweep_az_step",  type=float, default=10.0,
                   help="Coarse azimuth step in degrees (default 10)")
    p.add_argument("--sweep_el_range", type=float, default=15.0,
                   help="± elevation around VGGT estimate (default 15)")
    p.add_argument("--sweep_el_step",  type=float, default=5.0,
                   help="Elevation step in sweep (default 5)")
    p.add_argument("--fine_az_step",   type=float, default=10.0,
                   help="Fine azimuth step in degrees (default 2)")
    p.add_argument("--fine_range",     type=float, default=30.0,
                   help="± window for fine search (default 15)")
    p.add_argument("--checkpoint",     default=_DEFAULT_CKPT)
    args = p.parse_args()

    for label, path in [("image", args.image), ("mesh", args.mesh)]:
        if not os.path.isfile(path):
            sys.exit(f"Error: {label} not found: {path}")
    if not os.path.isfile(args.checkpoint):
        sys.exit(f"Error: checkpoint not found: {args.checkpoint}\n"
                 f"Pass --checkpoint /path/to/vggt_omega_1b_512.pt")

    if args.output is None:
        stem = os.path.splitext(os.path.basename(args.image))[0]
        args.output = os.path.join(os.getcwd(), f"rendered_{stem}.png")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device     : {device}")
    print(f"Image      : {args.image}")
    print(f"Mesh       : {args.mesh}")
    print(f"Output     : {args.output}")

    finder = CameraFinder(
        mesh_path  = args.mesh,
        image_size = args.size,
        device     = device,
        dist       = args.dist,
        num_views  = args.num_views,
        checkpoint = args.checkpoint,
    )
    finder.find_and_render(
        image_path     = args.image,
        output_path    = args.output,
        sweep_az_step  = args.sweep_az_step,
        sweep_el_range = args.sweep_el_range,
        sweep_el_step  = args.sweep_el_step,
        fine_az_step   = args.fine_az_step,
        fine_range     = args.fine_range,
    )


if __name__ == "__main__":
    main()
