"""
Hunyuan3D mesh generation/rendering and TripoSG texturizing.

Classes:
  MeshGenerator      — generate 3D meshes from images via Hunyuan3D-2, with
                       optional TripoSG texturizing backend.
  HunyuanRenderer    — render .glb meshes with PyTorch3D.
  TripoSGTexturizer  — generate complete textured meshes via TripoSG
                       (VAST-AI/TripoSG, diffusers-compatible pipeline).

TripoSG install:
    pip install git+https://github.com/VAST-AI-Research/TripoSG.git
Weights are auto-downloaded from HuggingFace (VAST-AI/TripoSG) on first use.
"""

import gc
import os,sys
import numpy as np
import torch
from PIL import Image

from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
from hy3dgen.texgen import Hunyuan3DPaintPipeline
from pytorch3d.io import IO
from pytorch3d.io.experimental_gltf_io import MeshGlbFormat
from pytorch3d.renderer import (
    BlendParams,
    MeshRasterizer,
    MeshRenderer,
    PerspectiveCameras,
    PointLights,
    RasterizationSettings,
    SoftPhongShader,
)


# ── TripoSGTexturizer ─────────────────────────────────────────────────────────

class TripoSGTexturizer:
    """
    Generates 3D meshes from a single image using TripoSG.

    TripoSG (VAST-AI/TripoSG) is a rectified-flow image-to-3D model that
    produces a watertight mesh directly from an input photograph.  When used
    as the texturizing backend for MeshGenerator, the returned mesh replaces
    the Hunyuan3D paint step — TripoSG produces geometry and per-vertex colour
    in one forward pass.

    The pipeline class ``TripoSGPipeline`` lives in the ``triposg`` package
    (part of the TripoSG GitHub repo) and follows the diffusers interface.

    Install::

        pip install git+https://github.com/VAST-AI-Research/TripoSG.git

    Weights are downloaded automatically from ``VAST-AI/TripoSG`` on
    HuggingFace the first time :meth:`load` is called (requires ~8 GB VRAM,
    ~5 GB disk).

    Example::

        tex = TripoSGTexturizer(device="cuda")
        mesh = tex.generate(Image.open("object.png"))
        mesh.export("object_triposg.glb")
    """

    HF_MODEL_ID = "VAST-AI/TripoSG"
    # Default weights location mirrors where the official TripoSG inference
    # script auto-downloads to: <triposg-repo>/pretrained_weights/TripoSG.
    # The TripoSGTexturizer resolves this relative to the cloned repo at runtime.
    DEFAULT_WEIGHTS_SUBDIR = "pretrained_weights/TripoSG"

    def __init__(
        self,
        model_path: str = None,
        device: str = "cuda",
        dtype: torch.dtype = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        seed: int = 42,
    ):
        """
        Args:
            model_path: Path to the TripoSG weights directory.  If ``None``,
                        the default location used by the official TripoSG repo
                        (``<triposg-repo>/pretrained_weights/TripoSG``) is tried
                        first; if absent, weights are auto-downloaded there by
                        the pipeline on first call.
            device: PyTorch device string ("cuda" or "cpu").
            dtype: Model precision.  Defaults to bfloat16 on Ampere+ GPUs,
                   float16 elsewhere, float32 on CPU.
            num_inference_steps: Diffusion denoising steps (50 is a good
                                 balance between speed and quality).
            guidance_scale: Classifier-free guidance scale (7.5 default).
            seed: RNG seed for reproducibility.
        """
        self.model_path = model_path  # None → resolved lazily after repo is found
        self.device = device

        if dtype is None:
            if "cuda" in str(device) and torch.cuda.is_available():
                cap = torch.cuda.get_device_capability()
                dtype = torch.bfloat16 if cap[0] >= 8 else torch.float16
            else:
                dtype = torch.float32
        self.dtype = dtype
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.seed = seed
        self._pipe = None

    # ── weight management ────────────────────────────────────────────────

    def _ensure_local_weights(self, triposg_repo_dir: str) -> str:
        """
        Ensure TripoSG weights exist locally and return the local directory path.

        ``TripoSGPipeline.from_pretrained`` MUST receive a local path, not a
        HuggingFace model ID.  Passing the HF ID causes diffusers to validate
        remote module names (e.g. ``triposg.schedulers.*``) against its own
        package, which always fails for TripoSG's custom classes.

        Priority for the local directory:
          1. Explicit ``model_path`` set by the caller.
          2. ``<triposg-repo>/pretrained_weights/TripoSG`` — the same location
             the official TripoSG inference script uses.
          3. ``/workspace/TripoSG-VGGT_Omega-3D/checkpoints/TripoSG`` — fallback.

        If the chosen directory is empty/absent, weights are downloaded from
        ``VAST-AI/TripoSG`` via ``snapshot_download`` before returning.
        """
        from huggingface_hub import snapshot_download

        candidates = []
        if self.model_path:
            candidates.append(self.model_path)
        if triposg_repo_dir:
            candidates.append(os.path.join(triposg_repo_dir, self.DEFAULT_WEIGHTS_SUBDIR))
        candidates.append("/workspace/TripoSG-VGGT_Omega-3D/checkpoints/TripoSG")

        # Use the first candidate that already has files
        for path in candidates:
            if os.path.isdir(path) and os.listdir(path):
                print(f"[TripoSG] Found weights at {path}")
                return path

        # Nothing cached — download to the preferred location
        target = candidates[0]
        os.makedirs(target, exist_ok=True)
        print(f"[TripoSG] Downloading {self.HF_MODEL_ID} → {target} ...")
        snapshot_download(self.HF_MODEL_ID, local_dir=target)
        print("[TripoSG] Download complete.")
        return target

    # ── pipeline lifecycle ────────────────────────────────────────────────

    def load(self):
        """Load the TripoSGPipeline (lazy — called automatically by :meth:`generate`).

        Follows the official TripoSG approach used in the HuggingFace space:
          1. Download weights to a local directory with ``snapshot_download``.
          2. Call ``TripoSGPipeline.from_pretrained(local_dir)``.

        Passing the HF model ID directly to ``from_pretrained`` does NOT work
        because diffusers validates all component class names against its own
        package and rejects TripoSG's custom ``triposg.schedulers.*`` classes.
        """
        if self._pipe is not None:
            return

        # ── locate the cloned repo and extend sys.path if needed ─────────────
        import sys
        _repo_candidates = [
            os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                "triposg-repo",
            ),
            "/workspace/TripoSG-VGGT_Omega-3D/triposg-repo",
        ]
        _triposg_repo = None
        for _cand in _repo_candidates:
            if os.path.isdir(_cand):
                _triposg_repo = _cand
                if _cand not in sys.path:
                    sys.path.insert(0, _cand)
                break

        try:
            from triposg.pipelines.pipeline_triposg import TripoSGPipeline
        except ImportError as exc:
            raise ImportError(
                "triposg is not importable.  Run final_setup.sh to clone the repo and "
                "set PYTHONPATH, or manually:\n"
                "  git clone https://github.com/VAST-AI-Research/TripoSG.git triposg-repo\n"
                "  export PYTHONPATH=triposg-repo:$PYTHONPATH"
            ) from exc

        # ── ensure weights are local, then load from the local path ──────────
        local_weights = self._ensure_local_weights(_triposg_repo or "")

        print(f"[TripoSG] Loading pipeline from {local_weights} ...")
        self._pipe = TripoSGPipeline.from_pretrained(local_weights)
        self._pipe = self._pipe.to(self.device, self.dtype)
        print("[TripoSG] Pipeline ready.")

    def unload(self):
        """Delete the pipeline and free GPU memory."""
        if self._pipe is not None:
            del self._pipe
            self._pipe = None
            gc.collect()
            if "cuda" in str(self.device) and torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("[TripoSG] Pipeline unloaded.")

    # ── mesh generation ───────────────────────────────────────────────────

    def generate(self, image: Image.Image) -> "trimesh.Trimesh":
        """
        Generate a 3D mesh from a single PIL image.

        RGBA images are composited onto a white background before inference
        (TripoSG expects an RGB input).

        Args:
            image: Input image (RGB or RGBA).

        Returns:
            ``trimesh.Trimesh`` with the generated mesh.  Export to .glb with
            ``mesh.export("output.glb")``.
        """
        import trimesh

        # RGBA → RGB with white background
        if image.mode == "RGBA":
            bg = Image.new("RGB", image.size, (255, 255, 255))
            bg.paste(image, mask=image.split()[3])
            image = bg
        else:
            image = image.convert("RGB")

        self.load()

        generator = torch.Generator(device=self._pipe.device).manual_seed(self.seed)
        outputs = self._pipe(
            image=image,
            generator=generator,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
        ).samples[0]

        vertices = outputs[0].astype(np.float32)
        faces = np.ascontiguousarray(outputs[1])

        # Include vertex colours if the pipeline provides them
        if len(outputs) > 2 and outputs[2] is not None:
            vertex_colors = outputs[2]
            mesh = trimesh.Trimesh(vertices, faces, vertex_colors=vertex_colors)
        else:
            mesh = trimesh.Trimesh(vertices, faces)

        return mesh

    def generate_from_path(self, image_path: str) -> "trimesh.Trimesh":
        """
        Convenience wrapper — loads the image from *image_path* and calls
        :meth:`generate`.

        Args:
            image_path: Path to the input image file.

        Returns:
            ``trimesh.Trimesh`` with the generated mesh.
        """
        image = Image.open(image_path).convert("RGBA")
        return self.generate(image)


# ── MeshGenerator ─────────────────────────────────────────────────────────────

class MeshGenerator:
    """
    Generates 3D meshes from 2D images.

    Two texturizing backends are supported:

    * ``"hunyuan"`` (default) — Hunyuan3D-2 two-step pipeline:
      :meth:`generate_mesh` creates the shape, then
      :meth:`generate_painted_mesh` applies texture.

    * ``"triposg"`` — TripoSG one-step pipeline:
      :meth:`generate_painted_mesh` runs TripoSG on the original image and
      returns a fully textured mesh (calling :meth:`generate_mesh` first is
      not required when using this backend).
    """

    def __init__(
        self,
        image_path: str,
        pipeline_flow=None,
        pipeline_paint=None,
        texture_backend: str = "hunyuan",
        triposg_model_path: str = None,
        triposg_num_steps: int = 50,
        triposg_guidance_scale: float = 7.5,
        triposg_seed: int = 42,
    ):
        """
        Args:
            image_path: Path to the input image.
            pipeline_flow: Pre-loaded ``Hunyuan3DDiTFlowMatchingPipeline``.
                           Loaded lazily from ``tencent/Hunyuan3D-2`` if
                           ``None`` and ``texture_backend != "triposg"``.
            pipeline_paint: Pre-loaded ``Hunyuan3DPaintPipeline``.
                            Loaded lazily if ``None`` and
                            ``texture_backend == "hunyuan"``.
            texture_backend: ``"hunyuan"`` or ``"triposg"``.
            triposg_model_path: Local path to TripoSG weights.  ``None``
                                triggers auto-download from HuggingFace.
            triposg_num_steps: TripoSG denoising steps (default 50).
            triposg_guidance_scale: TripoSG guidance scale (default 7.5).
            triposg_seed: TripoSG RNG seed (default 42).
        """
        self.image = Image.open(image_path).convert("RGBA")
        self.mesh = None
        self.painted_mesh = None
        self.texture_backend = texture_backend

        # TripoSG texturizer (lazy-loaded on first use)
        self._triposg = TripoSGTexturizer(
            model_path=triposg_model_path,
            num_inference_steps=triposg_num_steps,
            guidance_scale=triposg_guidance_scale,
            seed=triposg_seed,
        )

        # Hunyuan pipelines — only load when using the hunyuan backend
        if texture_backend != "triposg":
            if pipeline_flow is None:
                self.pipeline_flow = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
                    "tencent/Hunyuan3D-2"
                )
            else:
                self.pipeline_flow = pipeline_flow

            if pipeline_paint is None:
                self.pipeline_paint = Hunyuan3DPaintPipeline.from_pretrained(
                    "tencent/Hunyuan3D-2"
                )
            else:
                self.pipeline_paint = pipeline_paint
        else:
            self.pipeline_flow = pipeline_flow
            self.pipeline_paint = pipeline_paint

    # ── mesh generation ───────────────────────────────────────────────────

    def generate_mesh(self):
        """
        Generate the base (untextured) 3D mesh from the image using Hunyuan3D-2.

        Not required when ``texture_backend="triposg"`` since TripoSG
        generates both geometry and texture in one step.
        """
        if self.pipeline_flow is None:
            raise ValueError(
                "pipeline_flow is required for Hunyuan mesh generation. "
                "Pass one explicitly or use texture_backend='hunyuan'."
            )
        self.mesh = self.pipeline_flow(image=self.image)[0]
        return self.mesh

    def generate_painted_mesh(self, backend: str = None):
        """
        Generate the textured mesh.

        Args:
            backend: Override the instance-level ``texture_backend`` for this
                     call.  One of ``"hunyuan"`` or ``"triposg"``.

        Returns:
            The textured mesh object.  For ``"hunyuan"`` this is whatever
            ``Hunyuan3DPaintPipeline`` returns; for ``"triposg"`` it is a
            ``trimesh.Trimesh``.  Both support ``.export(path)`` for .glb output.
        """
        use_backend = backend if backend is not None else self.texture_backend

        if use_backend == "triposg":
            return self._generate_with_triposg()
        else:
            return self._generate_with_hunyuan()

    def _generate_with_hunyuan(self):
        """Internal — run Hunyuan3DPaintPipeline on the base mesh."""
        if self.mesh is None:
            raise ValueError(
                "Call generate_mesh() before generate_painted_mesh() "
                "when using the Hunyuan backend."
            )
        if self.pipeline_paint is None:
            raise ValueError(
                "pipeline_paint is required for Hunyuan texturing. "
                "Pass one explicitly or use texture_backend='hunyuan'."
            )
        self.painted_mesh = self.pipeline_paint(self.mesh, image=self.image)
        return self.painted_mesh

    def _generate_with_triposg(self):
        """
        Internal — run TripoSG on the original image to produce a complete
        textured mesh (bypasses the Hunyuan shape generation step).
        """
        print("[MeshGenerator] Generating textured mesh with TripoSG ...")
        self.painted_mesh = self._triposg.generate(self.image)
        return self.painted_mesh

    # ── save helpers ──────────────────────────────────────────────────────

    def save_mesh(self, path: str):
        """Save the base (untextured) mesh to *path*."""
        if self.mesh is None:
            raise ValueError("No mesh to save.  Call generate_mesh() first.")
        self.mesh.export(path)

    def save_painted_mesh(self, path: str):
        """Save the textured mesh to *path*."""
        if self.painted_mesh is None:
            raise ValueError(
                "No painted mesh to save.  Call generate_painted_mesh() first."
            )
        self.painted_mesh.export(path)


# ── HunyuanRenderer ──────────────────────────────────────────────────────────

class HunyuanRenderer:
    """
    Renders 3D meshes loaded from .glb files using PyTorch3D.
    """

    def __init__(self, mesh_path: str, device="cuda"):
        """
        Args:
            mesh_path: Path to the mesh file (.glb format).
            device: PyTorch device string ('cuda' or 'cpu').
        """
        self.device = device
        self.mesh = None
        self.io = None
        self.image_size = None
        self.load_mesh(mesh_path)

    def load_mesh(self, path: str):
        """Load a mesh from *path*."""
        self.io = IO()
        self.io.register_meshes_format(MeshGlbFormat())
        self.mesh = self.io.load_mesh(path, include_textures=True).to(self.device)

    def get_R_T_pytorch(self, R, T):
        """
        Convert OpenCV [R | T] to PyTorch3D camera convention.

        Args:
            R: (N, 3, 3) rotation tensor.
            T: (N, 3, 1) or (N, 3) translation tensor.

        Returns:
            (R_pytorch3d, T_pytorch3d)
        """
        R_pytorch3d = R.clone().permute(0, 2, 1)
        T_pytorch3d = T.clone().squeeze(-1) if T.dim() == 3 else T.clone()

        R_pytorch3d[:, :, :2] *= -1
        T_pytorch3d[:, :2] *= -1

        return R_pytorch3d, T_pytorch3d

    def get_pytorch3d_camera(self, R_pytorch3d, T_pytorch3d, focal_length, principal_point):
        """
        Build a ``PerspectiveCameras`` object from PyTorch3D-convention matrices.
        """
        return PerspectiveCameras(
            device=self.device,
            R=R_pytorch3d,
            T=T_pytorch3d,
            focal_length=focal_length,
            principal_point=principal_point,
            image_size=(self.image_size,),
            in_ndc=False,
        )

    def render(
        self,
        image_size,
        R,
        T,
        K=None,
        focal_length=None,
        principal_point=None,
        blur_radius: float = 0.0,
        location=((0.0, 0.0, -3.0),),
    ):
        """
        Render the mesh with specified camera parameters.

        Args:
            image_size: Output image size (int or (H, W) tuple).
            R: Rotation matrix (numpy or tensor, 3×3 or N×3×3).
            T: Translation vector (numpy or tensor, (3,), (3,1), or N×3).
            K: Optional 3×3 intrinsic matrix.  If given, *focal_length* and
               *principal_point* are derived from it.
            focal_length: Overrides default if *K* is ``None``.
            principal_point: Overrides default if *K* is ``None``.
            blur_radius: Soft-rasterization blur radius.
            location: Light source position(s).

        Returns:
            Rendered RGBA image tensor of shape (1, H, W, 4).
        """
        if isinstance(image_size, (list, tuple)):
            self.image_size = tuple(image_size)
        else:
            self.image_size = (image_size, image_size)

        if K is not None:
            focal_length = torch.tensor(
                [[K[0, 0], K[1, 1]]], device=self.device, dtype=torch.float32
            )
            principal_point = torch.tensor(
                [[K[0, 2], K[1, 2]]], device=self.device, dtype=torch.float32
            )
        else:
            if focal_length is None:
                focal_length = torch.tensor(
                    [[self.image_size[0], self.image_size[1]]],
                    device=self.device,
                    dtype=torch.float32,
                )
            if principal_point is None:
                principal_point = torch.tensor(
                    [[self.image_size[0] / 2, self.image_size[1] / 2]],
                    device=self.device,
                    dtype=torch.float32,
                )

        if isinstance(R, np.ndarray):
            R = torch.from_numpy(R).to(self.device).float()
        if isinstance(T, np.ndarray):
            T = torch.from_numpy(T).to(self.device).float()

        if R.dim() == 2:
            R = R.unsqueeze(0)
        if T.dim() in (1, 2):
            T = T.unsqueeze(0)

        R_pytorch3d, T_pytorch3d = self.get_R_T_pytorch(R, T)
        cameras = self.get_pytorch3d_camera(R_pytorch3d, T_pytorch3d, focal_length, principal_point)

        faces_per_pixel = 5 if blur_radius > 0 else 1
        raster_settings = RasterizationSettings(
            image_size=self.image_size,
            blur_radius=blur_radius,
            faces_per_pixel=faces_per_pixel,
        )
        rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)

        lights = PointLights(device=self.device, location=[list(location[0])])
        shader = SoftPhongShader(
            device=self.device,
            cameras=cameras,
            lights=lights,
            blend_params=BlendParams(background_color=(1.0, 1.0, 1.0)),
        )

        renderer = MeshRenderer(rasterizer=rasterizer, shader=shader)
        return renderer(self.mesh)
