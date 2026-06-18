"""
TRELLIS.2 mesh generation and rendering module.

This module contains:
- TrellisMeshGenerator: Generate 3D meshes from 2D images using TRELLIS.2-4B
- TrellisRenderer: Render TRELLIS.2-generated .glb meshes using PyTorch3D

References:
    https://huggingface.co/microsoft/TRELLIS.2-4B
    https://github.com/microsoft/TRELLIS.2
"""

import os
import torch
import numpy as np
from PIL import Image

# TRELLIS.2 pipeline
from trellis2.pipelines import Trellis2ImageTo3DPipeline
import o_voxel

# PyTorch3D rendering
from pytorch3d.io import IO
from pytorch3d.io.experimental_gltf_io import MeshGlbFormat
from pytorch3d.renderer import (
    PointLights,
    MeshRenderer,
    MeshRasterizer,
    RasterizationSettings,
    SoftPhongShader,
    PerspectiveCameras,
    BlendParams,
)


class TrellisMeshGenerator:
    """
    Generates 3D meshes from 2D images using TRELLIS.2-4B.

    TRELLIS.2 uses an O-Voxel representation that natively handles arbitrary
    topology, open surfaces and PBR materials. The generated asset is exported
    as a .glb file that is directly consumable by TrellisRenderer.
    """

    # Default GLB export settings (matches TRELLIS.2 HuggingFace example)
    DEFAULT_DECIMATION_TARGET = 1_000_000
    DEFAULT_TEXTURE_SIZE = 4096
    DEFAULT_AABB = [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]]
    DEFAULT_SIMPLIFY_LIMIT = 16_777_216  # nvdiffrast vertex limit

    def __init__(
        self,
        image_path: str,
        pipeline=None,
        device: str = "cuda",
        resolution: int = 1024,
    ):
        """
        Initialize the TRELLIS mesh generator.

        Args:
            image_path: Path to the input RGB/RGBA image.
            pipeline: Optional pre-loaded Trellis2ImageTo3DPipeline.
                      If None, loads "microsoft/TRELLIS.2-4B" from HuggingFace.
            device: Device string ("cuda" or "cpu").
            resolution: O-Voxel grid resolution.
                        512 ≈ 3 s, 1024 ≈ 17 s, 1536 ≈ 60 s on H100.
        """
        self.device = device
        self.resolution = resolution
        self.image = Image.open(image_path).convert("RGB")
        self.raw_mesh = None       # raw TRELLIS.2 mesh object
        self.glb_path = None       # path to the exported .glb file

        if pipeline is None:
            print("Loading TRELLIS.2-4B pipeline...")
            self.pipeline = Trellis2ImageTo3DPipeline.from_pretrained(
                "microsoft/TRELLIS.2-4B"
            )
            self.pipeline.cuda()
            print("TRELLIS.2-4B pipeline loaded.")
        else:
            self.pipeline = pipeline

    def generate_mesh(self):
        """
        Run the TRELLIS.2 image-to-3D pipeline.

        Returns:
            Raw TRELLIS.2 mesh object (O-Voxel based).
        """
        print("Running TRELLIS.2 image-to-3D generation...")
        outputs = self.pipeline.run(self.image)
        self.raw_mesh = outputs[0]
        # Stay within nvdiffrast's vertex limit
        self.raw_mesh.simplify(self.DEFAULT_SIMPLIFY_LIMIT)
        print("TRELLIS.2 mesh generated.")
        return self.raw_mesh

    def save_mesh(self, path: str, **export_kwargs):
        """
        Convert the raw O-Voxel mesh to a .glb file.

        This is the TRELLIS.2 equivalent of HunyuanRenderer's save_painted_mesh.
        All textures (PBR materials, opacity) are baked in during export.

        Args:
            path: Output .glb file path.
            **export_kwargs: Override default to_glb parameters:
                decimation_target, texture_size, aabb, remesh,
                remesh_band, remesh_project, verbose.
        """
        if self.raw_mesh is None:
            raise ValueError("No mesh to save. Call generate_mesh() first.")

        params = dict(
            decimation_target=self.DEFAULT_DECIMATION_TARGET,
            texture_size=self.DEFAULT_TEXTURE_SIZE,
            aabb=self.DEFAULT_AABB,
            remesh=True,
            remesh_band=1,
            remesh_project=0,
            verbose=True,
        )
        params.update(export_kwargs)

        print(f"Exporting TRELLIS.2 mesh to {path} ...")
        glb = o_voxel.postprocess.to_glb(
            vertices=self.raw_mesh.vertices,
            faces=self.raw_mesh.faces,
            attr_volume=self.raw_mesh.attrs,
            coords=self.raw_mesh.coords,
            attr_layout=self.raw_mesh.layout,
            voxel_size=self.raw_mesh.voxel_size,
            **params,
        )
        glb.export(path, extension_webp=True)
        self.glb_path = path
        print(f"Saved .glb → {path}")
        return path

    # Alias so callers that use the Hunyuan API also work without changes
    def generate_painted_mesh(self):
        """Alias: TRELLIS bakes textures during export — this is a no-op."""
        if self.raw_mesh is None:
            raise ValueError("Call generate_mesh() first.")
        return self.raw_mesh

    def save_painted_mesh(self, path: str, **export_kwargs):
        """Alias for save_mesh (TRELLIS bakes PBR materials at export time)."""
        return self.save_mesh(path, **export_kwargs)


class TrellisRenderer:
    """
    Renders TRELLIS.2-generated .glb meshes using PyTorch3D.

    The rendering pipeline is identical to HunyuanRenderer — both consume
    standard .glb files. This class exists as a named counterpart so
    backend selection (hanyuan / trellis) is explicit at call sites.
    """

    def __init__(self, mesh_path: str, device: str = "cuda"):
        """
        Initialize the renderer.

        Args:
            mesh_path: Path to the .glb mesh file produced by TrellisMeshGenerator.
            device: Device string ('cuda' or 'cpu').
        """
        self.device = device
        self.mesh = None
        self.io = None
        self.image_size = None
        self.load_mesh(mesh_path)

    def load_mesh(self, path: str):
        """Load a .glb mesh from disk."""
        self.io = IO()
        self.io.register_meshes_format(MeshGlbFormat())
        self.mesh = self.io.load_mesh(path, include_textures=True).to(self.device)

    # ── Camera convention helpers ─────────────────────────────────────────

    def get_R_T_pytorch(self, R: torch.Tensor, T: torch.Tensor):
        """
        Convert camera matrices from OpenCV [R|T] to PyTorch3D convention.

        Args:
            R: Rotation matrix  (N×3×3)
            T: Translation vector (N×3 or N×3×1)

        Returns:
            R_pytorch3d, T_pytorch3d
        """
        R_pytorch3d = R.clone().permute(0, 2, 1)
        T_pytorch3d = T.clone().squeeze(-1) if T.dim() == 3 else T.clone()

        R_pytorch3d[:, :, :2] *= -1
        T_pytorch3d[:, :2] *= -1

        return R_pytorch3d, T_pytorch3d

    def get_pytorch3d_camera(
        self,
        R_pytorch3d: torch.Tensor,
        T_pytorch3d: torch.Tensor,
        focal_length: torch.Tensor,
        principal_point: torch.Tensor,
    ) -> PerspectiveCameras:
        """Create a PyTorch3D PerspectiveCameras object."""
        return PerspectiveCameras(
            device=self.device,
            R=R_pytorch3d,
            T=T_pytorch3d,
            focal_length=focal_length,
            principal_point=principal_point,
            image_size=(self.image_size,),
            in_ndc=False,
        )

    # ── Main render method ────────────────────────────────────────────────

    def render(
        self,
        image_size,
        R,
        T,
        K=None,
        focal_length=None,
        principal_point=None,
        blur_radius: float = 0.0,
        location=None,
    ) -> torch.Tensor:
        """
        Render the loaded mesh from the given camera pose.

        Args:
            image_size: Output size — int or (H, W) tuple.
            R: Rotation matrix (numpy or torch, 3×3 or N×3×3).
            T: Translation vector (numpy or torch, 3, N×3 or N×3×1).
            K: Optional 3×3 intrinsic matrix (overrides focal_length / principal_point).
            focal_length: Explicit focal length tensor if K is None.
            principal_point: Explicit principal point tensor if K is None.
            blur_radius: Soft-rasterization blur radius.
            location: Point-light location [[x, y, z]].

        Returns:
            Rendered image tensor of shape (1, H, W, 4).
        """
        if location is None:
            location = [[0.0, 0.0, -3.0]]

        # ── resolve image size ────────────────────────────────────────────
        if isinstance(image_size, (list, tuple)):
            self.image_size = tuple(image_size)
        else:
            self.image_size = (image_size, image_size)

        # ── resolve intrinsics ────────────────────────────────────────────
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

        # ── numpy → tensor ────────────────────────────────────────────────
        if isinstance(R, np.ndarray):
            R = torch.from_numpy(R).to(self.device).float()
        if isinstance(T, np.ndarray):
            T = torch.from_numpy(T).to(self.device).float()

        # ── ensure batch dimension ────────────────────────────────────────
        if R.dim() == 2:
            R = R.unsqueeze(0)
        if T.dim() in [1, 2]:
            T = T.unsqueeze(0)

        # ── OpenCV → PyTorch3D convention ─────────────────────────────────
        R_p3d, T_p3d = self.get_R_T_pytorch(R, T)
        cameras = self.get_pytorch3d_camera(R_p3d, T_p3d, focal_length, principal_point)

        # ── rasterizer ────────────────────────────────────────────────────
        faces_per_pixel = 5 if blur_radius > 0 else 1
        raster_settings = RasterizationSettings(
            image_size=self.image_size,
            blur_radius=blur_radius,
            faces_per_pixel=faces_per_pixel,
        )
        rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)

        # ── shader ────────────────────────────────────────────────────────
        lights = PointLights(device=self.device, location=location)
        shader = SoftPhongShader(
            device=self.device,
            cameras=cameras,
            lights=lights,
            blend_params=BlendParams(background_color=(1.0, 1.0, 1.0)),
        )

        renderer = MeshRenderer(rasterizer=rasterizer, shader=shader)
        return renderer(self.mesh)
