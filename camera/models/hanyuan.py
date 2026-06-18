"""
Hunyuan3D mesh generation and rendering module.

This module contains:
- MeshGenerator: Generate 3D meshes from 2D images using Hunyuan3D
- HunyuanRenderer: Render 3D meshes using PyTorch3D
"""

import torch
import numpy as np
from PIL import Image
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
from hy3dgen.texgen import Hunyuan3DPaintPipeline
from pytorch3d.io import IO
from pytorch3d.io.experimental_gltf_io import MeshGlbFormat
from pytorch3d.renderer import (
    PointLights,
    MeshRenderer,
    MeshRasterizer,
    RasterizationSettings,
    SoftPhongShader,
    PerspectiveCameras,
    BlendParams
)


class MeshGenerator:
    """
    Generates 3D meshes from 2D images using Hunyuan3D models.
    """
    
    def __init__(self, image_path, pipeline_flow=None, pipeline_paint=None):
        """
        Initialize the mesh generator.
        
        Args:
            image_path: Path to the input image
            pipeline_flow: Optional pre-loaded flow matching pipeline
            pipeline_paint: Optional pre-loaded paint pipeline
        """
        self.image = Image.open(image_path).convert("RGBA")
        self.mesh = None
        self.painted_mesh = None
        
        # Load pipelines if not provided
        if pipeline_flow is None:
            self.pipeline_flow = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained('tencent/Hunyuan3D-2')
        else:
            self.pipeline_flow = pipeline_flow
            
        if pipeline_paint is None:
            self.pipeline_paint = Hunyuan3DPaintPipeline.from_pretrained('tencent/Hunyuan3D-2')
        else:
            self.pipeline_paint = pipeline_paint

    def generate_mesh(self):
        """Generate the base 3D mesh from the image."""
        self.mesh = self.pipeline_flow(image=self.image)[0]
        return self.mesh

    def generate_painted_mesh(self):
        """Apply texture painting to the mesh."""
        if self.mesh is None:
            raise ValueError("Must generate mesh before painting. Call generate_mesh() first.")
        self.painted_mesh = self.pipeline_paint(self.mesh, image=self.image)
        return self.painted_mesh

    def save_mesh(self, path):
        """Save the base mesh to file."""
        if self.mesh is None:
            raise ValueError("No mesh to save. Call generate_mesh() first.")
        self.mesh.export(path)

    def save_painted_mesh(self, path):
        """Save the painted mesh to file."""
        if self.painted_mesh is None:
            raise ValueError("No painted mesh to save. Call generate_painted_mesh() first.")
        self.painted_mesh.export(path)


class HunyuanRenderer:
    """
    Renders 3D meshes using PyTorch3D with custom camera parameters.
    """
    
    def __init__(self, mesh_path, device="cuda"):
        """
        Initialize the renderer.
        
        Args:
            mesh_path: Path to the mesh file (.glb format)
            device: Device to use for rendering ('cuda' or 'cpu')
        """
        self.device = device
        self.mesh = None
        self.io = None
        self.load_mesh(mesh_path)

    def load_mesh(self, path):
        """Load a mesh from file."""
        self.io = IO()
        self.io.register_meshes_format(MeshGlbFormat())
        self.mesh = self.io.load_mesh(path, include_textures=True).to(self.device)

    def get_R_T_pytorch(self, R, T):
        """
        Convert camera from OpenCV [R|T] to PyTorch3D's convention.
        
        Args:
            R: Rotation matrix (3x3 or Nx3x3)
            T: Translation vector (3,) or (Nx3)
            
        Returns:
            R_pytorch3d, T_pytorch3d: Converted matrices for PyTorch3D
        """
        R_pytorch3d = R.clone().permute(0, 2, 1)  # Transpose
        T_pytorch3d = T.clone().squeeze(-1) if T.dim() == 3 else T.clone()

        # Invert X and Y axes for rotation and translation
        R_pytorch3d[:, :, :2] *= -1
        T_pytorch3d[:, :2] *= -1

        return R_pytorch3d, T_pytorch3d

    def get_pytorch3d_camera(self, R_pytorch3d, T_pytorch3d, focal_length, principal_point):
        """
        Create a PyTorch3D PerspectiveCameras object.
        
        Args:
            R_pytorch3d: Rotation matrix in PyTorch3D convention
            T_pytorch3d: Translation vector in PyTorch3D convention
            focal_length: Camera focal length
            principal_point: Camera principal point
            
        Returns:
            PerspectiveCameras object
        """
        return PerspectiveCameras(
            device=self.device,
            R=R_pytorch3d,
            T=T_pytorch3d,
            focal_length=focal_length,
            principal_point=principal_point,
            image_size=(self.image_size,),
            in_ndc=False
        )

    def render(self, image_size, R, T, K=None, focal_length=None, principal_point=None,
               blur_radius=0.0, location=[[0.0, 0.0, -3.0]]):
        """
        Render the mesh with specified camera parameters.
        
        Args:
            image_size: Output image size (int or tuple)
            R: Rotation matrix
            T: Translation vector
            K: Optional intrinsic camera matrix
            focal_length: Optional focal length (used if K is None)
            principal_point: Optional principal point (used if K is None)
            blur_radius: Blur radius for soft rasterization
            location: Light location
            
        Returns:
            Rendered image tensor
        """
        if isinstance(image_size, (list, tuple)):
            self.image_size = image_size
        else:
            self.image_size = (image_size, image_size)

        # Set camera intrinsics
        if K is not None:
            focal_length = torch.tensor(
                [[K[0, 0], K[1, 1]]], device=self.device, dtype=torch.float32)
            principal_point = torch.tensor(
                [[K[0, 2], K[1, 2]]], device=self.device, dtype=torch.float32)
        else:
            if focal_length is None:
                focal_length = torch.tensor(
                    [[self.image_size[0], self.image_size[1]]], device=self.device, dtype=torch.float32)
            if principal_point is None:
                principal_point = torch.tensor(
                    [[self.image_size[0] / 2, self.image_size[1] / 2]], device=self.device, dtype=torch.float32)

        # Convert numpy arrays to tensors if needed
        if isinstance(R, np.ndarray):
            R = torch.from_numpy(R).to(self.device).float()
        if isinstance(T, np.ndarray):
            T = torch.from_numpy(T).to(self.device).float()

        # Ensure proper dimensions
        if R.dim() == 2:
            R = R.unsqueeze(0)
        if T.dim() in [1, 2]:
            T = T.unsqueeze(0)

        # Convert to PyTorch3D convention
        R_pytorch3d, T_pytorch3d = self.get_R_T_pytorch(R, T)

        # Create camera
        cameras = self.get_pytorch3d_camera(R_pytorch3d, T_pytorch3d, focal_length, principal_point)

        # Set up rasterization
        faces_per_pixel = 5 if blur_radius > 0 else 1
        raster_settings = RasterizationSettings(
            image_size=self.image_size,
            blur_radius=blur_radius,
            faces_per_pixel=faces_per_pixel
        )
        rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)

        # Set up lighting and shader
        lights = PointLights(device=self.device, location=location)
        shader = SoftPhongShader(
            device=self.device,
            cameras=cameras,
            lights=lights,
            blend_params=BlendParams(background_color=(1.0, 1.0, 1.0))
        )

        # Render
        renderer = MeshRenderer(rasterizer=rasterizer, shader=shader)
        rendered_output_tensor = renderer(self.mesh)

        return rendered_output_tensor
