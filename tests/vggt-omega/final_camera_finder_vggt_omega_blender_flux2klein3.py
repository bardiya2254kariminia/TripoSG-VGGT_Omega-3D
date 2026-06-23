from enum import Enum

from PIL import Image

try:
    from diffusers import Flux2KleinPipeline
except ImportError as e:
    raise ImportError(
        "Flux2KleinPipeline requires diffusers>=0.37.1, transformers>=4.51.0, and torch>=2.4 "
        "(torch.nn.RMSNorm). Install with: pip install 'diffusers>=0.37.1' "
        "'transformers>=4.51.0,<5.0' 'torch>=2.4' --index-url https://download.pytorch.org/whl/cu121"
    ) from e
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

import torch
import numpy as np
import shutil
import os
import subprocess
import tempfile
import textwrap
import matplotlib.pyplot as plt
import gc
import sys
import cv2

from pytorch3d.renderer import (
    MeshRenderer,
    MeshRasterizer,
    RasterizationSettings
)
from pytorch3d.renderer.blending import BlendParams

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

# Blending utilities import
blending_utils_path = os.path.abspath(os.path.join(current_dir,"../..", "blending"))
# print(f"blending_utils_path: {blending_utils_path}")
if blending_utils_path not in sys.path:
    sys.path.insert(0, blending_utils_path)

from blending_utils import (
    create_soft_blend_mask,
    create_distance_soft_blend_mask,
    create_hierarchical_blend_mask,
    poisson_blend,
    multi_scale_blend,
)


# ---------------------------------------------------------------------------
# Blender subprocess rendering (high-quality Cycles with 3-point lighting)
# ---------------------------------------------------------------------------

BLENDER_RENDER_SCRIPT = textwrap.dedent(r'''
import bpy
from mathutils import Vector
import math
from pathlib import Path
import sys
import json

args = json.loads(sys.argv[sys.argv.index("--") + 1])
camera_location = tuple(args["camera_location"])
camera_rotation = tuple(args["camera_rotation"])
output_path = args["output_path"]
mesh_path = args["mesh_path"]
resolution = args["resolution"]
use_perspective = args.get("use_perspective", True)
focal_length_mm = args.get("focal_length_mm", 50.0)
ortho_scale = args.get("ortho_scale", 1.0)


def enable_gpus(max_gpus=None):
    preferences = bpy.context.preferences
    cycles_preferences = preferences.addons["cycles"].preferences
    cycles_preferences.refresh_devices()
    available_types = []
    for compute_type in ['OPTIX', 'HIP', 'CUDA', 'METAL', 'OPENCL']:
        try:
            cycles_preferences.compute_device_type = compute_type
            cycles_preferences.refresh_devices()
            gpu_devices = [d for d in cycles_preferences.devices if d.type != 'CPU']
            if gpu_devices:
                available_types.append((compute_type, len(gpu_devices)))
        except (AttributeError, TypeError):
            continue

    if not available_types:
        raise RuntimeError("No GPU compute devices available.")

    priority = {'OPTIX': 3, 'CUDA': 4, 'HIP': 2, 'METAL': 1, 'OPENCL': 0}
    available_types.sort(key=lambda x: priority.get(x[0], -1), reverse=True)
    selected_type = available_types[0][0]

    cycles_preferences.compute_device_type = selected_type
    cycles_preferences.refresh_devices()

    gpu_devices = [d for d in cycles_preferences.devices if d.type != 'CPU']
    if gpu_devices and selected_type == 'OPTIX':
        compute_indicators = ['H100', 'A100', 'A40', 'A30', 'A10', 'V100', 'P100', 'Tesla']
        for device in gpu_devices:
            if any(ind in device.name for ind in compute_indicators):
                if 'CUDA' in [t[0] for t in available_types]:
                    selected_type = 'CUDA'
                    break

    cycles_preferences.compute_device_type = selected_type
    cycles_preferences.refresh_devices()

    all_devices = cycles_preferences.devices
    gpu_devices = [d for d in all_devices if d.type != 'CPU']
    devices_to_use = gpu_devices if max_gpus is None else gpu_devices[:max_gpus]

    for device in all_devices:
        device.use = device in devices_to_use

    bpy.context.scene.cycles.device = "GPU"
    bpy.context.scene.cycles.use_persistent_data = True
    return selected_type


bpy.ops.wm.read_factory_settings(use_empty=True)
compute_type = enable_gpus()

scene = bpy.context.scene
scene.render.engine = 'CYCLES'
scene.cycles.samples = 512
scene.cycles.use_adaptive_sampling = False
if hasattr(scene.cycles, "use_denoising"):
    scene.cycles.use_denoising = True
scene.render.resolution_x = resolution
scene.render.resolution_y = resolution
scene.render.resolution_percentage = 100
scene.render.filepath = output_path
scene.render.image_settings.file_format = 'PNG'
scene.render.image_settings.color_mode = 'RGBA'
scene.render.film_transparent = True

view_layer = bpy.context.view_layer
if hasattr(view_layer, "cycles"):
    view_layer.cycles.use_denoising = True
    if hasattr(view_layer.cycles, "denoiser"):
        if compute_type == 'OPTIX':
            view_layer.cycles.denoiser = 'OPTIX'
        else:
            view_layer.cycles.denoiser = 'OPENIMAGEDENOISE'

# Transparent world
scene.world = bpy.data.worlds.new("TransparentWorld")
scene.world.use_nodes = True
scene.world.node_tree.nodes.clear()

# Import GLB mesh
bpy.ops.object.select_all(action='DESELECT')
bpy.ops.import_scene.gltf(filepath=mesh_path)
imported_meshes = [obj for obj in bpy.context.selected_objects if obj.type == 'MESH']
if not imported_meshes:
    raise RuntimeError(f"No mesh imported from {mesh_path}")

# Setup camera
scn = bpy.context.scene
cam_data = bpy.data.cameras.new("RenderCam")
cam_obj = bpy.data.objects.new("RenderCam", cam_data)
cam_obj.location = camera_location
cam_obj.rotation_euler = camera_rotation

if use_perspective:
    cam_data.type = 'PERSP'
    cam_data.lens = focal_length_mm
else:
    cam_data.type = 'ORTHO'
    cam_data.ortho_scale = ortho_scale

scn.collection.objects.link(cam_obj)
scene.camera = cam_obj

# 3-point lighting (key + fill + rim)
light_data = bpy.data.lights.new(name='KeyLight', type='AREA')
light_data.energy = 3500.0
light_data.size = 20
light_obj = bpy.data.objects.new('KeyLight', light_data)
light_obj.location = cam_obj.location
light_obj.rotation_euler = cam_obj.rotation_euler
scene.collection.objects.link(light_obj)

fill_data = bpy.data.lights.new(name='FillLight', type='AREA')
fill_data.energy = 2000.0
fill_data.size = 15
fill_obj = bpy.data.objects.new('FillLight', fill_data)
fill_obj.location = (cam_obj.location[0] - 3.0, cam_obj.location[1], cam_obj.location[2])
fill_obj.rotation_euler = (math.radians(90.0), 0.0, math.radians(-30.0))
scene.collection.objects.link(fill_obj)

rim_data = bpy.data.lights.new(name='RimLight', type='AREA')
rim_data.energy = 2000.0
rim_data.size = 15
rim_obj = bpy.data.objects.new('RimLight', rim_data)
rim_obj.location = (cam_obj.location[0] + 3.0, cam_obj.location[1], cam_obj.location[2])
rim_obj.rotation_euler = (math.radians(90.0), 0.0, math.radians(30.0))
scene.collection.objects.link(rim_obj)

bpy.context.view_layer.update()
bpy.ops.render.render(write_still=True)
print("BLENDER_RENDER_DONE")
''')


def _opencv_RT_to_blender_camera(R: np.ndarray, T: np.ndarray):
    """
    Convert OpenCV camera R, T to Blender camera location and Euler rotation.
    OpenCV: X-right, Y-down, Z-forward
    Blender: X-right, Y-forward, Z-up (camera looks along local -Z)
    """
    if R.shape == (3, 3) and T.shape in [(3, 1), (3,)]:
        T_flat = T.flatten()
    else:
        raise ValueError(f"Unexpected R shape {R.shape} or T shape {T.shape}")

    cam_pos_world = -R.T @ T_flat

    # Build Blender rotation matrix from OpenCV R
    # OpenCV camera axes in world: right = R[0], down = R[1], forward = R[2]
    # Blender camera axes: right = local X, up = local Y, look = local -Z
    # Blender world: Z-up, Y-forward (but camera convention is local -Z = look direction)
    
    # Convert OpenCV R to Blender rotation
    # OpenCV: cam_z = forward (into scene), cam_y = down, cam_x = right
    # Blender: cam_z = backward (out of scene = -look), cam_y = up, cam_x = right
    # So: blender_x = opencv_x, blender_y = -opencv_y, blender_z = -opencv_z
    R_blender = R.copy()
    R_blender[1, :] *= -1  # flip Y
    R_blender[2, :] *= -1  # flip Z

    import mathutils
    mat = mathutils.Matrix(R_blender.tolist()).to_3x3()
    euler = mat.to_euler('XYZ')
    
    return tuple(cam_pos_world.tolist()), (euler.x, euler.y, euler.z)


def _opencv_RT_to_blender_camera_numpy(R: np.ndarray, T: np.ndarray):
    """
    Convert OpenCV R, T to Blender camera location and Euler rotation (XYZ).
    Pure numpy — no mathutils dependency (for use outside Blender).
    """
    T_flat = T.flatten()
    cam_pos_world = -R.T @ T_flat

    # Blender rotation from OpenCV R
    R_bl = R.copy()
    R_bl[1, :] *= -1
    R_bl[2, :] *= -1

    # Euler XYZ decomposition from rotation matrix
    # R_bl is camera-to-world rotation in Blender convention
    sy = np.sqrt(R_bl[0, 0]**2 + R_bl[1, 0]**2)
    singular = sy < 1e-6
    if not singular:
        x = np.arctan2(R_bl[2, 1], R_bl[2, 2])
        y = np.arctan2(-R_bl[2, 0], sy)
        z = np.arctan2(R_bl[1, 0], R_bl[0, 0])
    else:
        x = np.arctan2(-R_bl[1, 2], R_bl[1, 1])
        y = np.arctan2(-R_bl[2, 0], sy)
        z = 0.0

    return list(cam_pos_world), [x, y, z]


def render_with_blender(
    R: np.ndarray,
    T: np.ndarray,
    mesh_path: str,
    output_path: str,
    resolution: int = 1024,
    focal_length_mm: float = 50.0,
    blender_bin: str = "blender",
):
    """
    Render a GLB mesh using Blender Cycles via subprocess.
    Converts OpenCV R, T to Blender camera parameters and renders with
    3-point lighting, denoising, and transparent background.
    """
    cam_loc, cam_rot = _opencv_RT_to_blender_camera_numpy(R, T)

    # Cast to native Python floats so json.dumps can serialize them
    cam_loc = [float(v) for v in cam_loc]
    cam_rot = [float(v) for v in cam_rot]

    render_args = json.dumps({
        "camera_location": cam_loc,
        "camera_rotation": cam_rot,
        "output_path": output_path,
        "mesh_path": os.path.abspath(mesh_path),
        "resolution": resolution,
        "use_perspective": True,
        "focal_length_mm": focal_length_mm,
    })

    script_path = os.path.join(tempfile.gettempdir(), "_blender_render_script.py")
    with open(script_path, "w") as f:
        f.write(BLENDER_RENDER_SCRIPT)

    cmd = [
        blender_bin, "--background", "--python", script_path, "--", render_args
    ]
    print(f"Launching Blender render: {' '.join(cmd[:4])} ...")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

    if result.returncode != 0 or "BLENDER_RENDER_DONE" not in result.stdout:
        print(f"Blender stdout:\n{result.stdout[-2000:]}")
        print(f"Blender stderr:\n{result.stderr[-2000:]}")
        raise RuntimeError("Blender render failed. See output above.")

    print(f"Blender render complete: {output_path}")
    return Image.open(output_path).convert("RGBA")


import json


# ---------------------------------------------------------------------------
# Flux2Klein model identifier
# ---------------------------------------------------------------------------

FLUX2KLEIN_MODEL_ID = "black-forest-labs/FLUX.2-klein-9B"


# ---------------------------------------------------------------------------
# 16-view canonical camera positions (8 azimuths × 2 elevations)
# ---------------------------------------------------------------------------

class ViewAngle(Enum):
    """Exactly 16 canonical views used for mesh rendering and Flux2Klein input."""
    AZ000_EL_0 = (0,   0)
    AZ045_EL_0 = (45,  0)
    AZ090_EL_0 = (90,  0)
    AZ135_EL_0 = (135, 0)
    AZ180_EL_0 = (180, 0)
    AZ225_EL_0 = (225, 0)
    AZ270_EL_0 = (270, 0)
    AZ315_EL_0 = (315, 0)
    AZ000_EL_NEG20 = (0,   -20)
    AZ045_EL_NEG20 = (45,  -20)
    AZ090_EL_NEG20 = (90,  -20)
    AZ135_EL_NEG20 = (135, -20)
    AZ180_EL_NEG20 = (180, -20)
    AZ225_EL_NEG20 = (225, -20)
    AZ270_EL_NEG20 = (270, -20)
    AZ315_EL_NEG20 = (315, -20)
    AZ000_EL_POS20 = (0,   20)
    AZ045_EL_POS20 = (45,  20)
    AZ090_EL_POS20 = (90,  20)
    AZ135_EL_POS20 = (135, 20)
    AZ180_EL_POS20 = (180, 20)
    AZ225_EL_POS20 = (225, 20)
    AZ270_EL_POS20 = (270, 20)
    AZ315_EL_POS20 = (315, 20)


# ---------------------------------------------------------------------------
# Blending / compositing utilities
# ---------------------------------------------------------------------------

def composite_rendered_onto_target(
    rendered_rgba: np.ndarray,
    target_rgb: np.ndarray,
    use_poisson: bool = False,
    use_laplacian: bool = True,
) -> np.ndarray:
    """
    Composite a rendered RGBA image onto a target RGB image using
    advanced blending from blending_utils.
    
    rendered_rgba: (H, W, 4) uint8 — Blender output with alpha
    target_rgb: (H, W, 3) uint8 — original target image
    Returns: (H, W, 3) uint8 — composited result
    """
    alpha = rendered_rgba[:, :, 3].astype(np.float32) / 255.0
    rendered_rgb = rendered_rgba[:, :, :3]

    if use_poisson and alpha.sum() > 100:
        return poisson_blend(rendered_rgb, target_rgb, alpha)

    if use_laplacian:
        masks = create_hierarchical_blend_mask(alpha, num_levels=4)
        return multi_scale_blend(rendered_rgb, target_rgb, masks, use_laplacian=True)

    # Fallback: distance-based soft blend
    soft_mask = create_distance_soft_blend_mask(alpha, feather_px=16)
    soft_mask_3ch = soft_mask[:, :, np.newaxis]
    blended = rendered_rgb.astype(np.float32) * soft_mask_3ch + \
              target_rgb.astype(np.float32) * (1.0 - soft_mask_3ch)
    return np.clip(blended, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Mesh generation (Hunyuan3D)
# ---------------------------------------------------------------------------

class MeshGenerator:
    def __init__(self, pth):
        self.image = Image.open(pth).convert("RGBA")

    def generate_mesh(self):
        self.mesh = pipeline_flow(image=self.image)[0]

    def generate_painted_mesh(self):
        self.painted_mesh = pipeline_paint(self.mesh, image=self.image)

    def save_mesh(self, pth):
        self.mesh.export(pth)

    def save_painted_mesh(self, pth):
        self.painted_mesh.export(pth)


# ---------------------------------------------------------------------------
# PyTorch3D renderer (for fast multi-view generation during pose estimation)
# ---------------------------------------------------------------------------

class HunyuanRenderer():
    def __init__(self, mesh_pth, device="cuda"):
        self.device = device
        self.load_mesh(mesh_pth)

    def load_mesh(self, pth):
        self.io = IO()
        self.io.register_meshes_format(MeshGlbFormat())
        self.mesh = self.io.load_mesh(
            pth, include_textures=True).to(self.device)

    def get_R_T_pytorch(self, R, T):
        """Converts camera from OpenCV [R|T] to PyTorch3D's convention."""
        R_pytorch3d = R.clone().permute(0, 2, 1)
        T_pytorch3d = T.clone().squeeze(-1) if T.dim() == 3 else T.clone()
        R_pytorch3d[:, :, :2] *= -1
        T_pytorch3d[:, :2] *= -1
        return R_pytorch3d, T_pytorch3d

    def get_pytorch3d_camera(self, R_pytorch3d, T_pytorch3d, focal_length, principal_point):
        """Creates a PyTorch3D PerspectiveCameras object."""
        return PerspectiveCameras(
            device=self.device,
            R=R_pytorch3d,
            T=T_pytorch3d,
            focal_length=focal_length,
            principal_point=principal_point,
            image_size=(self.image_size, ),
            in_ndc=False
        )

    def render(self, image_size, R, T, K=None, focal_length=None, principal_point=None, blur_radius=0.0):
        if isinstance(image_size, (list, tuple)):
            self.image_size = image_size
        else:
            self.image_size = (image_size, image_size)

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
                    [[self.image_size[0]/2, self.image_size[1]/2]], device=self.device, dtype=torch.float32)

        if isinstance(R, np.ndarray):
            R = torch.from_numpy(R).to(self.device).float()
        if isinstance(T, np.ndarray):
            T = torch.from_numpy(T).to(self.device).float()

        if R.dim() == 2:
            R = R.unsqueeze(0)
        if T.dim() in [1, 2]:
            T = T.unsqueeze(0)

        R_sq = R[0]
        T_sq = T[0].reshape(3)
        cam_pos = -R_sq.T @ T_sq
        light_pos = [3.0, 0.0, 3.0]

        R_pytorch3d, T_pytorch3d = self.get_R_T_pytorch(R, T)

        cameras = self.get_pytorch3d_camera(
            R_pytorch3d, T_pytorch3d, focal_length, principal_point)

        faces_per_pixel = 5 if blur_radius > 0 else 1

        raster_settings = RasterizationSettings(
            image_size=self.image_size,
            blur_radius=blur_radius,
            faces_per_pixel=faces_per_pixel
        )

        rasterizer = MeshRasterizer(
            cameras=cameras, raster_settings=raster_settings)

        lights = PointLights(
            device=self.device, location=[light_pos])

        shader = SoftPhongShader(device=self.device, cameras=cameras, lights=lights,
                                 blend_params=BlendParams(background_color=(1.0, 1.0, 1.0)))

        renderer = MeshRenderer(
            rasterizer=rasterizer, shader=shader)

        rendered_output_tensor = renderer(self.mesh)
        return rendered_output_tensor


# ---------------------------------------------------------------------------
# Camera Pose Finder (VGGT-Omega with RGB inputs)
# ---------------------------------------------------------------------------

class CameraPoseFinder:
    def __init__(self, mesh_pth, image_size, device,temp_path, dist=3.0, fov=40):
        self.original_image_size = image_size if isinstance(
            image_size, (list, tuple)) else (image_size, image_size)

        self.device = device
        self.dist = dist
        self.fov = fov
        self.mesh_renderer = HunyuanRenderer(mesh_pth, device)

        self.vggt_model = None
        self.flux_pipe = None

        self.temp_path = temp_path
        self.view_path = os.path.join(temp_path, "temp_views")
        self.view_path2 = os.path.join(temp_path, "temp_views2")
        shutil.rmtree(self.view_path, ignore_errors=True)
        os.makedirs(self.view_path, exist_ok=True)

    def _load_vggt(self):
        if self.vggt_model is None:
            print("Loading VGGT-Omega model...")
            self.vggt_model = self.load_vggt_omega(
                checkpoint_path="/workspace/TripoSG-VGGT_Omega-3D/checkpoints/vggt-omega/vggt_omega_1b_512.pt"
            )
            print("VGGT-Omega model loaded.")

    def load_vggt_omega(
        self,
        checkpoint_path: str,
        device=None,
        enable_alignment: bool = False,
    ) -> VGGTOmega:
        """Load VGGT-Omega once and reuse across multiple meshes/images."""
        if device is None:
            device = self.device
        dev = torch.device(device)
        print("Loading VGGT-Omega model...")
        model = VGGTOmega(enable_alignment=enable_alignment).to(dev)
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        model.load_state_dict(state_dict)
        model.eval()
        print("VGGT-Omega model loaded.")
        return model

    def _unload_vggt(self):
        if self.vggt_model is not None:
            del self.vggt_model
            self.vggt_model = None
            gc.collect()
            torch.cuda.empty_cache()
            print("VGGT-Omega model unloaded.")

    # ------------------------------------------------------------------
    # Flux2Klein helpers
    # ------------------------------------------------------------------

    def _load_flux(self):
        if self.flux_pipe is None:
            print("Loading Flux2Klein pipeline...")
            self.flux_pipe = Flux2KleinPipeline.from_pretrained(
                FLUX2KLEIN_MODEL_ID,
                torch_dtype=torch.bfloat16,
            ).to(self.device)
            print("Flux2Klein pipeline loaded.")

    def _unload_flux(self):
        if self.flux_pipe is not None:
            del self.flux_pipe
            self.flux_pipe = None
            gc.collect()
            torch.cuda.empty_cache()
            print("Flux2Klein pipeline unloaded.")

    def _apply_flux_texture_transfer(
        self,
        rendered_pil: Image.Image,
        target_pil: Image.Image,
        seed: int = 0,
    ) -> Image.Image:
        """
        Use Flux2KleinPipeline to retexture `rendered_pil` so that its
        surface appearance matches the object in `target_pil`, while keeping
        the 3-D geometry, pose, and camera angle unchanged.

        Both images are passed to the model (rendered as image 1, target as
        image 2) so the pipeline can attend to the texture reference directly.
        """
        input_height, input_width = 512, 512

        rendered_512 = rendered_pil.convert("RGB").resize((input_width, input_height))
        target_512   = target_pil.convert("RGB").resize((input_width, input_height))
        # prompt = (
        #     "Change only the texture, colors, and surface materials of the 3-D "
        #     "rendered mesh object in image 1 to exactly match the texture and "
        #     "visual appearance of the object in image 2. "
        #     "Keep the 3-D geometry, silhouette, pose, and camera viewpoint of "
        #     "image 1 completely unchanged."
        #     "Only modify the texture and material appearance."
        # )
        prompt = (
            "Texture transfer: keep image 1's exact geometry, silhouette, pose, and camera "
            "angle unchanged. Apply only image 2's surface texture — colors, patterns, "
            "material, reflectivity — as if re-skinning the same 3D model. Ignore image 2's "
            "shape or pose entirely. Output = image 1's structure with image 2's texture."
        )
        # prompt = (
        #     "Improve only the surface texture detail of the object in this image — "
        #     "Preserve color palette and shading and focus on the light. "
        #     "Do not change the object's geometry, silhouette, pose, proportions, "
        #     "Only the texture fidelity should change; everything else must remain pixel-for-pixel "
        #     "consistent with the input."
        # )
        num_inference_steps = 4
        sigmas = np.linspace(0.95, 1 / num_inference_steps, num_inference_steps)
        gen_kwargs = dict(
            guidance_scale=1.0,
            num_inference_steps=num_inference_steps,
            sigmas=sigmas.tolist(),
            generator=torch.Generator(device="cpu").manual_seed(seed),
            height=input_height,
            width=input_width,
        )

        with torch.inference_mode():
            result = self.flux_pipe(
                prompt=prompt,
                image=[rendered_512, target_512],
                # image=[rendered_512],
                **gen_kwargs,
            ).images[0]

        return result

    def generate_initial_views(self, img_name, target_image_path):
        """
        Render exactly 16 views from the mesh (defined by ViewAngle), then run
        each rendered view + the target image through Flux2Klein to transfer only
        the texture.  The 16 flux-processed images are saved and used as the
        context set fed to VGGT-Omega.
        """
        print(f"Generating 16 mesh views and applying Flux2Klein texture transfer...")
        os.makedirs(self.view_path, exist_ok=True)

        target_pil = Image.open(target_image_path).convert("RGB").resize(
            (512, 512))

        all_Rs: list = []
        all_Ts: list = []
        raw_render_paths: list = []

        # ── Phase 1: render all 16 views ──────────────────────────────────
        for view in ViewAngle:
            azimuth, elevation = view.value
            r, t = self.get_opencv_camera_matrix(azimuth, elevation, self.dist)
            all_Rs.append(torch.from_numpy(r))
            all_Ts.append(torch.from_numpy(t))

            rendered_rgba = self.mesh_renderer.render(
                image_size=(512, 512),
                R=r,
                T=t,
            )[0].cpu().numpy()            # (H, W, 4)

            rendered_rgb = np.clip(rendered_rgba[..., :3], 0.0, 1.0)
            rendered_pil = Image.fromarray(
                (rendered_rgb * 255).astype(np.uint8))

            view_idx = len(raw_render_paths)
            raw_path = os.path.join(self.view_path, f"{img_name}_raw_{view_idx}.png")
            rendered_pil.save(raw_path)
            raw_render_paths.append(raw_path)

        print(f"Rendered {len(raw_render_paths)} views. "
              "Applying Flux2Klein texture transfer...")

        # ── Phase 2: Flux2Klein texture transfer ──────────────────────────
        self._load_flux()
        self.known_image_paths = []

        for view_idx, raw_path in enumerate(raw_render_paths):
            rendered_pil = Image.open(raw_path).convert("RGB")
            flux_result = self._apply_flux_texture_transfer(
                rendered_pil=rendered_pil,
                target_pil=target_pil,
                seed=view_idx,
            )
            flux_path = os.path.join(self.view_path, f"{img_name}_{view_idx}.png")
            flux_result.save(flux_path)
            self.known_image_paths.append(flux_path)
            print(f"  Flux2Klein texture transfer {view_idx + 1}/16 complete.")

        self._unload_flux()
        print("All 16 Flux2Klein texture-transferred views saved.")
        return all_Rs, all_Ts

    def get_vggt_initial_guess(self, target_image_path, all_Rs, all_Ts, img_name):
        """
        Use VGGT-Omega with RGB images (rendered views + real target).
        No silhouette conversion — all images are in RGB color space.
        """
        self._load_vggt()

        # Use the target image directly in RGB (no silhouette conversion)
        # Resize to match the rendered views' resolution
        target_img = Image.open(target_image_path).convert("RGB").resize(self.original_image_size)
        target_rgb_path = os.path.join(self.view_path, f"_rgb_target_{img_name}.png")
        target_img.save(target_rgb_path)

        all_image_paths = self.known_image_paths + [target_rgb_path]
        target_index = len(all_image_paths) - 1

        images = load_and_preprocess_images_omega(all_image_paths).to(self.device)

        print("Running VGGT-Omega for initial pose estimation (RGB mode)...")
        with torch.inference_mode():
            predictions = self.vggt_model(images)

        pose_enc = predictions["pose_enc"]   # (1, N_total, 9)
        extrinsic, intrinsic = encoding_to_camera(
            pose_enc, images.shape[-2:])     # extrinsic: (1, N_total, 3, 4)

        R_reference = all_Rs[0]
        M_reference = torch.eye(4, device=self.device)
        M_reference[0:3, :3] = R_reference

        extrinsics_new = []
        for i in range(extrinsic.shape[1]):
            M_vggt = torch.eye(4, device=self.device)
            M_vggt[0:3, :] = extrinsic[0][i]

            M_aligned = torch.matmul(M_vggt, M_reference)
            M_aligned[0:3, 3] = torch.tensor([0, 0, self.dist])

            extrinsics_new.append(M_aligned)

        extrinsics_new = torch.stack(extrinsics_new, dim=0).to(self.device)

        target_extrinsic = extrinsics_new[target_index]
        target_intrinsic = intrinsic[0, target_index]

        os.makedirs(self.view_path2, exist_ok=True)
        for i in range(len(all_image_paths)):
            rendered_image = self.mesh_renderer.render(
                image_size=self.original_image_size,
                R=extrinsics_new[i][0:3, :3].cpu().numpy(),
                T=extrinsics_new[i][0:3, 3].cpu().numpy())
            rendered_image = rendered_image[0, ..., :3].cpu().numpy()
            plt.imsave(os.path.join(self.view_path2, f'{img_name}_{i}.png'), rendered_image)

        self._unload_vggt()

        return target_extrinsic, target_intrinsic

    def save_found_pose_render(self, img_name: str, extrinsic: torch.Tensor):
        """
        Render the mesh at the R,T found by VGGT-Omega and save to view_path
        so the quality of pose estimation can be visually inspected.
        """
        R_found = extrinsic[:3, :3].cpu().numpy()
        T_found = extrinsic[:3, 3].cpu().numpy()
        rendered = self.mesh_renderer.render(
            image_size=self.original_image_size,
            R=R_found,
            T=T_found,
        )
        rendered_np = rendered[0, ..., :3].cpu().numpy()
        out_path = os.path.join(self.view_path, f"_found_pose_{img_name}.png")
        plt.imsave(out_path, np.clip(rendered_np, 0.0, 1.0))
        print(f"Found-pose render saved: {out_path}")

    def get_opencv_camera_matrix(self, azimuth_deg, elevation_deg, distance_from_origin):
        azimuth_rad = np.deg2rad(azimuth_deg)
        elevation_rad = np.deg2rad(elevation_deg)
        cam_x = distance_from_origin * \
            np.cos(elevation_rad) * np.sin(azimuth_rad)
        cam_y = distance_from_origin * np.sin(elevation_rad)
        cam_z = distance_from_origin * \
            np.cos(elevation_rad) * np.cos(azimuth_rad)
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


def compute_new_pose_from_relative(view_matrix_B_with_pose_A, view_matrix_B, view_matrix_B_prime):
    V_B_with_pose_A = view_matrix_B_with_pose_A.cpu().numpy()
    V_B = view_matrix_B.cpu().numpy()
    V_B_prime = view_matrix_B_prime.cpu().numpy()
    try:
        V_B_with_pose_A_inv = np.linalg.inv(V_B_with_pose_A)
    except np.linalg.LinAlgError:
        print("Error: view_matrix_B_with_pose_A is not invertible.")
        return None

    delta_transform = V_B @ V_B_with_pose_A_inv
    V_B_prime = delta_transform @ V_B_prime
    return V_B_prime


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_size = (512, 512)

# Image paths
image_B_with_pose_A_path = "/workspace/TripoSG-VGGT_Omega-3D/images_with_glb/data2/B.png"
image_B_path = "/workspace/TripoSG-VGGT_Omega-3D/images_with_glb/data2/input.png"
image_B_prime_path = "/workspace/TripoSG-VGGT_Omega-3D/images_with_glb/data2/B_prime.png"

# Mesh paths
mesh_B_with_pose_A_path = image_B_with_pose_A_path.replace(".png", ".glb")
mesh_B_path = image_B_path.replace(".png", ".glb")
mesh_B_prime_path = image_B_prime_path.replace(".png", ".glb")

# --- Step 1: Find camera for B with pose A ---
temp_path = os.path.join(os.path.dirname(image_B_with_pose_A_path), "temp_views_B_with_pose_A")
pose_finder = CameraPoseFinder(mesh_B_with_pose_A_path, 512, device, temp_path)
all_Rs, all_Ts = pose_finder.generate_initial_views("B_with_pose_A", image_B_with_pose_A_path)
initial_extrinsic_B_with_pose_A, initial_intrinsic = pose_finder.get_vggt_initial_guess(
    image_B_with_pose_A_path, all_Rs, all_Ts, "B_with_pose_A")
print(f"\nVGGT Initial Extrinsic (3x4) for image B_with_pose_A:\n", initial_extrinsic_B_with_pose_A)
print(f"\nVGGT Initial Intrinsic (3x3) for image B_with_pose_A:\n", initial_intrinsic)
pose_finder.save_found_pose_render("B_with_pose_A", initial_extrinsic_B_with_pose_A)

# --- Step 2: Find camera for B ---
temp_path = os.path.join(os.path.dirname(image_B_path), "temp_views_B")
pose_finder = CameraPoseFinder(mesh_B_path, 512, device, temp_path)
all_Rs, all_Ts = pose_finder.generate_initial_views("B", image_B_path)
initial_extrinsic_B, initial_intrinsic = pose_finder.get_vggt_initial_guess(
    image_B_path, all_Rs, all_Ts, "B")
print(f"\nVGGT Initial Extrinsic (3x4) for image B:\n", initial_extrinsic_B)
print(f"\nVGGT Initial Intrinsic (3x3) for image B:\n", initial_intrinsic)
pose_finder.save_found_pose_render("B", initial_extrinsic_B)

# --- Step 3: Find camera for B prime ---
temp_path = os.path.join(os.path.dirname(image_B_prime_path), "temp_views_B_prime")
pose_finder = CameraPoseFinder(mesh_B_prime_path, 512, device, temp_path)
all_Rs, all_Ts = pose_finder.generate_initial_views("B_prime", image_B_prime_path)
initial_extrinsic_B_prime, initial_intrinsic = pose_finder.get_vggt_initial_guess(
    image_B_prime_path, all_Rs, all_Ts, "B_prime")
print(f"\nVGGT Initial Extrinsic (3x4) for image B':\n", initial_extrinsic_B_prime)
print(f"\nVGGT Initial Intrinsic (3x3) for image B':\n", initial_intrinsic)
pose_finder.save_found_pose_render("B_prime", initial_extrinsic_B_prime)

# --- Step 4: Compute final camera via relative pose transfer ---
final_B_prime_extrinsic = compute_new_pose_from_relative(
    initial_extrinsic_B_with_pose_A, initial_extrinsic_B, initial_extrinsic_B_prime)

R_B_prime = final_B_prime_extrinsic[0:3, :3]
T_B_prime = final_B_prime_extrinsic[0:3, 3]
print(f"\nFinal Rotation Matrix for image B':\n", R_B_prime)

# os.makedirs("/workspace/rendered_images/", exist_ok=True)
output_filename = image_B_prime_path.split("/")[-1]
os.makedirs(os.path.join(temp_path, "rendered_image"), exist_ok=True)
output_path = os.path.join(temp_path, "rendered_image", output_filename)


# --- Step 5: High-quality render with Blender Cycles ---
try:
    rendered_pil = render_with_blender(
        R=R_B_prime,
        T=T_B_prime,
        mesh_path=mesh_B_prime_path,
        output_path=output_path,
        resolution=1024,
        focal_length_mm=50.0,
    )
    print(f"Blender Cycles render saved: {output_path}")

    # --- Step 6: Composite with target using advanced blending ---
    target_img = Image.open(image_B_prime_path).convert("RGB").resize((1024, 1024))
    target_np = np.array(target_img)
    rendered_np = np.array(rendered_pil)  # RGBA

    composited = composite_rendered_onto_target(
        rendered_np, target_np, use_poisson=False, use_laplacian=True)

    composite_path = output_path.replace(".png", "_composited.png")
    Image.fromarray(composited).save(composite_path)
    print(f"Composited result saved: {composite_path}")

except (FileNotFoundError, RuntimeError) as e:
    print(f"Blender rendering unavailable ({e}), falling back to PyTorch3D...")
    mesh_renderer = HunyuanRenderer(mesh_B_prime_path, device)
    final_rendered_image = mesh_renderer.render(512, R_B_prime, T_B_prime)
    final_rendered_image = final_rendered_image[0, ..., :3].cpu().numpy()
    plt.imsave(output_path, final_rendered_image)
    print(f"PyTorch3D fallback render saved: {output_path}")

print(f"\nPipeline complete. Final render: {output_path}")
