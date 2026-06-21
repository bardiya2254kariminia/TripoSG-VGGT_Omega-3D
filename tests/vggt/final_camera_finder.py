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

import torch
import numpy as np
import shutil
import os
import matplotlib.pyplot as plt
import gc
import sys

from pytorch3d.renderer import (
    MeshRenderer,
    MeshRasterizer,
    RasterizationSettings
)
from pytorch3d.renderer.blending import BlendParams

# VGGT imports (regular VGGT, not Omega)
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    vggt_path = os.path.abspath(os.path.join(current_dir, "..", "vggt-repo"))
    if vggt_path not in sys.path:
        sys.path.insert(0, vggt_path)

    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    print("Successfully imported VGGT modules.")
except ImportError as e:
    print(f"FATAL: Could not import VGGT modules. Error: {e}")
    sys.exit(1)




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
        # R (OpenCV): (B, 3, 3)
        # T (OpenCV): (B, 3, 1) or (B, 3)
        R_pytorch3d = R.clone().permute(0, 2, 1)  # Transpose
        T_pytorch3d = T.clone().squeeze(-1) if T.dim() == 3 else T.clone()

        # Invert X and Y axes for rotation and translation
        R_pytorch3d[:, :, :2] *= -1
        T_pytorch3d[:, :2] *= -1

        return R_pytorch3d, T_pytorch3d

    def get_pytorch3d_camera(self, R_pytorch3d, T_pytorch3d, focal_length, principal_point):
        """Creates a PyTorch3D PerspectiveCameras object."""
        # Add bounds checking (minor improvement)
        return PerspectiveCameras(
            device=self.device,
            R=R_pytorch3d,
            T=T_pytorch3d,
            focal_length=focal_length,
            principal_point=principal_point,
            image_size=(self.image_size, ),
            in_ndc=False
        )

    def render(self, image_size, R, T, K=None, focal_length=None, principal_point=None, blur_radius=0.0, location=[[0.0, 0.0, -3.0]]):
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
            device=self.device, location=[[0.0, 0.0, -3.0]])
        
        shader = SoftPhongShader(device=self.device, cameras=cameras, lights=lights,
                                        blend_params=BlendParams(background_color=(1.0, 1.0, 1.0)))
        
        renderer = MeshRenderer(
            rasterizer=rasterizer, shader=shader)
        
        rendered_output_tensor = renderer(self.mesh)
        
        return rendered_output_tensor


class CameraPoseFinder:
    def __init__(self, mesh_pth, image_size, device, dist=3.0, fov=40):
        self.original_image_size = image_size if isinstance(
            image_size, (list, tuple)) else (image_size, image_size)

        self.device = device
        self.dist = dist
        self.fov = fov
        self.mesh_renderer = HunyuanRenderer(mesh_pth, device)

        self.vggt_model = None

        self.view_path = "./temp_views"
        shutil.rmtree(self.view_path, ignore_errors=True)
        os.makedirs(self.view_path, exist_ok=True)

    def _load_vggt(self):
        if self.vggt_model is None:
            print("Loading VGGT model...")
            self.vggt_model = VGGT.from_pretrained(
                "facebook/VGGT-1B").to(self.device)
            print("VGGT model loaded.")

    def _unload_vggt(self):
        if self.vggt_model is not None:
            del self.vggt_model
            self.vggt_model = None
            gc.collect()
            torch.cuda.empty_cache()
            print("VGGT model unloaded.")

    def generate_initial_views(self, num_views, img_name):
        print(f"Generating {num_views} initial views for context...")
        self.known_image_paths = []

        azimuths = np.linspace(0, 360, int(
            np.sqrt(num_views*2)), endpoint=False)
        elevations = np.linspace(-45, 45,
                                 int(np.sqrt(num_views/2)), endpoint=True)

        num_of_views = 0
        all_Rs = []
        all_Ts = []
        print(self.original_image_size, type(self.original_image_size))
        os.makedirs("./temp_views", exist_ok=True)
        for azimuth in azimuths:
            for elevation in elevations:
                r, t = self.get_opencv_camera_matrix(
                    azimuth, elevation, self.dist)
                all_Rs.append(torch.from_numpy(r))
                all_Ts.append(torch.from_numpy(t))
                rendered_image = self.mesh_renderer.render(
                    image_size=self.original_image_size,
                    R=r,
                    T=t
                )
                rendered_image = rendered_image[0, ..., :3].cpu().numpy()

                plt.imsave(
                    f'temp_views/{img_name}_{num_of_views}.png', rendered_image)
                self.known_image_paths.append(
                    f'temp_views/{img_name}_{num_of_views}.png')
                num_of_views += 1

        print(f"Generated {num_views} views.")
        return all_Rs, all_Ts

    def get_vggt_initial_guess(self, target_image_path, all_Rs, all_Ts, img_name):
        self._load_vggt()

        all_image_paths = self.known_image_paths + [target_image_path]

        target_index = len(all_image_paths) - 1

        images = load_and_preprocess_images(all_image_paths).to(self.device)

        print("Running VGGT for initial pose estimation...")
        with torch.no_grad():
            dtype = torch.bfloat16 if torch.cuda.get_device_capability()[
                0] >= 8 else torch.float16
            with torch.cuda.amp.autocast(dtype=dtype):
                images_batch = images.unsqueeze(0)
                aggregated_tokens_list, _ = self.vggt_model.aggregator(
                    images_batch)

            pose_enc = self.vggt_model.camera_head(
                aggregated_tokens_list)[-1]

            extrinsic, intrinsic = pose_encoding_to_extri_intri(
                pose_enc, images_batch.shape[-2:])

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

        os.makedirs("./temp_views2", exist_ok=True)
        for i in range(len(all_image_paths)):
            rendered_image = self.mesh_renderer.render(
                    image_size=self.original_image_size, R=extrinsics_new[i][0:3, :3].cpu().numpy(), T=extrinsics_new[i][0:3, 3].cpu().numpy())
            rendered_image = rendered_image[0, ..., :3].cpu().numpy()
            plt.imsave(f'temp_views2/{img_name}_{i}.png', rendered_image)

        self._unload_vggt()

        return target_extrinsic, target_intrinsic

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


def compute_new_pose_from_relative(view_matrix_B_with_pose_A,view_matrix_B,view_matrix_B_prime):
    # Ensure inputs are NumPy arrays
    V_B_with_pose_A = view_matrix_B_with_pose_A.cpu().numpy()
    V_B = view_matrix_B.cpu().numpy()
    V_B_prime = view_matrix_B_prime.cpu().numpy()
    # (V_B * V_B_pose_A^-1) * V_B'_pose_A'
    try:
        V_B_with_pose_A_inv = np.linalg.inv(V_B_with_pose_A)
    except np.linalg.LinAlgError:
        print("Error: view_matrix_B_with_pose_A is not invertible.")
        return None

    delta_transform = V_B @ V_B_with_pose_A_inv
    V_B_prime = delta_transform @ V_B_prime

    return V_B_prime



# Main script to run the pipeline
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_size = (512, 512)
num_views = 100

# define the images' paths
image_B_with_pose_A_path = "/workspace/TripoSG-VGGT_Omega-3D/images_with_glb/B.png"
image_B_path = "/workspace/TripoSG-VGGT_Omega-3D/images_with_glb/input.png"
image_B_prime_path = "/workspace/TripoSG-VGGT_Omega-3D/images_with_glb/B_prime.png"

# define the meshes' paths
mesh_B_with_pose_A_path = image_B_with_pose_A_path.replace(".png", ".glb")
mesh_B_path = image_B_path.replace(".png", ".glb")
mesh_B_prime_path = image_B_prime_path.replace(".png", ".glb")

# finding the camera parameters for B with pose A
pose_finder = CameraPoseFinder(mesh_B_with_pose_A_path, 512, device)
all_Rs, all_Ts = pose_finder.generate_initial_views(num_views, "B_with_pose_A")
initial_extrinsic_B_with_pose_A, initial_intrinsic = pose_finder.get_vggt_initial_guess(image_B_with_pose_A_path, all_Rs, all_Ts, "B_with_pose_A")
print(
    f"\nVGGT Initial Extrinsic (3x4) for image B_with_pose_A:\n", initial_extrinsic_B_with_pose_A)
print(
    f"\nVGGT Initial Intrinsic (3x3) for image B_with_pose_A:\n", initial_intrinsic)

# finding the camera parameters for B
pose_finder = CameraPoseFinder(mesh_B_path, 512, device)
all_Rs, all_Ts = pose_finder.generate_initial_views(num_views, "B")
initial_extrinsic_B, initial_intrinsic = pose_finder.get_vggt_initial_guess(image_B_path, all_Rs, all_Ts, "B")
print(
    f"\nVGGT Initial Extrinsic (3x4) for image B:\n", initial_extrinsic_B)
print(
    f"\nVGGT Initial Intrinsic (3x3) for image B:\n", initial_intrinsic)

# finding the camera parameters for B prime
pose_finder = CameraPoseFinder(mesh_B_prime_path, 512, device)
all_Rs, all_Ts = pose_finder.generate_initial_views(num_views, "B_prime")
initial_extrinsic_B_prime, initial_intrinsic = pose_finder.get_vggt_initial_guess(image_B_prime_path, all_Rs, all_Ts, "B_prime")
print(
    f"\nVGGT Initial Extrinsic (3x4) for image B':\n", initial_extrinsic_B_prime)
print(
    f"\nVGGT Initial Intrinsic (3x3) for image B':\n", initial_intrinsic)

# finding the final camera parameters for B prime (relative change)
final_B_prime_extrinsic = compute_new_pose_from_relative(initial_extrinsic_B_with_pose_A, initial_extrinsic_B, initial_extrinsic_B_prime)

target_img_pil = Image.open(image_B_prime_path).convert(
    "RGBA").resize(image_size)
target_img_tensor = torch.from_numpy(np.array(target_img_pil)).to(device)

R_B_prime = final_B_prime_extrinsic[0:3, :3]
T_B_prime = final_B_prime_extrinsic[0:3, 3]
print(
    f"\nfinal Rotation Matrix for image B':\n", R_B_prime)

mesh_renderer = HunyuanRenderer(mesh_B_prime_path, device)

final_rendered_image = mesh_renderer.render(512, R_B_prime, T_B_prime)
final_rendered_image = final_rendered_image[0, ..., :3].cpu().numpy()
os.makedirs("/workspace/rendered_images/", exist_ok=True)
plt.imsave(f'/workspace/rendered_images/{image_B_prime_path.split("/")[-1]}', final_rendered_image)
print(
    f"Final optimized render saved to rendered_images/{image_B_prime_path.split('/')[-1]}")



# base_path = "final_assets/"

# image_paths = ["grey_mouse_jumping.png", "grey_mouse_with_pose_A.png", "grey_mouse.png"]
# for path in image_paths:
#     full_path = base_path + path
#     mesh_path = base_path + path.replace(".png", ".glb")
#     print(f"\nGenerating the mesh for image {full_path}\n")
#     mesh_generator = MeshGenerator(full_path)
#     mesh_generator.generate_mesh()
#     mesh_generator.generate_painted_mesh()
#     mesh_generator.save_painted_mesh(mesh_path)
#     print(f"\nsave the painted mesh for {mesh_path}\n")
