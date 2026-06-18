"""
Main script for camera pose estimation and mesh rendering.

This script demonstrates the complete pipeline:
1. Generate 3D meshes from 2D images
2. Estimate camera poses using VGGT
3. Transfer poses between different objects
4. Render final results
"""

import torch
import os
import matplotlib.pyplot as plt
import gc
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
from hy3dgen.texgen import Hunyuan3DPaintPipeline

from camera import (
    MeshGenerator,
    HunyuanRenderer,
    CameraPoseFinder,
    compute_new_pose_from_relative,
    create_pose_finder
)


# Load Hunyuan3D pipelines
pipeline_flow = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained('tencent/Hunyuan3D-2')
pipeline_paint = Hunyuan3DPaintPipeline.from_pretrained('tencent/Hunyuan3D-2')



def main():
    """Main execution function."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_size = (512, 512)
    num_views = 50

    # Image paths
    image_B_with_pose_A_path = "final_assets/horse_with_pose_A.png"
    image_B_path = "final_assets/horse.png"
    image_B_prime_path = "final_assets/horse_stretching.png"

    # Mesh paths
    mesh_B_with_pose_A_path = image_B_with_pose_A_path.replace(".png", ".glb")
    mesh_B_path = image_B_path.replace(".png", ".glb")
    mesh_B_prime_path = image_B_prime_path.replace(".png", ".glb")

    os.makedirs("rendered_images", exist_ok=True)

    # Generate meshes
    for img_path, mesh_path in [
        (image_B_with_pose_A_path, mesh_B_with_pose_A_path),
        (image_B_path, mesh_B_path),
        (image_B_prime_path, mesh_B_prime_path),
    ]:
        print(f"\nGenerating mesh for {img_path}")
        mg = MeshGenerator(img_path, pipeline_flow, pipeline_paint)
        mg.generate_mesh()
        mg.generate_painted_mesh()
        mg.save_painted_mesh(mesh_path)
        print(f"Saved painted mesh → {mesh_path}")

    # Free Hunyuan3D before loading VGGT
    global pipeline_flow, pipeline_paint
    del pipeline_flow, pipeline_paint
    pipeline_flow = pipeline_paint = None
    gc.collect()
    torch.cuda.empty_cache()

    # Find camera pose for B_with_pose_A
    print("\n── Pose estimation for B_with_pose_A ──")
    pose_finder = CameraPoseFinder(mesh_B_with_pose_A_path, image_size, device)
    all_Rs, all_Ts = pose_finder.generate_initial_views(num_views, "B_with_pose_A")
    initial_extrinsic_B_with_pose_A, _ = pose_finder.get_vggt_initial_guess(
        image_B_with_pose_A_path, all_Rs, all_Ts, "B_with_pose_A")
    print(f"\nExtrinsic B_with_pose_A:\n{initial_extrinsic_B_with_pose_A}")

    # Find camera pose for B
    print("\n── Pose estimation for B ──")
    pose_finder = CameraPoseFinder(mesh_B_path, image_size, device)
    all_Rs, all_Ts = pose_finder.generate_initial_views(num_views, "B")
    initial_extrinsic_B, _ = pose_finder.get_vggt_initial_guess(
        image_B_path, all_Rs, all_Ts, "B")
    print(f"\nExtrinsic B:\n{initial_extrinsic_B}")

    # Find camera pose for B'
    print("\n── Pose estimation for B' ──")
    pose_finder = CameraPoseFinder(mesh_B_prime_path, image_size, device)
    all_Rs, all_Ts = pose_finder.generate_initial_views(num_views, "B_prime")
    initial_extrinsic_B_prime, _ = pose_finder.get_vggt_initial_guess(
        image_B_prime_path, all_Rs, all_Ts, "B_prime")
    print(f"\nExtrinsic B':\n{initial_extrinsic_B_prime}")

    # Transfer pose
    print("\n── Computing final pose for B' ──")
    final_B_prime_extrinsic = compute_new_pose_from_relative(
        initial_extrinsic_B_with_pose_A,
        initial_extrinsic_B,
        initial_extrinsic_B_prime
    )

    R_B_prime = final_B_prime_extrinsic[0:3, :3]
    T_B_prime = final_B_prime_extrinsic[0:3, 3]
    print(f"\nFinal R for B':\n{R_B_prime}")
    print(f"Final T for B':\n{T_B_prime}")

    # Final render
    mesh_renderer = HunyuanRenderer(mesh_B_prime_path, device)
    final_rendered_image = mesh_renderer.render(image_size[0], R_B_prime, T_B_prime)
    final_rendered_image = final_rendered_image[0, ..., :3].cpu().numpy()

    out_name = image_B_prime_path.split("/")[-1]
    plt.imsave(f'rendered_images/{out_name}', final_rendered_image)
    print(f"\nFinal render saved → rendered_images/{out_name}")


if __name__ == "__main__":
    main()