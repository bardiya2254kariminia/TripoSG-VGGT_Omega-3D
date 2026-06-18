# Camera Backend Selection Guide

This guide explains how to choose between different backends for mesh generation, rendering, and pose estimation in the camera pose estimation pipeline.

## Overview

The system now supports **2×2×2 = 8 combinations**:

```
Pose Model × Renderer Backend × Mesh Generator
    ↓              ↓                   ↓
 VGGT or      Hanyuan or          Hunyuan3D or
VGGT-Omega     TRELLIS             TRELLIS.2
```

---

## 1. Pose Estimation Models

### VGGT (Default)
- **Model**: Facebook's VGGT-1B
- **Class**: `CameraPoseFinder`
- **Setup**: `bash vggt_setup.sh`
- **Pros**: 
  - Standard, well-tested
  - Automatic model download from HuggingFace
  - Faster inference
- **Use when**: Default choice for most tasks

### VGGT-Omega (Advanced)
- **Model**: Facebook's VGGT-Omega
- **Class**: `Omega_CameraPoseFinder`
- **Setup**: `bash vggt_omega_setup.sh` + manual checkpoint download
- **Pros**:
  - Higher accuracy on complex scenes
  - Configurable alignment module
  - Custom resolution support
- **Use when**: Need maximum accuracy and have checkpoint file

---

## 2. Renderer Backends

### Hanyuan (Default)
- **Model**: Hunyuan3D-2 + PyTorch3D
- **Classes**: `MeshGenerator`, `HunyuanRenderer`
- **Pros**:
  - Established pipeline
  - Good quality meshes
- **Use when**: Default choice, proven workflow

### TRELLIS (Recommended for complex geometry)
- **Model**: TRELLIS.2-4B + PyTorch3D
- **Classes**: `TrellisMeshGenerator`, `TrellisRenderer`
- **Setup**: `bash trellis.sh`
- **Pros**:
  - Native O-Voxel representation
  - Handles arbitrary topology (open surfaces, non-manifold)
  - Full PBR materials with transparency
  - Faster generation at 512³ resolution (~3s on H100)
- **Resolution options**:
  - `512³` — 3 seconds (fast, good quality)
  - `1024³` — 17 seconds (balanced, default)
  - `1536³` — 60 seconds (best quality)
- **Use when**: 
  - Complex geometry (cloth, hair, open surfaces)
  - Need PBR materials
  - Speed is important (use 512³)

---

## 3. Quick Start Examples

### Example 1: Standard Pipeline (VGGT + Hanyuan)
```python
from camera import create_pose_finder

# Default: VGGT pose model + Hanyuan renderer
pose_finder = create_pose_finder(
    mesh_pth="horse.glb",
    image_size=512,
    device="cuda",
    pose_model="vggt",      # default
    backend="hanyuan"       # default
)
```

### Example 2: Fast TRELLIS Pipeline
```python
from camera import create_pose_finder, TrellisMeshGenerator

# Generate mesh with TRELLIS at 512³ (fast)
generator = TrellisMeshGenerator(
    image_path="horse.png",
    device="cuda",
    resolution=512  # 3 seconds on H100
)
generator.generate_mesh()
generator.save_mesh("horse.glb")

# Use VGGT with TRELLIS renderer
pose_finder = create_pose_finder(
    mesh_pth="horse.glb",
    image_size=512,
    device="cuda",
    pose_model="vggt",
    backend="trellis"  # Use TRELLIS renderer
)
```

### Example 3: Maximum Accuracy (VGGT-Omega + TRELLIS)
```python
from camera import create_pose_finder

pose_finder = create_pose_finder(
    mesh_pth="horse.glb",
    image_size=512,
    device="cuda",
    pose_model="vggt_omega",
    backend="trellis",
    checkpoint_path="./checkpoints/vggt_omega_1b_512.pt",
    model_image_resolution=512,
    enable_alignment=True
)
```

### Example 4: Batch Processing with TRELLIS Pipeline Manager
```python
from camera import TrellisPipelineManager, build_trellis_mesh, create_pose_finder

# Keep TRELLIS pipeline in memory for multiple meshes
with TrellisPipelineManager() as manager:
    for img_path in ["horse1.png", "horse2.png", "horse3.png"]:
        glb_path = build_trellis_mesh(
            image_path=img_path,
            pipeline=manager.pipeline,  # Reuse loaded pipeline
            resolution=512
        )
        
        # Now use for pose estimation
        pose_finder = create_pose_finder(
            mesh_pth=glb_path,
            image_size=512,
            device="cuda",
            pose_model="vggt",
            backend="trellis"
        )
        # ... rest of pose estimation
```

---

## 4. Setup Instructions

### Install VGGT
```bash
bash vggt_setup.sh
```
This will:
- Clone the VGGT repository
- Install dependencies
- Download VGGT-1B model from HuggingFace

### Install VGGT-Omega
```bash
bash vggt_omega_setup.sh
```
Then **manually download** the checkpoint:
1. Visit VGGT-Omega release page
2. Download `vggt_omega_1b_512.pt` (or similar)
3. Place in `./checkpoints/`

### Install TRELLIS.2
```bash
bash trellis.sh
```
This will:
- Clone TRELLIS.2 repository
- Create conda environment
- Install PyTorch + CUDA
- Install nvdiffrast and o_voxel
- Log in to HuggingFace Hub

**System Requirements**:
- Linux only
- NVIDIA GPU with ≥24GB VRAM
- CUDA Toolkit 12.4

---

## 5. Resolution Guidelines

### TRELLIS O-Voxel Resolution

| Resolution | Time (H100) | Use Case |
|------------|-------------|----------|
| 512³       | ~3s         | Fast prototyping, batch processing |
| 1024³      | ~17s        | Balanced quality/speed (default) |
| 1536³      | ~60s        | Maximum quality, final renders |

### Render Image Size

Separate from voxel resolution! Set via `image_size` parameter:

```python
# Generate at 512³ voxels (fast 3D gen)
generator = TrellisMeshGenerator(
    image_path="input.png",
    resolution=512  # O-Voxel grid
)

# But render at high resolution (2D output)
renderer = TrellisRenderer("output.glb", device="cuda")
image = renderer.render(
    image_size=2048,  # Output 2048×2048 pixels
    R=R, T=T
)
```

---

## 6. API Reference

### Factory Function (Recommended)

```python
create_pose_finder(
    mesh_pth: str,
    image_size: int | tuple,
    device: str,
    pose_model: Literal["vggt", "vggt_omega"] = "vggt",
    backend: Literal["hanyuan", "trellis"] = "hanyuan",
    dist: float = 3.0,
    fov: float = 40,
    # VGGT-Omega only:
    checkpoint_path: str = "...",
    model_image_resolution: int = 512,
    enable_alignment: bool = False,
)
```

### Direct Class Usage

```python
# VGGT-based (standard)
from camera import CameraPoseFinder

pose_finder = CameraPoseFinder(
    mesh_pth="mesh.glb",
    image_size=512,
    device="cuda",
    backend="trellis"  # or "hanyuan"
)

# VGGT-Omega-based (advanced)
from camera import Omega_CameraPoseFinder

pose_finder = Omega_CameraPoseFinder(
    mesh_pth="mesh.glb",
    image_size=512,
    device="cuda",
    checkpoint_path="./checkpoints/vggt_omega_1b_512.pt",
    backend="trellis"  # or "hanyuan"
)
```

---

## 7. Decision Tree

```
Need maximum accuracy?
├─ Yes → Use VGGT-Omega (need checkpoint)
└─ No → Use VGGT (default, auto-download)

Complex geometry (open surfaces, transparency)?
├─ Yes → Use TRELLIS backend
└─ No → Use Hanyuan backend

Need speed?
├─ Yes → Use TRELLIS at resolution=512
└─ No → Use default settings

Batch processing many meshes?
├─ Yes → Use TrellisPipelineManager context manager
└─ No → Use standard workflow
```

---

## 8. Troubleshooting

### "TRELLIS.2 model not found"
```bash
# Make sure you've logged into HuggingFace
huggingface-cli login

# Activate the trellis2 conda environment
conda activate trellis2
```

### "VGGT-Omega checkpoint not found"
The checkpoint must be **manually downloaded**. Check the error message for the expected path and download from the VGGT-Omega releases.

### "nvdiffrast error" (TRELLIS)
TRELLIS requires nvdiffrast. Make sure:
```bash
pip install git+https://github.com/NVlabs/nvdiffrast.git
```

### Import errors
Make sure the repositories are in the parent directory:
```
SIA3D-Camera-inversion/
├── camera/
├── vggt/              ← git clone here
├── vggt-omega/        ← git clone here
└── TRELLIS.2/         ← git clone here
```

---

## 9. Performance Comparison

| Config | Mesh Gen | Pose Est | Total | Quality |
|--------|----------|----------|-------|---------|
| Hunyuan + VGGT | ~30s | ~5s | ~35s | Good |
| TRELLIS-512 + VGGT | ~3s | ~5s | ~8s | Good |
| TRELLIS-1024 + VGGT | ~17s | ~5s | ~22s | Better |
| TRELLIS-1536 + VGGT-Omega | ~60s | ~8s | ~68s | Best |

*Times measured on NVIDIA H100 GPU*

---

## Summary

- **Default**: VGGT + Hanyuan (proven, stable)
- **Fast**: VGGT + TRELLIS-512 (4× faster mesh gen)
- **Quality**: VGGT-Omega + TRELLIS-1536 (best results)
- **Complex geometry**: Always use TRELLIS backend
