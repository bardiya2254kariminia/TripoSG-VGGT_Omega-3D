"""
Model modules for 3D mesh generation, texturizing, and rendering.

Backends:
  hanyuan   — Hunyuan3D-2 mesh generation + pluggable texturizing + PyTorch3D rendering
  trellis   — TRELLIS.2-4B mesh generation + PyTorch3D rendering (optional)

Texturizing backends (for MeshGenerator):
  "hunyuan"   — Hunyuan3DPaintPipeline (UV paint on Hunyuan3D-2 mesh)
  "mvadapter" — MV-Adapter geometry-guided multi-view texturizing
  "triposg"   — TripoSG mesh backbone (geometry + colour in one pass)

Mesh backbone models (standalone geometry generators):
  TripoSGMeshBackbone  — generate watertight meshes from a single image via TripoSG

Backward compat alias:
  TripoSGTexturizer = TripoSGMeshBackbone
"""

from .hanyuan import (
    Renderer,
    MeshGenerator,
    MVAdapterTexturizer,
    TripoSGMeshBackbone,
    TripoSGTexturizer,   # backward compat alias
)

__all__ = [
    "MeshGenerator",
    "Renderer",
    "TripoSGMeshBackbone",
    "TripoSGTexturizer",       # backward compat — same as TripoSGMeshBackbone
    "MVAdapterTexturizer",
]
