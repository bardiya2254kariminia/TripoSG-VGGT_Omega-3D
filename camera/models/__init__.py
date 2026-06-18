"""
Model modules for 3D mesh generation, texturizing, and rendering.

Backends:
  hanyuan   — Hunyuan3D-2 mesh generation + TripoSG texturizing + PyTorch3D rendering
  trellis   — TRELLIS.2-4B mesh generation + PyTorch3D rendering (optional)
"""

from .hanyuan import HunyuanRenderer, MeshGenerator, TripoSGTexturizer

__all__ = [
    "MeshGenerator",
    "HunyuanRenderer",
    "TripoSGTexturizer",
]
