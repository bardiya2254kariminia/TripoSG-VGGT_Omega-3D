"""
TRELLIS.2-specific utility helpers.

This module contains:
- build_trellis_mesh: One-shot helper to generate and export a .glb from an image
- default_glb_path: Derive the expected .glb path from an image path
- TrellisPipelineManager: Lazy singleton that keeps the heavy pipeline in memory
  across multiple calls so it is only loaded once per process.
"""

import os
from typing import Optional

_TRELLIS_IMPORT_ERROR = (
    "TRELLIS helpers require trellis2. Install with bash bash_scripts/trellis.sh."
)


def _get_trellis_mesh_generator():
    try:
        from camera.models.trellis import TrellisMeshGenerator
    except ImportError as exc:
        raise ImportError(_TRELLIS_IMPORT_ERROR) from exc
    return TrellisMeshGenerator


# Path helper
def default_glb_path(image_path: str) -> str:
    """
    Return the canonical .glb path for a given image path.

    Example:
        "final_assets/horse.png"  →  "final_assets/horse.glb"

    Args:
        image_path: Path to the source image.

    Returns:
        Corresponding .glb file path (extension replaced).
    """
    root, _ = os.path.splitext(image_path)
    return root + ".glb"


# One-shot mesh builder
def build_trellis_mesh(
    image_path: str,
    glb_path: Optional[str] = None,
    pipeline=None,
    device: str = "cuda",
    resolution: int = 1024,
    **export_kwargs,
) -> str:
    """
    Generate a .glb file from an image using TRELLIS.2-4B.

    This is a convenience wrapper around TrellisMeshGenerator that mirrors the
    one-shot style used by the Hunyuan pipeline in camera_pose_estimation.py.

    Args:
        image_path: Path to the source image.
        glb_path: Output .glb path.  Defaults to replacing the image extension.
        pipeline: Pre-loaded Trellis2ImageTo3DPipeline (avoids reloading).
        device: "cuda" or "cpu".
        resolution: O-Voxel grid resolution (512 / 1024 / 1536).
        **export_kwargs: Forwarded to TrellisMeshGenerator.save_mesh().

    Returns:
        Path to the saved .glb file.
    """
    if glb_path is None:
        glb_path = default_glb_path(image_path)

    TrellisMeshGenerator = _get_trellis_mesh_generator()
    generator = TrellisMeshGenerator(
        image_path=image_path,
        pipeline=pipeline,
        device=device,
        resolution=resolution,
    )
    generator.generate_mesh()
    generator.save_mesh(glb_path, **export_kwargs)
    return glb_path


# Pipeline singleton — reuse across multiple build_trellis_mesh calls
class TrellisPipelineManager:
    """
    Lazy singleton that loads Trellis2ImageTo3DPipeline once and keeps it alive.

    Use this when you need to generate multiple meshes in one session without
    paying the model-load cost each time.

    Example::

        manager = TrellisPipelineManager()

        for img_path, glb_path in pairs:
            build_trellis_mesh(img_path, glb_path, pipeline=manager.pipeline)

        manager.unload()   # free VRAM before loading VGGT
    """

    def __init__(self, model_id: str = "microsoft/TRELLIS.2-4B", device: str = "cuda"):
        """
        Args:
            model_id: HuggingFace model identifier.
            device: Device string.
        """
        self.model_id = model_id
        self.device = device
        self._pipeline = None

    @property
    def pipeline(self):
        """Load pipeline on first access."""
        if self._pipeline is None:
            from trellis2.pipelines import Trellis2ImageTo3DPipeline
            print(f"Loading {self.model_id} pipeline...")
            self._pipeline = Trellis2ImageTo3DPipeline.from_pretrained(self.model_id)
            self._pipeline.cuda()
            print("Pipeline ready.")
        return self._pipeline

    def unload(self):
        """Delete the pipeline and free GPU memory."""
        if self._pipeline is not None:
            import gc
            import torch
            del self._pipeline
            self._pipeline = None
            gc.collect()
            torch.cuda.empty_cache()
            print("TRELLIS.2 pipeline unloaded.")

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.unload()
