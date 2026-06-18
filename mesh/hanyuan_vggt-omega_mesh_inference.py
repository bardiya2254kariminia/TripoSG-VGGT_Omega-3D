"""
Mesh generation inference script — hanyuan backend (Hunyuan3D-2).

Pipeline per image:
  1. Remove background with BEN2 → saves <stem>_no_bg.png (RGBA, transparent bg)
  2. Generate base 3D mesh with Hunyuan3DDiTFlowMatchingPipeline
  3. Apply texture painting with Hunyuan3DPaintPipeline
  4. Save <stem>_mesh.glb and <stem>_mesh_textured.glb

Uses:
  camera.models.hanyuan.MeshGenerator  (this repo)
  BEN2  (github.com/PramaLLC/BEN2)

Usage:
    python mesh/hanyuan_vggt-omega_mesh_inference.py --image path/to/image.png

Optional flags:
    --output_dir   DIR    Where to save outputs        (default: same dir as image)
    --no_texture          Skip texture painting; save only the bare mesh
    --no_bg_removal       Skip BEN2; use the original image as-is
    --model        HF_ID  HuggingFace model id          (default: tencent/Hunyuan3D-2)
"""

import argparse
import os
import sys

import torch
from PIL import Image

# ── make this script runnable from anywhere ───────────────────────────────────
# Add the repo root (for `camera.*`) and the cloned Hunyuan3D-2 directory (for
# `hy3dgen`, as a fallback to the editable install) to the import path.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
HUNYUAN_DIR = os.path.join(REPO_ROOT, "Hunyuan3D-2")
for _path in (REPO_ROOT, HUNYUAN_DIR):
    if os.path.isdir(_path) and _path not in sys.path:
        sys.path.insert(0, _path)

from camera.models.hanyuan import MeshGenerator


# ─────────────────────────────────────────────────────────────────────────────
# Background removal
# ─────────────────────────────────────────────────────────────────────────────

def remove_background(image_path: str, output_path: str) -> str:
    """
    Remove the background of an image using BEN2 and save the result as RGBA.

    The saved PNG has a transparent background so Hunyuan3D-2 can use the
    alpha channel to focus on the foreground object.

    Args:
        image_path:  Path to the original input image.
        output_path: Path where the RGBA no-background image will be saved.

    Returns:
        output_path
    """
    from ben2 import AutoModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("[INFO] Loading BEN2 background removal model …")
    ben_model = AutoModel.from_pretrained("PramaLLC/BEN2")
    ben_model.to(device).eval()

    image = Image.open(image_path).convert("RGB")

    print("[INFO] Running BEN2 inference …")
    foreground = ben_model.inference(image)  # returns RGBA PIL image

    foreground.save(output_path)
    print(f"[INFO] Background removed → {output_path}")

    del ben_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return output_path


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a textured .glb mesh from a single image using Hunyuan3D-2 + BEN2."
    )
    parser.add_argument(
        "--image", required=True,
        help="Path to the input image (PNG / JPG)."
    )
    parser.add_argument(
        "--output_dir", default=None,
        help="Directory where output files are saved. Defaults to the same directory as --image."
    )
    parser.add_argument(
        "--no_texture", default=False,
        help="Skip texture painting and save only the bare mesh."
    )
    parser.add_argument(
        "--no_bg_removal", action="store_true",
        help="Skip BEN2 background removal and use the original image directly."
    )
    parser.add_argument(
        "--model", default="tencent/Hunyuan3D-2",
        help="HuggingFace model ID for Hunyuan3D-2 (default: tencent/Hunyuan3D-2)."
    )
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    image_path = os.path.abspath(args.image)
    if not os.path.isfile(image_path):
        sys.exit(f"[ERROR] Image not found: {image_path}")

    # resolve output directory
    if args.output_dir:
        output_dir = os.path.abspath(args.output_dir)
        os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = os.path.dirname(image_path)

    stem = os.path.splitext(os.path.basename(image_path))[0]
    no_bg_path   = os.path.join(output_dir, f"{stem}_no_bg.png")
    bare_path    = os.path.join(output_dir, f"{stem}_mesh.glb")
    painted_path = os.path.join(output_dir, f"{stem}_mesh_textured.glb")

    print(f"[INFO] Input image    : {image_path}")
    print(f"[INFO] Output dir     : {output_dir}")
    print(f"[INFO] Model          : {args.model}")
    print(f"[INFO] BG removal     : {'disabled' if args.no_bg_removal else 'BEN2'}")
    print(f"[INFO] Texture paint  : {'disabled' if args.no_texture else 'enabled'}")

    # ── step 1: background removal ────────────────────────────────────────────
    if args.no_bg_removal:
        mesh_input_path = image_path
    else:
        mesh_input_path = remove_background(image_path, no_bg_path)

    # ── step 2: load Hunyuan3D-2 pipelines ───────────────────────────────────
    from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
    from hy3dgen.texgen import Hunyuan3DPaintPipeline

    print("[INFO] Loading shape generation pipeline …")
    pipeline_flow = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(args.model)

    pipeline_paint = None
    if not args.no_texture:
        print("[INFO] Loading texture painting pipeline …")
        pipeline_paint = Hunyuan3DPaintPipeline.from_pretrained(args.model,subfolder="hunyuan3d-paint-v2-0")

    # ── step 3: generate mesh ─────────────────────────────────────────────────
    generator = MeshGenerator(
        image_path=mesh_input_path,
        pipeline_flow=pipeline_flow,
        pipeline_paint=pipeline_paint,
    )

    print("[INFO] Generating base mesh …")
    generator.generate_mesh()

    print(f"[INFO] Saving bare mesh → {bare_path}")
    generator.save_mesh(bare_path)

    if not args.no_texture:
        print("[INFO] Applying texture painting …")
        generator.generate_painted_mesh()
        print(f"[INFO] Saving textured mesh → {painted_path}")
        generator.save_painted_mesh(painted_path)

    # ── summary ───────────────────────────────────────────────────────────────
    print("\n[INFO] Done.")
    if not args.no_bg_removal:
        print(f"  bg-removed image : {no_bg_path}")
    print(f"  bare mesh        : {bare_path}")
    if not args.no_texture:
        print(f"  textured mesh    : {painted_path}")


if __name__ == "__main__":
    main()
