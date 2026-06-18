"""
Mesh generation inference script.

Pipeline per image:
  1. Remove background with BEN2         → <stem>_no_bg.png  (RGBA, transparent bg)
  2. Generate base 3D shape with Hunyuan3D-2 → <stem>_mesh.glb
  3. Texturize with TripoSG (default) or Hunyuan3DPaintPipeline
                                         → <stem>_mesh_textured.glb

Texture backends (--texture_backend):
  triposg  (default) — TripoSG (VAST-AI/TripoSG) generates a complete textured
                        mesh from the input image in one forward pass.
                        Hunyuan3D paint pipeline is NOT loaded.
  hunyuan             — Hunyuan3DPaintPipeline paints the Hunyuan-generated mesh.

Uses:
  camera.models.hanyuan.MeshGenerator  (this repo)
  BEN2  (github.com/PramaLLC/BEN2)
  TripoSG  (github.com/VAST-AI-Research/TripoSG)   [for triposg backend]

Usage:
    python mesh/inference.py --image path/to/image.png

    # Explicitly choose TripoSG texturizing (default):
    python mesh/inference.py --image photo.png --texture_backend triposg

    # Fall back to Hunyuan painting:
    python mesh/inference.py --image photo.png --texture_backend hunyuan

    # Skip texturizing entirely (bare mesh only):
    python mesh/inference.py --image photo.png --no_texture

Optional flags:
    --output_dir          DIR    Where to save outputs  (default: same dir as image)
    --no_texture                 Skip texturizing; save only the bare shape mesh
    --no_bg_removal              Skip BEN2; use original image as-is
    --hunyuan_model       HF_ID  Hunyuan3D-2 model ID  (default: tencent/Hunyuan3D-2)
    --texture_backend     STR    "triposg" or "hunyuan" (default: triposg)
    --triposg_model_path  DIR    Local TripoSG weights dir; auto-downloads if absent
    --triposg_steps       INT    TripoSG denoising steps  (default: 50)
    --triposg_guidance    FLOAT  TripoSG guidance scale   (default: 7.5)
    --triposg_seed        INT    TripoSG RNG seed         (default: 42)
"""

import argparse
import os,sys
import torch
from PIL import Image

# ── make this script runnable from anywhere ───────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT   = os.path.dirname(SCRIPT_DIR)
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
    Remove the background of an image with BEN2 and save the result as RGBA.

    Args:
        image_path:  Path to the original input image.
        output_path: Path where the RGBA no-background image will be saved.

    Returns:
        output_path
    """
    from ben2 import AutoModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("[BG] Loading BEN2 background removal model …")
    ben_model = AutoModel.from_pretrained("PramaLLC/BEN2")
    ben_model.to(device).eval()

    image = Image.open(image_path).convert("RGB")

    print("[BG] Running BEN2 inference …")
    foreground = ben_model.inference(image)   # returns RGBA PIL image

    foreground.save(output_path)
    print(f"[BG] Background removed → {output_path}")

    del ben_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return output_path


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a textured .glb mesh from a single image.\n"
            "Shape: Hunyuan3D-2.  Texture: TripoSG (default) or Hunyuan3D paint."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--image", required=True,
        help="Path to the input image (PNG / JPG).",
    )
    parser.add_argument(
        "--output_dir", default=None,
        help="Directory for output files. Defaults to the same directory as --image.",
    )
    parser.add_argument(
        "--no_texture", default=False,
        help="Skip texturizing; save only the bare Hunyuan3D shape mesh.",
    )
    parser.add_argument(
        "--no_bg_removal", default=False,
        help="Skip BEN2 background removal and use the original image directly.",
    )
    # ── Hunyuan shape model ───────────────────────────────────────────────────
    parser.add_argument(
        "--hunyuan_model", default="tencent/Hunyuan3D-2",
        help="HuggingFace model ID for Hunyuan3D-2 shape generation.",
    )
    # ── Texture backend ───────────────────────────────────────────────────────
    parser.add_argument(
        "--texture_backend", default="triposg",
        choices=["triposg", "hunyuan"],
        help=(
            "Texturizing backend: 'triposg' (default) uses TripoSG to generate "
            "a full textured mesh from the image; 'hunyuan' uses Hunyuan3DPaintPipeline "
            "to paint the Hunyuan-generated base mesh."
        ),
    )
    # ── TripoSG-specific ──────────────────────────────────────────────────────
    parser.add_argument(
        "--triposg_model_path", default=None,
        help=(
            "Local directory containing TripoSG weights. "
            "If absent, weights are auto-downloaded from VAST-AI/TripoSG on HuggingFace "
            f"into {os.path.join(REPO_ROOT, 'checkpoints', 'TripoSG')}."
        ),
    )
    parser.add_argument(
        "--triposg_steps", type=int, default=50,
        help="Number of TripoSG denoising steps (default: 50).",
    )
    parser.add_argument(
        "--triposg_guidance", type=float, default=7.5,
        help="TripoSG classifier-free guidance scale (default: 7.5).",
    )
    parser.add_argument(
        "--triposg_seed", type=int, default=42,
        help="TripoSG RNG seed for reproducibility (default: 42).",
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

    # ── resolve output paths ──────────────────────────────────────────────────
    if args.output_dir:
        output_dir = os.path.abspath(args.output_dir)
        os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = os.path.dirname(image_path)

    stem          = os.path.splitext(os.path.basename(image_path))[0]
    no_bg_path    = os.path.join(output_dir, f"{stem}_no_bg.png")
    bare_path     = os.path.join(output_dir, f"{stem}_mesh.glb")
    painted_path  = os.path.join(output_dir, f"{stem}_mesh_textured.glb")

    texture_label = (
        "disabled"
        if args.no_texture
        else f"{args.texture_backend} (steps={args.triposg_steps}, "
             f"guidance={args.triposg_guidance}, seed={args.triposg_seed})"
        if args.texture_backend == "triposg"
        else "hunyuan paint"
    )

    print("[INFO] ─────────────────────────────────────────────────────────")
    print(f"[INFO] Input image    : {image_path}")
    print(f"[INFO] Output dir     : {output_dir}")
    print(f"[INFO] BG removal     : {'disabled' if args.no_bg_removal else 'BEN2'}")
    print(f"[INFO] Shape model    : {args.hunyuan_model}")
    print(f"[INFO] Texture        : {texture_label}")
    print("[INFO] ─────────────────────────────────────────────────────────")

    # ── step 1: background removal ────────────────────────────────────────────
    if args.no_bg_removal:
        mesh_input_path = image_path
    else:
        mesh_input_path = remove_background(image_path, no_bg_path)

    # ── step 2: load Hunyuan3D-2 shape pipeline ───────────────────────────────
    from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

    print(f"[SHAPE] Loading Hunyuan3D-2 shape pipeline ({args.hunyuan_model}) …")
    pipeline_flow = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(args.hunyuan_model)

    # Load Hunyuan paint pipeline only when needed
    pipeline_paint = None
    if not args.no_texture and args.texture_backend == "hunyuan":
        from hy3dgen.texgen import Hunyuan3DPaintPipeline
        print("[TEXTURE] Loading Hunyuan3D paint pipeline …")
        pipeline_paint = Hunyuan3DPaintPipeline.from_pretrained(
            args.hunyuan_model, subfolder="hunyuan3d-paint-v2-0"
        )

    # ── step 3: build generator ───────────────────────────────────────────────
    generator = MeshGenerator(
        image_path=mesh_input_path,
        pipeline_flow=pipeline_flow,
        pipeline_paint=pipeline_paint,
        texture_backend=args.texture_backend,
        triposg_model_path=args.triposg_model_path,
        triposg_num_steps=args.triposg_steps,
        triposg_guidance_scale=args.triposg_guidance,
        triposg_seed=args.triposg_seed,
    )

    # ── step 4: generate shape mesh ───────────────────────────────────────────
    print("[SHAPE] Generating base mesh with Hunyuan3D-2 …")
    generator.generate_mesh()

    print(f"[SHAPE] Saving bare mesh → {bare_path}")
    generator.save_mesh(bare_path)

    # ── step 5: texturize ─────────────────────────────────────────────────────
    if not args.no_texture:
        if args.texture_backend == "triposg":
            print("[TEXTURE] Generating textured mesh with TripoSG …")
        else:
            print("[TEXTURE] Painting mesh with Hunyuan3D paint pipeline …")

        generator.generate_painted_mesh()
        print(f"[TEXTURE] Saving textured mesh → {painted_path}")
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
