"""
Mesh generation inference script.

Pipeline per image:
  1. Remove background with BEN2              → <stem>_no_bg.png  (RGBA, transparent bg)
  2. Generate base 3D mesh with mesh backbone → <stem>_mesh.glb
  3. Texturize (optional)                     → <stem>_mesh_textured.glb

Mesh backbones (--mesh_backbone):
  triposg  (default) — TripoSGMeshBackbone generates watertight mesh geometry
                       in one forward pass from the image.
  hunyuan            — Hunyuan3D-2 shape generation (two-step: shape then paint).

Texture backends (--texture_backend):
  mvadapter (default) — MV-Adapter applies geometry-guided multi-view texturing.
                        Requires the mesh backbone output. Image-conditioned by default
                        (uses the BEN2 background-removed image as reference).
  hunyuan             — Hunyuan3DPaintPipeline paints the Hunyuan-generated base mesh
                        (only works when mesh_backbone="hunyuan").
  none                — Skip texturing; save only the bare mesh.

Uses:
  camera.models.hanyuan.MeshGenerator, TripoSGMeshBackbone, MVAdapterTexturizer
  BEN2  (github.com/PramaLLC/BEN2)
  MV-Adapter  (github.com/huanngzh/MV-Adapter)
  TripoSG  (github.com/VAST-AI-Research/TripoSG)

Usage:
    # Default: TripoSG mesh → MVAdapter texturing
    python mesh/inference.py --image path/to/image.png

    # Hunyuan mesh → Hunyuan paint texturing:
    python mesh/inference.py --image photo.png --mesh_backbone hunyuan --texture_backend hunyuan

    # TripoSG mesh → no texturing (bare geometry only):
    python mesh/inference.py --image photo.png --mesh_backbone triposg --texture_backend none

    # Hunyuan mesh → MVAdapter texturing:
    python mesh/inference.py --image photo.png --mesh_backbone hunyuan --texture_backend mvadapter

    # MVAdapter with SD2.1 (lower VRAM):
    python mesh/inference.py --image photo.png --mvadapter_variant sd21

Optional flags:
    --output_dir               DIR    Where to save outputs  (default: same dir as image)
    --no_bg_removal                   Skip BEN2; use original image as-is
    --mesh_backbone            STR    "triposg" (default) or "hunyuan"
    --texture_backend          STR    "mvadapter" (default), "hunyuan", or "none"

  Hunyuan mesh backbone options:
    --hunyuan_model            HF_ID  Hunyuan3D-2 model ID  (default: tencent/Hunyuan3D-2)

  MVAdapter texturing options:
    --mvadapter_variant        STR    "sdxl" (default, 768px) or "sd21" (512px, lower VRAM)
    --mvadapter_steps          INT    Multi-view denoising steps (default: 50)
    --mvadapter_guidance       FLOAT  CFG scale (default: 3.0 for image-conditioned)
    --mvadapter_seed           INT    RNG seed (-1 = random, default: -1)
    --mvadapter_text           STR    Optional text prompt (image-conditioned by default)
    --mvadapter_checkpoints    DIR    Directory with RealESRGAN and LaMa weights
    --mvadapter_repo           DIR    Path to the cloned MV-Adapter repo

  TripoSG mesh backbone options:
    --triposg_model_path  DIR    Local TripoSG weights dir; auto-downloads if absent
    --triposg_steps       INT    TripoSG denoising steps  (default: 50)
    --triposg_guidance    FLOAT  TripoSG guidance scale   (default: 7.5)
    --triposg_seed        INT    TripoSG RNG seed         (default: 42)
"""

import argparse
import os
import sys
import torch
from PIL import Image

# ── make this script runnable from anywhere ───────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT   = os.path.dirname(SCRIPT_DIR)
HUNYUAN_DIR = os.path.join(REPO_ROOT, "Hunyuan3D-2")
for _path in (REPO_ROOT, HUNYUAN_DIR):
    if os.path.isdir(_path) and _path not in sys.path:
        sys.path.insert(0, _path)

from camera.models.hanyuan import MeshGenerator, TripoSGMeshBackbone, MVAdapterTexturizer


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

    print("[BG] Loading BEN2 background removal model ...")
    ben_model = AutoModel.from_pretrained("PramaLLC/BEN2")
    ben_model.to(device).eval()

    image = Image.open(image_path).convert("RGB")

    print("[BG] Running BEN2 inference ...")
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
            "Default: TripoSG mesh backbone → MV-Adapter texturing."
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
        "--no_bg_removal", action="store_true", default=False,
        help="Skip BEN2 background removal and use the original image directly.",
    )

    # ── Mesh backbone ─────────────────────────────────────────────────────────
    parser.add_argument(
        "--mesh_backbone", default="triposg",
        choices=["triposg", "hunyuan"],
        help=(
            "Mesh generation backbone:\n"
            "  'triposg' (default) — TripoSG generates watertight mesh geometry.\n"
            "  'hunyuan'           — Hunyuan3D-2 shape generation."
        ),
    )

    # ── Texture backend ───────────────────────────────────────────────────────
    parser.add_argument(
        "--texture_backend", default="mvadapter",
        choices=["mvadapter", "hunyuan", "none"],
        help=(
            "Texturizing backend:\n"
            "  'mvadapter' (default) — MV-Adapter geometry-guided multi-view texturing.\n"
            "  'hunyuan'             — Hunyuan3DPaintPipeline (requires mesh_backbone='hunyuan').\n"
            "  'none'                — Skip texturing."
        ),
    )

    # ── Hunyuan backbone options ──────────────────────────────────────────────
    hunyuan_group = parser.add_argument_group("Hunyuan mesh backbone options")
    hunyuan_group.add_argument(
        "--hunyuan_model", default="tencent/Hunyuan3D-2",
        help="HuggingFace model ID for Hunyuan3D-2 shape generation.",
    )

    # ── MVAdapter texturing options ───────────────────────────────────────────
    mvadapter_group = parser.add_argument_group("MVAdapter texturing options")
    mvadapter_group.add_argument(
        "--mvadapter_variant", default="sd21", choices=["sdxl", "sd21"],
        help=(
            "MVAdapter base model variant: 'sd21' (default, 512px, fast, <10GB VRAM) "
            "or 'sdxl' (768px, higher quality, ~16GB VRAM)."
        ),
    )
    mvadapter_group.add_argument(
        "--mvadapter_steps", type=int, default=25,
        help="MVAdapter multi-view denoising steps (default: 25).",
    )
    mvadapter_group.add_argument(
        "--mvadapter_guidance", type=float, default=3.0,
        help="MVAdapter CFG scale (default: 3.0 for image-conditioned).",
    )
    mvadapter_group.add_argument(
        "--mvadapter_seed", type=int, default=-1,
        help="MVAdapter RNG seed (-1 = random, default: -1).",
    )
    mvadapter_group.add_argument(
        "--mvadapter_text", default=None,
        help=(
            "Optional text prompt for MVAdapter. If not set, uses image-conditioned mode "
            "(reference image from BEN2 background removal)."
        ),
    )
    mvadapter_group.add_argument(
        "--mvadapter_checkpoints", default=None,
        help=(
            "Directory containing RealESRGAN_x2plus.pth and big-lama.pt. "
            f"Defaults to {os.path.join(REPO_ROOT, 'checkpoints')} "
            "(files are auto-downloaded there if absent)."
        ),
    )
    mvadapter_group.add_argument(
        "--mvadapter_repo", default=None,
        help=(
            "Path to the cloned MV-Adapter repository. "
            f"Auto-detected from {os.path.join(REPO_ROOT, 'mvadapter-repo')} if absent."
        ),
    )

    # ── TripoSG backbone options ──────────────────────────────────────────────
    triposg_group = parser.add_argument_group("TripoSG mesh backbone options")
    triposg_group.add_argument(
        "--triposg_model_path", default=None,
        help=(
            "Local directory containing TripoSG weights. "
            "If absent, weights are auto-downloaded from VAST-AI/TripoSG on HuggingFace."
        ),
    )
    triposg_group.add_argument(
        "--triposg_steps", type=int, default=50,
        help="Number of TripoSG denoising steps (default: 50).",
    )
    triposg_group.add_argument(
        "--triposg_guidance", type=float, default=7.5,
        help="TripoSG classifier-free guidance scale (default: 7.5).",
    )
    triposg_group.add_argument(
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

    # ── validate backend combination ──────────────────────────────────────────
    if args.texture_backend == "hunyuan" and args.mesh_backbone != "hunyuan":
        sys.exit(
            "[ERROR] texture_backend='hunyuan' requires mesh_backbone='hunyuan'.\n"
            f"       Current mesh_backbone='{args.mesh_backbone}'."
        )

    # ── pretty-print run config ───────────────────────────────────────────────
    mesh_label = args.mesh_backbone
    if args.texture_backend == "none":
        texture_label = "disabled"
    elif args.texture_backend == "mvadapter":
        mode = f"text='{args.mvadapter_text}'" if args.mvadapter_text else "image-conditioned"
        texture_label = (
            f"mvadapter ({args.mvadapter_variant}, {mode}, "
            f"steps={args.mvadapter_steps}, guidance={args.mvadapter_guidance}, "
            f"seed={args.mvadapter_seed})"
        )
    else:
        texture_label = "hunyuan paint"

    print("[INFO] ─────────────────────────────────────────────────────────")
    print(f"[INFO] Input image    : {image_path}")
    print(f"[INFO] Output dir     : {output_dir}")
    print(f"[INFO] BG removal     : {'disabled' if args.no_bg_removal else 'BEN2'}")
    print(f"[INFO] Mesh backbone  : {mesh_label}")
    print(f"[INFO] Texture        : {texture_label}")
    print("[INFO] ─────────────────────────────────────────────────────────")

    # ── step 1: background removal ────────────────────────────────────────────
    if args.no_bg_removal:
        mesh_input_path = image_path
    else:
        mesh_input_path = remove_background(image_path, no_bg_path)

    # ── step 2: generate mesh with chosen backbone ────────────────────────────
    if args.mesh_backbone == "triposg":
        print("[MESH] Generating mesh with TripoSG backbone ...")
        triposg = TripoSGMeshBackbone(
            model_path=args.triposg_model_path,
            num_inference_steps=args.triposg_steps,
            guidance_scale=args.triposg_guidance,
            seed=args.triposg_seed,
        )
        mesh = triposg.generate_from_path(mesh_input_path)
        print(f"[MESH] Saving mesh → {bare_path}")
        mesh.export(bare_path)
        triposg.unload()

    else:  # hunyuan
        from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

        print(f"[MESH] Loading Hunyuan3D-2 shape pipeline ({args.hunyuan_model}) ...")
        pipeline_flow = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(args.hunyuan_model)

        # Also load paint pipeline if needed
        pipeline_paint = None
        if args.texture_backend == "hunyuan":
            from hy3dgen.texgen import Hunyuan3DPaintPipeline
            print("[TEXTURE] Loading Hunyuan3D paint pipeline ...")
            pipeline_paint = Hunyuan3DPaintPipeline.from_pretrained(
                args.hunyuan_model, subfolder="hunyuan3d-paint-v2-0"
            )

        print("[MESH] Generating base mesh with Hunyuan3D-2 ...")
        image = Image.open(mesh_input_path).convert("RGBA")
        mesh = pipeline_flow(image=image)[0]
        print(f"[MESH] Saving mesh → {bare_path}")
        mesh.export(bare_path)

        # If Hunyuan paint texturing is requested, do it now
        if args.texture_backend == "hunyuan":
            print("[TEXTURE] Painting mesh with Hunyuan3D paint pipeline ...")
            painted_mesh = pipeline_paint(mesh, image=image)
            print(f"[TEXTURE] Saving textured mesh → {painted_path}")
            painted_mesh.export(painted_path)

    # ── step 3: texturize with MVAdapter (if requested) ───────────────────────
    if args.texture_backend == "mvadapter":
        print("[TEXTURE] Texturizing with MV-Adapter ...")
        mvadapter = MVAdapterTexturizer(
            variant=args.mvadapter_variant,
            num_inference_steps=args.mvadapter_steps,
            guidance_scale_image=args.mvadapter_guidance,
            guidance_scale_text=args.mvadapter_guidance,
            seed=args.mvadapter_seed,
            checkpoints_dir=args.mvadapter_checkpoints,
            mvadapter_repo_dir=args.mvadapter_repo,
        )

        mv_save_dir  = output_dir
        mv_save_name = f"{stem}_mesh_textured_mv"

        shaded_path = mvadapter.texturize(
            mesh_path=bare_path,
            image=None if args.mvadapter_text else mesh_input_path,
            text=args.mvadapter_text,
            save_dir=mv_save_dir,
            save_name=mv_save_name,
        )

        # Move to canonical output path if different
        if os.path.abspath(shaded_path) != os.path.abspath(painted_path):
            import shutil
            shutil.move(shaded_path, painted_path)
            print(f"[TEXTURE] Textured mesh → {painted_path}")
        else:
            print(f"[TEXTURE] Textured mesh → {shaded_path}")

        mvadapter.unload()

    # ── summary ───────────────────────────────────────────────────────────────
    print("\n[INFO] Done.")
    if not args.no_bg_removal:
        print(f"  bg-removed image : {no_bg_path}")
    print(f"  mesh             : {bare_path}")
    if args.texture_backend != "none":
        print(f"  textured mesh    : {painted_path}")


if __name__ == "__main__":
    main()
