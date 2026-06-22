"""
white_background.py — Composite a transparent-background image onto white.

Usage:
    python white_background.py <image_path> [output_path]

    <image_path>   Path to the input image (RGBA PNG or any PIL-supported format).
    [output_path]  Where to save the result.
                   Defaults to <stem>_white_bg.<ext> next to the input file.

Examples:
    python white_background.py mesh/generated_meshes/B_combined1/B_combined1_no_bg.png
    python white_background.py input.png output_white.png
"""

import sys
import os
from PIL import Image


def add_white_background(image_path: str, output_path: str = None) -> str:
    """
    Composite *image_path* onto a solid white background and save the result.

    Args:
        image_path:  Path to the input image.
        output_path: Destination path.  If None, a name is auto-generated next
                     to the input file (<stem>_white_bg<ext>).

    Returns:
        The path where the result was saved.
    """
    image_path = os.path.abspath(image_path)
    img = Image.open(image_path).convert("RGBA")

    white = Image.new("RGBA", img.size, (255, 255, 255, 255))
    white.paste(img, mask=img.split()[3])
    result = white.convert("RGB")

    if output_path is None:
        stem, ext = os.path.splitext(image_path)
        output_path = f"{stem}_white_bg{ext}"

    result.save(output_path)
    print(f"Saved → {output_path}")
    return output_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    src = sys.argv[1]
    dst = sys.argv[2] if len(sys.argv) > 2 else None
    add_white_background(src, dst)
