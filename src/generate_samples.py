"""
=============================================================================
MathVision: Synthetic Sample Equation Generator
=============================================================================

Generates clean, realistic handwritten-style mathematical equation images
for data/samples/ along with a ground-truth labels.csv.

This enables end-to-end testing of the CV preprocessing, segmentation,
parsing, and solving pipeline before user-supplied photos or webcam captures
are introduced.
=============================================================================
"""

import os
import csv
import random
from typing import List, Tuple
import numpy as np
from PIL import Image, ImageDraw, ImageFont


# List of candidate TrueType fonts available on Windows
FONT_CANDIDATES = [
    r"C:\Windows\Fonts\Inkfree.ttf",     # Windows Ink Free handwriting font
    r"C:\Windows\Fonts\segoepr.ttf",     # Windows Segoe Print font
    r"C:\Windows\Fonts\segoeprb.ttf",    # Windows Segoe Print Bold font
    r"C:\Windows\Fonts\comic.ttf",       # Comic Sans
    r"C:\Windows\Fonts\arial.ttf",       # Arial (fallback clean)
]

EQUATIONS: List[Tuple[str, str, str]] = [
    # (filename, visual_equation_text, true_sympy_expression)
    ("sample_01.png", "2 + 3 * 4 = 14", "2+3*4=14"),
    ("sample_02.png", "3x + 5 = 20", "3*x+5=20"),
    ("sample_03.png", "(8 - 2) / 2 = 3", "(8-2)/2=3"),
    ("sample_04.png", "x^2 - 5x + 6 = 0", "x**2-5*x+6=0"),
    ("sample_05.png", "7 - 2x = 1", "7-2*x=1"),
    ("sample_06.png", "4x + 8 = 24", "4*x+8=24"),
    ("sample_07.png", "15 / (2 + 1) = 5", "15/(2+1)=5"),
    ("sample_08.png", "x + 9 = 12", "x+9=12"),
    ("sample_09.png", "6 * 7 - 10 = 32", "6*7-10=32"),
    ("sample_10.png", "2x - 7 = 3", "2*x-7=3"),
]


def get_available_font(font_size: int = 54) -> ImageFont.ImageFont:
    """Finds the best available handwriting or clean font on the system."""
    for font_path in FONT_CANDIDATES:
        if os.path.exists(font_path):
            try:
                return ImageFont.truetype(font_path, font_size)
            except Exception:
                continue
    return ImageFont.load_default()


def render_equation_image(
    eq_text: str,
    output_path: str,
    font_size: int = 54,
    add_paper_texture: bool = True
) -> None:
    """
    Renders an equation onto a synthetic paper canvas with subtle ink jitter.
    """
    font = get_available_font(font_size)

    # Estimate text dimensions
    dummy_img = Image.new("RGB", (100, 100))
    dummy_draw = ImageDraw.Draw(dummy_img)
    bbox = dummy_draw.textbbox((0, 0), eq_text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    # Create image with comfortable margins
    pad_x = 70
    pad_y = 50
    img_w = max(500, text_w + 2 * pad_x)
    img_h = max(160, text_h + 2 * pad_y)

    # Background color: subtle warm off-white (paper-like)
    bg_color = (250, 249, 246)
    img = Image.new("RGB", (img_w, img_h), color=bg_color)
    draw = ImageDraw.Draw(img)

    # Subtle background noise to simulate paper fiber
    if add_paper_texture:
        arr = np.array(img, dtype=np.int16)
        noise = np.random.normal(0, 3, arr.shape).astype(np.int16)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)
        draw = ImageDraw.Draw(img)

    # Ink color: dark charcoal/blue-black ink
    ink_color = (25, 28, 35)

    # Render character-by-character or as text
    # Handling exponent '^2' visually (superscript)
    if "^" in eq_text:
        # Render superscript for exponent
        parts = eq_text.split("^")
        base = parts[0]
        exp_and_rest = parts[1]
        exp_char = exp_and_rest[0]
        rest = exp_and_rest[1:]

        x_cur = pad_x
        y_cur = (img_h - text_h) // 2

        # Draw base
        draw.text((x_cur, y_cur), base, fill=ink_color, font=font)
        base_bbox = draw.textbbox((x_cur, y_cur), base, font=font)
        x_cur = base_bbox[2] + 4

        # Draw superscript exponent (smaller font, shifted higher)
        exp_font = get_available_font(int(font_size * 0.65))
        draw.text((x_cur, y_cur - 12), exp_char, fill=ink_color, font=exp_font)
        exp_bbox = draw.textbbox((x_cur, y_cur - 12), exp_char, font=exp_font)
        x_cur = exp_bbox[2] + 6

        # Draw rest
        draw.text((x_cur, y_cur), rest, fill=ink_color, font=font)
    else:
        # Center text
        x_cur = (img_w - text_w) // 2
        y_cur = (img_h - text_h) // 2
        draw.text((x_cur, y_cur), eq_text, fill=ink_color, font=font)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    img.save(output_path, "PNG")


def generate_all_samples(samples_dir: str = "data/samples") -> str:
    """
    Generates all sample equation images and outputs labels.csv.

    Returns:
        str: Path to the generated labels.csv.
    """
    os.makedirs(samples_dir, exist_ok=True)
    labels_csv_path = os.path.join(samples_dir, "labels.csv")

    with open(labels_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "visual_text", "canonical_expression"])

        for filename, visual_text, canonical_expr in EQUATIONS:
            out_img_path = os.path.join(samples_dir, filename)
            render_equation_image(visual_text, out_img_path)
            writer.writerow([filename, visual_text, canonical_expr])

    print(f"[MathVision] Successfully generated {len(EQUATIONS)} samples in {samples_dir}")
    print(f"[MathVision] Ground truth metadata written to: {labels_csv_path}")
    return labels_csv_path


if __name__ == "__main__":
    generate_all_samples()
