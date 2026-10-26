"""
=============================================================================
MathVision: Extract Target 18 Classes from Kaggle Archive
=============================================================================
Extracts only the 18 required handwritten math symbol classes from the 
downloaded Kaggle dataset archive directly into data/raw/.
=============================================================================
"""

import os
import shutil
import subprocess
from typing import Dict, List

ARCHIVE_PATH = r"C:\Users\Jainam\.cache\kagglehub\datasets\xainano\handwrittenmathsymbols\versions\2\data.rar"
TARGET_DIR = r"data\raw"

# Mapping from archive folder inside 'extracted_images/' to target folder in data/raw/
CLASS_MAPPING: Dict[str, str] = {
    "0": "0",
    "1": "1",
    "2": "2",
    "3": "3",
    "4": "4",
    "5": "5",
    "6": "6",
    "7": "7",
    "8": "8",
    "9": "9",
    "+": "+",
    "-": "-",
    "times": "times",
    "div": "div",
    "=": "=",
    "(": "(",
    ")": ")",
    "X": "x",  # Map capital X to canonical 'x'
}

def extract_classes():
    os.makedirs(TARGET_DIR, exist_ok=True)
    temp_extract = os.path.join(TARGET_DIR, "_extracted_temp")
    os.makedirs(temp_extract, exist_ok=True)
    
    print("[1/3] Extracting 18 target class folders from archive...")
    tar_args = [f"extracted_images/{src}" for src in CLASS_MAPPING.keys()]
    cmd = ["tar", "-xf", ARCHIVE_PATH, "-C", temp_extract] + tar_args
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"Warning: tar exited with code {res.returncode}. Stderr: {res.stderr[:300]}")
    
    print("[2/3] Organizing folders into data/raw/...")
    src_base = os.path.join(temp_extract, "extracted_images")
    counts = {}
    
    for src_folder, dst_folder in CLASS_MAPPING.items():
        src_path = os.path.join(src_base, src_folder)
        dst_path = os.path.join(TARGET_DIR, dst_folder)
        
        if os.path.exists(src_path):
            if os.path.exists(dst_path) and os.path.abspath(src_path) != os.path.abspath(dst_path):
                shutil.rmtree(dst_path)
            shutil.move(src_path, dst_path)
            num_files = len([f for f in os.listdir(dst_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            counts[dst_folder] = num_files
        elif os.path.exists(dst_path):
            num_files = len([f for f in os.listdir(dst_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            counts[dst_folder] = num_files
        else:
            print(f"Warning: Folder {src_folder} not found in archive.")
            counts[dst_folder] = 0
            
    # Cleanup temp directory
    if os.path.exists(temp_extract):
        shutil.rmtree(temp_extract, ignore_errors=True)
        
    print("[3/3] Extraction Summary:")
    print(f"{'Class':<10} | {'Folder':<10} | {'Image Count':<12}")
    print("-" * 38)
    total_imgs = 0
    for cls_name, cnt in sorted(counts.items()):
        print(f"{cls_name:<10} | {cls_name:<10} | {cnt:<12}")
        total_imgs += cnt
    print("-" * 38)
    print(f"Total Images Extracted: {total_imgs} across {len(counts)} classes.")

if __name__ == "__main__":
    extract_classes()
