#!/usr/bin/env python3
"""
Generate character and background images using the Replicate API,
then upload each image to imgbb for permanent public hosting.

Pipeline per image:
  1. replicate.run() → generate image
  2. Download PNG to ./characters/ or ./backgrounds/
  3. Upload PNG to imgbb → get permanent public URL
  4. Store imgbb URL as image_url in characters.json / backgrounds.json

Config — create a .env file in your project directory:
  REPLICATE_API_TOKEN=your_replicate_api_key_here
  IMGBB_API_KEY=your_imgbb_api_key_here

Usage:
  cd /path/to/your/project
  python generate_images.py

Requirements:
  pip install requests replicate python-dotenv
"""

import os
import json
import base64
from pathlib import Path
import requests
import replicate
from dotenv import load_dotenv

# ── Config loading (.env) ─────────────────────────────────────────────────────

_ENV_TEMPLATE = """\
# Story-to-Animation — API Keys
REPLICATE_API_TOKEN=your_replicate_api_key_here
IMGBB_API_KEY=your_imgbb_api_key_here
"""

env_path = Path(".env")
if not env_path.exists():
    env_path.write_text(_ENV_TEMPLATE, encoding="utf-8")
    print("\n  .env file created in your project directory.")
    print("  -> Open .env, fill in your API keys, then re-run this script.\n")
    raise SystemExit(0)

load_dotenv()
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN", "").strip()
IMGBB_API_KEY = os.getenv("IMGBB_API_KEY", "").strip()

if not REPLICATE_API_TOKEN or REPLICATE_API_TOKEN == "your_replicate_api_key_here":
    print("\nERROR: 'REPLICATE_API_TOKEN' not set in .env")
    raise SystemExit(1)

if not IMGBB_API_KEY or IMGBB_API_KEY == "your_imgbb_api_key_here":
    print("\nERROR: 'IMGBB_API_KEY' not set in .env")
    raise SystemExit(1)

# Set Replicate token
os.environ["REPLICATE_API_TOKEN"] = REPLICATE_API_TOKEN

# ── Step 3: Download locally ──────────────────────────────────────────────────

def download_file(url: str, output_path: Path) -> bool:
    """Download image to local path. Returns True on success."""
    try:
        resp = requests.get(url, timeout=60)
        output_path.write_bytes(resp.content)
        return output_path.stat().st_size > 0
    except Exception as e:
        print(f"    Download failed: {e}")
        return False

# ── Step 4: Upload to imgbb ───────────────────────────────────────────────────

def upload_to_imgbb(local_path: Path) -> tuple:
    """Upload local PNG to imgbb. Returns (public_url, None) or (None, error)."""
    try:
        with open(local_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        resp = requests.post(
            "https://api.imgbb.com/1/upload",
            params={"key": IMGBB_API_KEY},
            data={"image": b64},
            timeout=60,
        )
        result = resp.json()
        if result.get("success"):
            return result["data"]["url"], None
        err_msg = result.get("error", {}).get("message", "Upload failed")
        return None, err_msg
    except Exception as e:
        return None, str(e)

# ── Step 1 & 2: Replicate image generation ───────────────────────────────────

def generate_with_replicate(prompt: str, width: int, height: int) -> tuple:
    """Generate an image using replicate.run and return a URL or error."""
    try:
        output = replicate.run(
            "prunaai/z-image-turbo",
            input={
                "width": width,
                "height": height,
                "prompt": prompt,
                "go_fast": False,
                "output_format": "png",
                "guidance_scale": 0,
                "output_quality": 80,
                "num_inference_steps": 8,
            }
        )
        return output, None
    except Exception as e:
        return None, str(e)

# ── Full per-image pipeline ───────────────────────────────────────────────────

def generate_and_host(prompt: str, aspect_ratio: str, out_path: Path) -> tuple:
    """
    Full pipeline: generate → download → upload to imgbb.
    Returns:
      ("SKIP", None)     — file already exists, skip
      (imgbb_url, None)  — success, imgbb permanent URL
      (replicate_url, None) — imgbb failed, fallback to replicate URL
      (None, error)      — complete failure
    """
    if out_path.exists():
        return "SKIP", None

    if aspect_ratio == "1:1":
        width, height = 1024, 1024
    elif aspect_ratio == "16:9":
        width, height = 1920, 1080
    else:
        width, height = 1024, 1024

    rep_url, err = generate_with_replicate(prompt, width, height)
    if not rep_url:
        return None, f"Replicate generation failed: {err}"

    if not download_file(rep_url, out_path):
        return None, "Download failed"

    imgbb_url, err = upload_to_imgbb(out_path)
    if not imgbb_url:
        print(f"\n    [imgbb] Upload failed ({err}) — using replicate URL as fallback")
        return rep_url, None

    print(f"\n    [imgbb] Hosted at: {imgbb_url}")
    return imgbb_url, None

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("\n=== Story-to-Animation: Image Generation (Replicate) ===\n")

    for fname in ["characters.json", "backgrounds.json"]:
        if not Path(fname).exists():
            print(f"ERROR: {fname} not found in current directory")
            raise SystemExit(1)

    chars_data = json.loads(Path("characters.json").read_text(encoding="utf-8"))
    bgs_data   = json.loads(Path("backgrounds.json").read_text(encoding="utf-8"))

    Path("characters").mkdir(exist_ok=True)
    Path("backgrounds").mkdir(exist_ok=True)

    results    = {"success": [], "skipped": [], "failed": []}
    json_dirty = {"chars": False, "bgs": False}

    # ── Character images ───────────────────────────────────────────────
    characters = chars_data.get("characters", [])
    print(f"Generating {len(characters)} character image(s)...\n")

    for char in characters:
        cid = char["character_id"]
        out = Path("characters") / f"{cid}.png"
        print(f"  [{cid}] {char['name']}...")

        url, err = generate_and_host(char["prompt"], "1:1", out)

        if url == "SKIP":
            print(f"         → already exists, skipping")
            results["skipped"].append(f"characters/{cid}.png")
        elif url:
            char["image_url"] = url
            json_dirty["chars"] = True
            print(f"         → saved to characters/{cid}.png")
            results["success"].append(f"characters/{cid}.png")
        else:
            print(f"         → FAILED: {err}")
            results["failed"].append(f"characters/{cid}.png")

    # ── Background images ───────────────────────────────────────────────
    backgrounds = bgs_data.get("backgrounds", [])
    print(f"\nGenerating {len(backgrounds)} background image(s)...\n")

    for bg in backgrounds:
        bid = bg["bg_id"]
        out = Path("backgrounds") / f"{bid}.png"
        print(f"  [{bid}] {bg['name']}...")

        url, err = generate_and_host(bg["prompt"], "16:9", out)

        if url == "SKIP":
            print(f"         → already exists, skipping")
            results["skipped"].append(f"backgrounds/{bid}.png")
        elif url:
            bg["image_url"] = url
            json_dirty["bgs"] = True
            print(f"         → saved to backgrounds/{bid}.png")
            results["success"].append(f"backgrounds/{bid}.png")
        else:
            print(f"         → FAILED: {err}")
            results["failed"].append(f"backgrounds/{bid}.png")

    # ── Write image_url back into JSON files ──────────────────────────
    if json_dirty["chars"]:
        Path("characters.json").write_text(
            json.dumps(chars_data, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print("\n  ✓ characters.json updated with image_url (imgbb)")
    if json_dirty["bgs"]:
        Path("backgrounds.json").write_text(
            json.dumps(bgs_data, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print("  ✓ backgrounds.json updated with image_url (imgbb)")

    # ── Summary ─────────────────────────────────────────────────────
    print(f"\n{'='*50}")
    print(f"  Generated : {len(results['success'])}")
    print(f"  Skipped   : {len(results['skipped'])} (already existed)")
    print(f"  Failed    : {len(results['failed'])}")
    print(f"{'='*50}")

    if results["skipped"]:
        print("\nNote: Skipped images have no image_url in JSON.")
        print("      Delete the PNG file and re-run to generate + host a fresh URL.")

    if results["failed"]:
        print("\nFailed images (update prompt in JSON, delete PNG, re-run):")
        for f in results["failed"]:
            print(f"  - {f}")
        raise SystemExit(1)
    else:
        print("\nAll done! image_url fields are permanent imgbb URLs.")

if __name__ == "__main__":
    main()
