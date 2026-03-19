#!/usr/bin/env python3
"""
Generate composite images and video clips for each shot — fully parallelized using Replicate,
then upload each output to imgbb and update shots.json with permanent URLs.
"""

import os
import json
import time
import threading
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import requests
import replicate
from dotenv import load_dotenv
import base64

# ── Config ────────────────────────────────────────────────────────────────
_ENV_TEMPLATE = """\
REPLICATE_API_TOKEN=your_replicate_api_key_here
IMGBB_API_KEY=your_imgbb_api_key_here
"""

env_path = Path(".env")
if not env_path.exists():
    env_path.write_text(_ENV_TEMPLATE, encoding="utf-8")
    print("\n.env template created. Fill in API keys and re-run.")
    raise SystemExit(0)

load_dotenv()

REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN", "").strip()
IMGBB_API_KEY = os.getenv("IMGBB_API_KEY", "").strip()

if not REPLICATE_API_TOKEN or REPLICATE_API_TOKEN == "your_replicate_api_key_here":
    raise SystemExit("ERROR: REPLICATE_API_TOKEN not set in .env")

if not IMGBB_API_KEY or IMGBB_API_KEY == "your_imgbb_api_key_here":
    raise SystemExit("ERROR: IMGBB_API_KEY not set in .env")

os.environ["REPLICATE_API_TOKEN"] = REPLICATE_API_TOKEN

# ── Settings ─────────────────────────────────────────────────────────────
COMPOSITE_MAX_WORKERS = 5
VIDEO_MAX_WORKERS     = 5

VIDEO_DURATION        = 8   # seconds per shot (IMPORTANT)
VIDEO_ASPECT_RATIO    = "16:9"

# ── Thread-safe print ─────────────────────────────────────────────────────
_print_lock = threading.Lock()

def tprint(*args, **kwargs):
    with _print_lock:
        print(*args, **kwargs)

def fmt_elapsed(start: float) -> str:
    s = int(time.time() - start)
    return f"{s // 60}m{s % 60:02d}s" if s >= 60 else f"{s}s"

# ── Helpers ──────────────────────────────────────────────────────────────
def download_file(url: str, output_path: Path) -> bool:
    try:
        resp = requests.get(url, timeout=120)
        output_path.write_bytes(resp.content)
        return output_path.stat().st_size > 0
    except Exception as e:
        tprint(f"Download failed: {e}")
        return False

def upload_to_imgbb(local_path: Path) -> tuple:
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

# ── Replicate generation ─────────────────────────────────────────────────
def generate_composite(prompt: str, input_urls: list) -> tuple:
    try:
        output = replicate.run(
            "prunaai/z-image-turbo",
            input={
                "width": 1920,
                "height": 1080,
                "prompt": prompt,
                "input_urls": input_urls,
                "output_format": "png",
                "num_inference_steps": 8,
            }
        )
        return output, None
    except Exception as e:
        return None, str(e)

def generate_video(prompt: str, input_url: str) -> tuple:
    try:
        output = replicate.run(
            "prunaai/p-video",
            input={
                "fps": 24,
                "draft": True,
                "image": input_url,
                "prompt": prompt,
                "duration": VIDEO_DURATION,
                "resolution": "720p",
                "save_audio": True,
                "aspect_ratio": VIDEO_ASPECT_RATIO,
                "prompt_upsampling": False,
                "disable_safety_filter": True
            }
        )
        return output, None
    except Exception as e:
        return None, str(e)

# ── Workers ──────────────────────────────────────────────────────────────
def run_composite(shot, bg_map, char_map, composite_path):
    sid = shot["shot_id"]
    t0  = time.time()

    bg = bg_map.get(shot.get("background", ""), {})
    input_urls = [bg["image_url"]] if bg.get("image_url") else []

    for cid in shot.get("characters", []):
        c = char_map.get(cid, {})
        if c.get("image_url"):
            input_urls.append(c["image_url"])

    if not input_urls:
        return sid, None, "No image_urls"

    prompt = f"Composited animation scene: {shot.get('action')}. Pixar-style cinematic lighting."

    url, err = generate_composite(prompt, input_urls)
    if not url:
        return sid, None, err

    if not download_file(url, composite_path):
        return sid, None, "Download failed"

    imgbb_url, _ = upload_to_imgbb(composite_path)

    tprint(f"[{sid}] composite done → {composite_path}")
    return sid, imgbb_url or url, None


def run_video_worker(shot, composite_url, clip_path):
    sid = shot["shot_id"]
    t0  = time.time()

    prompt = shot.get("veo_prompt", shot.get("action", ""))

    output, err = generate_video(prompt, composite_url)
    if not output:
        return sid, False, err

    try:
        with open(clip_path, "wb") as f:
            f.write(output.read())
    except Exception as e:
        return sid, False, str(e)

    imgbb_url, _ = upload_to_imgbb(clip_path)

    tprint(f"[{sid}] video done → {clip_path}")
    return sid, True, imgbb_url or str(clip_path)

# ── Main ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shot", help="Single shot")
    args = parser.parse_args()

    start = time.time()

    shots_data = json.loads(Path("shots.json").read_text())
    chars_data = json.loads(Path("characters.json").read_text())
    bgs_data   = json.loads(Path("backgrounds.json").read_text())

    char_map = {c["character_id"]: c for c in chars_data["characters"]}
    bg_map   = {b["bg_id"]: b for b in bgs_data["backgrounds"]}

    shots = shots_data["shots"]

    Path("composites").mkdir(exist_ok=True)
    Path("clips").mkdir(exist_ok=True)

    composite_urls = {}

    # Phase 1
    with ThreadPoolExecutor(max_workers=COMPOSITE_MAX_WORKERS) as ex:
        futures = {
            ex.submit(run_composite, s, bg_map, char_map, Path("composites") / f"{s['shot_id']}.png"): s
            for s in shots
        }

        for f in as_completed(futures):
            s = futures[f]
            sid, url, err = f.result()
            if url:
                composite_urls[sid] = url
                s["composite_url"] = url

    # Phase 2
    with ThreadPoolExecutor(max_workers=VIDEO_MAX_WORKERS) as ex:
        futures = {
            ex.submit(run_video_worker, s, composite_urls[s["shot_id"]], Path("clips") / f"{s['shot_id']}.mp4"): s
            for s in shots if s["shot_id"] in composite_urls
        }

        for f in as_completed(futures):
            s = futures[f]
            sid, ok, url = f.result()
            if ok:
                s["video_url"] = url

    Path("shots.json").write_text(json.dumps(shots_data, indent=2, ensure_ascii=False))

    print(f"\nDONE in {fmt_elapsed(start)} 🚀")


if __name__ == "__main__":
    main()
