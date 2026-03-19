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

VIDEO_ASPECT_RATIO     = "landscape"  # "landscape", "portrait", "square"
VIDEO_N_FRAMES         = 10
VIDEO_REMOVE_WATERMARK = True

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
        tprint(f"    Download failed: {e}")
        return False

def upload_to_imgbb(local_path: Path) -> tuple:
    """Upload local PNG/MP4 to imgbb."""
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
def generate_composite(prompt: str, input_urls: list, width=1920, height=1080) -> tuple:
    try:
        output = replicate.run(
            "prunaai/z-image-turbo",
            input={
                "width": width,
                "height": height,
                "prompt": prompt,
                "input_urls": input_urls,
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

def generate_video(prompt: str, input_url: str, n_frames=VIDEO_N_FRAMES, aspect_ratio=VIDEO_ASPECT_RATIO) -> tuple:
    try:
        output = replicate.run(
            "prunaai/z-video-generator",
            input={
                "prompt": prompt,
                "input_url": input_url,
                "n_frames": n_frames,
                "aspect_ratio": aspect_ratio,
                "remove_watermark": VIDEO_REMOVE_WATERMARK,
            }
        )
        return output, None
    except Exception as e:
        return None, str(e)

# ── Phase workers ───────────────────────────────────────────────────────
def run_composite(shot: dict, bg_map: dict, char_map: dict, composite_path: Path) -> tuple:
    sid = shot["shot_id"]
    t0  = time.time()
    # Build input_urls
    bg = bg_map.get(shot.get("background", ""), {})
    input_urls = [bg["image_url"]] if bg.get("image_url") else []
    for cid in shot.get("characters", []):
        c = char_map.get(cid, {})
        if c.get("image_url"):
            input_urls.append(c["image_url"])
    if not input_urls:
        return sid, None, "No image_urls found"
    prompt = f"Composited animation scene: {shot.get('action','characters in location')}. Pixar-style 3D cinematic, high quality."
    rep_url, err = generate_composite(prompt, input_urls)
    if not rep_url:
        return sid, None, err
    if not download_file(rep_url, composite_path):
        return sid, None, "Download failed"
    # Upload to imgbb
    imgbb_url, err = upload_to_imgbb(composite_path)
    if not imgbb_url:
        tprint(f"  [{sid}] imgbb upload failed ({err}) — using local path only")
        return sid, composite_path, None
    tprint(f"  [{sid}] composite done ({fmt_elapsed(t0)}) → {composite_path} → imgbb URL: {imgbb_url}")
    return sid, imgbb_url, None

def run_video_worker(shot: dict, composite_url: str, clip_path: Path) -> tuple:
    sid = shot["shot_id"]
    t0  = time.time()
    prompt = shot.get("veo_prompt", shot.get("action", ""))
    rep_url, err = generate_video(prompt, str(composite_url))
    if not rep_url:
        return sid, False, err
    if not download_file(rep_url, clip_path):
        return sid, False, "Download failed"
    imgbb_url, err = upload_to_imgbb(clip_path)
    if not imgbb_url:
        tprint(f"  [{sid}] imgbb upload failed ({err}) — using local path only")
        return sid, True, None
    tprint(f"  [{sid}] video done ({fmt_elapsed(t0)}) → {clip_path} → imgbb URL: {imgbb_url}")
    return sid, True, imgbb_url

# ── Main ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shot", help="Process a single shot only")
    args = parser.parse_args()
    run_start = time.time()

    # Load JSON
    for fname in ["shots.json", "characters.json", "backgrounds.json"]:
        if not Path(fname).exists():
            raise SystemExit(f"{fname} not found.")
    shots_data = json.loads(Path("shots.json").read_text())
    chars_data = json.loads(Path("characters.json").read_text())
    bgs_data   = json.loads(Path("backgrounds.json").read_text())
    char_map = {c["character_id"]: c for c in chars_data["characters"]}
    bg_map   = {b["bg_id"]: b for b in bgs_data["backgrounds"]}

    # Verify image_url
    missing = [f for f in chars_data["characters"]+bgs_data["backgrounds"] if not f.get("image_url")]
    if missing:
        raise SystemExit("image_url missing. Run generate_images.py first.")

    all_shots = shots_data.get("shots", [])
    if args.shot:
        all_shots = [s for s in all_shots if s["shot_id"] == args.shot]

    Path("composites").mkdir(exist_ok=True)
    Path("clips").mkdir(exist_ok=True)

    # Phase 1: composites
    composite_urls = {}
    with ThreadPoolExecutor(max_workers=COMPOSITE_MAX_WORKERS) as ex:
        futures = {ex.submit(run_composite, s, bg_map, char_map, Path("composites")/f"{s['shot_id']}.png"): s["shot_id"] for s in all_shots}
        for f in as_completed(futures):
            sid, url, err = f.result()
            if url:
                composite_urls[sid] = url
                # Update shots.json
                shot_obj = next(s for s in shots_data["shots"] if s["shot_id"] == sid)
                shot_obj["composite_url"] = url
            else:
                tprint(f"  [{sid}] composite FAILED: {err}")

    # Phase 2: videos
    with ThreadPoolExecutor(max_workers=VIDEO_MAX_WORKERS) as ex:
        futures = {ex.submit(run_video_worker, s, composite_urls[s["shot_id"]], Path("clips")/f"{s['shot_id']}.mp4"): s["shot_id"] for s in all_shots if s["shot_id"] in composite_urls}
        for f in as_completed(futures):
            sid, ok, url = f.result()
            shot_obj = next(s for s in shots_data["shots"] if s["shot_id"] == sid)
            if ok and url:
                shot_obj["video_url"] = url

    # Write updated shots.json
    Path("shots.json").write_text(json.dumps(shots_data, indent=2, ensure_ascii=False))
    tprint(f"\nAll done in {fmt_elapsed(run_start)}! shots.json updated with imgbb URLs.")

if __name__ == "__main__":
    main()