"""
Fetches live Earth satellite imagery + metadata from NASA's EPIC API
(DSCOVR satellite, natural-color full-disc images) and runs basic
weather/climate-oriented analysis:
 
  - Metadata pull (timestamps, sun position, satellite position, centroid coords)
  - Image download
  - Brightness / rough cloud-coverage estimate per image
  - Centroid drift plot over time
  - Brightness/cloud-coverage trend plot
 
Usage:
    python hw4.py                   # latest available day
    python hw4.py --date 2024-06-01    # specific date (YYYY-MM-DD)
    python hw4.py --download-images    # also save PNGs locally
    python hw4.py --max-images 4       # limit how many images to process
 
Requires: requests, pillow, numpy, matplotlib
API docs: https://epic.gsfc.nasa.gov/about/api
"""
 
import argparse
import io
import json
import os
import sys
from datetime import datetime
from typing import Optional
 
import numpy as np
import requests
from PIL import Image
 
try:
    import matplotlib
    matplotlib.use("Agg")  # headless-safe backend
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
 
EPIC_BASE = "https://epic.gsfc.nasa.gov"
NASA_API_KEY = os.environ.get("NASA_API_KEY", "DEMO_KEY")
 
 
def fetch_metadata(date: Optional[str] = None) -> list:
    """
    Fetch EPIC metadata. If `date` (YYYY-MM-DD) is given, fetch that day's
    images; otherwise fetch the most recent available day.
    """
    if date:
        url = f"{EPIC_BASE}/api/natural/date/{date}"
    else:
        url = f"{EPIC_BASE}/api/natural"
 
    resp = requests.get(url, params={"api_key": NASA_API_KEY}, timeout=30)
    resp.raise_for_status()
    data = resp.json()
 
    if not data:
        raise ValueError(f"No EPIC images found for date={date!r}")
    return data
 
 
def image_url_for(entry: dict, image_format: str = "png") -> str:
    """Build the full image URL for a given EPIC metadata entry."""
    dt = datetime.strptime(entry["date"], "%Y-%m-%d %H:%M:%S")
    return (
        f"{EPIC_BASE}/archive/natural/"
        f"{dt.year:04d}/{dt.month:02d}/{dt.day:02d}/{image_format}/"
        f"{entry['image']}.{image_format}"
    )
 
 
def download_image(entry: dict) -> Image.Image:
    """Download a single EPIC image as a PIL Image (RGB)."""
    url = image_url_for(entry, image_format="png")
    resp = requests.get(url, params={"api_key": NASA_API_KEY}, timeout=60)
    resp.raise_for_status()
    return Image.open(io.BytesIO(resp.content)).convert("RGB")
 
 
def analyze_image(img: Image.Image) -> dict:
    """
    Very rough analysis:
      - mean brightness (0-255) as a proxy for overall reflectivity
      - a naive cloud-coverage estimate: % of pixels above a brightness
        threshold, restricted to the visible Earth disc (non-black background)
    """
    arr = np.asarray(img).astype(np.float32)
    gray = arr.mean(axis=2)  # simple luminance proxy
 
    disc_mask = gray > 10          # exclude near-black space background
    disc_pixels = gray[disc_mask]
 
    if disc_pixels.size == 0:
        return {"mean_brightness": 0.0, "cloud_coverage_pct": 0.0}
 
    mean_brightness = float(disc_pixels.mean())
    cloud_threshold = 180  # bright pixels ~ cloud tops / ice, tune as needed
    cloud_coverage_pct = float((disc_pixels > cloud_threshold).mean() * 100)
 
    return {
        "mean_brightness": round(mean_brightness, 2),
        "cloud_coverage_pct": round(cloud_coverage_pct, 2),
    }
 
 
def run(date: Optional[str], download_images: bool, max_images: int, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    if download_images:
        os.makedirs(os.path.join(out_dir, "images"), exist_ok=True)
 
    print(f"Fetching EPIC metadata (date={date or 'latest'})...")
    metadata = fetch_metadata(date)
    metadata = metadata[:max_images]
    print(f"Got {len(metadata)} image record(s). Downloading + analyzing...")
 
    results = []
    for entry in metadata:
        img = download_image(entry)
        analysis = analyze_image(img)
 
        centroid = entry.get("centroid_coordinates", {})
        record = {
            "identifier": entry.get("identifier"),
            "timestamp": entry.get("date"),
            "caption": entry.get("caption"),
            "centroid_lat": centroid.get("lat"),
            "centroid_lon": centroid.get("lon"),
            **analysis,
        }
        results.append(record)
        print(
            f"  {record['timestamp']} | brightness={record['mean_brightness']} "
            f"| cloud_coverage~{record['cloud_coverage_pct']}% "
            f"| centroid=({record['centroid_lat']}, {record['centroid_lon']})"
        )
 
        if download_images:
            fname = os.path.join(out_dir, "images", f"{entry['image']}.png")
            img.save(fname)
 
    # Save raw results as JSON
    json_path = os.path.join(out_dir, "epic_analysis.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved analysis data -> {json_path}")
 
    # Plot trends if matplotlib is available and we have >1 data point
    if plt is not None and len(results) > 1:
        timestamps = [datetime.strptime(r["timestamp"], "%Y-%m-%d %H:%M:%S") for r in results]
        brightness = [r["mean_brightness"] for r in results]
        cloud_cov = [r["cloud_coverage_pct"] for r in results]
 
        fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
 
        axes[0].plot(timestamps, brightness, marker="o", color="#2b6cb0")
        axes[0].set_ylabel("Mean brightness (0-255)")
        axes[0].set_title("Earth Disc Brightness Over Time (EPIC / DSCOVR)")
        axes[0].grid(alpha=0.3)
 
        axes[1].plot(timestamps, cloud_cov, marker="o", color="#38a169")
        axes[1].set_ylabel("Est. cloud coverage (%)")
        axes[1].set_xlabel("Timestamp (UTC)")
        axes[1].grid(alpha=0.3)
 
        fig.autofmt_xdate()
        plot_path = os.path.join(out_dir, "epic_trends.png")
        fig.tight_layout()
        fig.savefig(plot_path, dpi=150)
        print(f"Saved trend plot -> {plot_path}")
    elif len(results) <= 1:
        print("Only one data point — skipping trend plot (need 2+ for a trend).")
 
 
def main():
    parser = argparse.ArgumentParser(description="Analyze live NASA EPIC satellite imagery.")
    parser.add_argument("--date", type=str, default=None, help="YYYY-MM-DD (default: latest available)")
    parser.add_argument("--download-images", action="store_true", help="Save PNGs locally")
    parser.add_argument("--max-images", type=int, default=6, help="Max number of images to process")
    parser.add_argument("--out-dir", type=str, default="epic_output", help="Output directory")
    args = parser.parse_args()
 
    try:
        run(args.date, args.download_images, args.max_images, args.out_dir)
    except requests.HTTPError as e:
        print(f"API request failed: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
 
 
if __name__ == "__main__":
    main()