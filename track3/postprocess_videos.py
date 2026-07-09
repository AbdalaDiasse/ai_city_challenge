#!/usr/bin/env python3
"""
Post-process partially downloaded video datasets for Track 3.

Run this after download_videos.py when the following post-processing steps
failed because 7z or ffmpeg were missing:

  so_tad   — 51-part PKWARE zip -> so-tad/{train,test}/<id>.mp4  (needs 7z)
  tad      — JPG frame folders  -> TAD/<name>.mp4                 (needs ffmpeg)
  htv      — .avi files         -> HTV/<name>.mp4                 (needs ffmpeg)
  barbados — completely missing  -> downloads via download_videos.py

Prerequisites:
    pip install imageio-ffmpeg          # provides ffmpeg without system install
    conda install -c conda-forge p7zip  # provides 7z for so-tad extraction

Usage:
    python track3/postprocess_videos.py --work-dir $WORK
    python track3/postprocess_videos.py --work-dir $WORK --only so_tad htv
    python track3/postprocess_videos.py --work-dir $WORK --check
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


# ── helpers ─────────────────────────────────────────────────────────────────

def find_ffmpeg():
    ff = shutil.which("ffmpeg")
    if ff:
        return ff
    try:
        from imageio_ffmpeg import get_ffmpeg_exe
        return get_ffmpeg_exe()
    except ImportError:
        return None


def find_7z(explicit: str | None = None):
    if explicit:
        return explicit
    return shutil.which("7z") or shutil.which("7za") or shutil.which("7zz")


def run(cmd, **kwargs):
    print("  $", " ".join(str(c) for c in cmd))
    subprocess.run([str(c) for c in cmd], check=True, **kwargs)


# ── per-dataset post-processors ─────────────────────────────────────────────


def process_sotad(videos_dir: Path, _annotations_dir: Path, sevenz_bin: str | None = None):
    """Extract the 51-part PKWARE split-zip -> so-tad/{train,test}/<id>.mp4"""
    sotad_dir = videos_dir / "so-tad"
    main_zip = sotad_dir / "so_tad.zip"

    # Already extracted?
    train_dir = sotad_dir / "train"
    if train_dir.is_dir() and any(train_dir.iterdir()):
        n = sum(1 for _ in sotad_dir.rglob("*.mp4"))
        print(f"[so-tad] already extracted ({n} mp4s) — skip")
        return

    if not main_zip.exists():
        print(f"[so-tad] so_tad.zip not found at {main_zip} — nothing to extract")
        return

    parts = sorted(sotad_dir.glob("so_tad.z[0-9]*"))

    sevenz = find_7z(sevenz_bin)
    if not sevenz:
        print(
            "[so-tad] ERROR: 7z binary not found. Options:\n"
            "  A) conda install -c conda-forge p7zip -y\n"
            "  B) wget https://www.7-zip.org/a/7z2409-linux-x64.tar.xz -O /tmp/7z.tar.xz\n"
            "     tar xf /tmp/7z.tar.xz -C /tmp 7zz\n"
            "     python track3/postprocess_videos.py --work-dir $WORK --only so_tad --sevenz /tmp/7zz"
        )
        return
    print(f"[so-tad] extracting via {sevenz} ...")
    run([sevenz, "x", "-bb1", "-y", str(main_zip), f"-o{sotad_dir}"])

    # Remove archive parts to reclaim ~27 GB (7z may already have deleted them)
    to_remove = list(sotad_dir.glob("so_tad.z*")) + [main_zip]
    for p in to_remove:
        p.unlink(missing_ok=True)
    n = sum(1 for _ in sotad_dir.rglob("*.mp4"))
    print(f"[so-tad] done — {n} mp4s, removed {len(to_remove)} archive files")


def process_htv(videos_dir: Path, _annotations_dir: Path):
    """Transcode HTV .avi -> flat HTV/<name>.mp4"""
    htv_dir = videos_dir / "HTV"
    avi_dir = htv_dir / "video"

    if not avi_dir.is_dir():
        print(f"[HTV] {avi_dir} not found — skip")
        return

    avis = sorted(avi_dir.glob("*.avi"))
    if not avis:
        print("[HTV] no .avi files — skip")
        return

    existing = list(htv_dir.glob("*.mp4"))
    if len(existing) >= len(avis):
        print(f"[HTV] {len(existing)} mp4s already present — skip")
        return

    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        print(
            "[HTV] ERROR: ffmpeg not found.\n"
            "  Install: pip install imageio-ffmpeg\n"
            "  Then re-run this script."
        )
        return

    todo = [a for a in avis if not (htv_dir / (a.stem + ".mp4")).exists()]
    print(f"[HTV] transcoding {len(todo)}/{len(avis)} .avi -> .mp4 ...")
    ok = failed = 0
    for src in todo:
        dst = htv_dir / (src.stem + ".mp4")
        try:
            subprocess.run(
                [ffmpeg, "-y", "-loglevel", "error",
                 "-i", str(src),
                 "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
                 str(dst)],
                check=True,
            )
            ok += 1
        except subprocess.CalledProcessError:
            print(f"  [fail] {src.name}")
            failed += 1

    print(f"[HTV] done: ok={ok} failed={failed}")
    if failed == 0 and avi_dir.is_dir() and not any(avi_dir.iterdir()):
        avi_dir.rmdir()


def process_tad(videos_dir: Path, _annotations_dir: Path):
    """Stitch TAD JPG frame folders -> flat TAD/<name>.mp4"""
    tad_dir = videos_dir / "TAD"
    frames_root = tad_dir / "TAD" / "frames"

    existing = list(tad_dir.glob("*.mp4"))
    if existing:
        print(f"[TAD] {len(existing)} mp4s already present — skip")
        return

    if not frames_root.is_dir():
        print(f"[TAD] frames dir not found at {frames_root} — skip")
        return

    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        print(
            "[TAD] ERROR: ffmpeg not found.\n"
            "  Install: pip install imageio-ffmpeg\n"
            "  Then re-run this script."
        )
        return

    # Use the stitcher from the companion repo
    stitch_script = (
        Path(__file__).parent.parent
        / "PhysicalAI-Traffic-Anomaly-Reasoning"
        / "stitch_tad_frames.py"
    )
    if not stitch_script.exists():
        print(f"[TAD] stitch_tad_frames.py not found at {stitch_script}")
        return

    print(f"[TAD] stitching frames from {frames_root.parent} -> {tad_dir} ...")
    run([
        sys.executable, str(stitch_script),
        "--in", str(frames_root.parent),  # TAD/TAD/ — contains frames/
        "--out", str(tad_dir),
        "--fps", "30",
        "--ffmpeg", ffmpeg,
    ])
    n = len(list(tad_dir.glob("*.mp4")))
    print(f"[TAD] done — {n} mp4s")


def process_barbados(videos_dir: Path, annotations_dir: Path):
    """Download barbados dataset (missing entirely)."""
    barb_dir = videos_dir / "barbados_traffic_challenge"
    if barb_dir.is_dir() and any(barb_dir.rglob("*.mp4")):
        n = sum(1 for _ in barb_dir.rglob("*.mp4"))
        print(f"[barbados] already present ({n} mp4s) — skip")
        return

    dl_script = annotations_dir / "download_videos.py"
    if not dl_script.exists():
        print(f"[barbados] download_videos.py not found at {dl_script}")
        return

    # Use the venv python if available, fall back to sys.executable
    venv_python = Path(sys.executable).parent / "python"
    python = str(venv_python) if venv_python.exists() else sys.executable

    print("[barbados] installing download deps ...")
    subprocess.run([python, str(dl_script), "--install-deps"], check=False)

    print("[barbados] downloading via download_videos.py --only barbados ...")
    run([python, str(dl_script), "--out", str(videos_dir), "--only", "barbados"])


# ── check mode ──────────────────────────────────────────────────────────────

EXPECTED = {
    "Accident-Bench":             {"pattern": "**/*.mp4", "min": 2000},
    "TAD-benchmark":              {"pattern": "**/*.mp4", "min": 400},
    "UCF_Crimes":                 {"pattern": "Videos/**/*.mp4", "min": 1600},
    "Vad-R1":                     {"pattern": "**/*.mp4", "min": 7000},
    "so-tad":                     {"pattern": "{train,test}/*.mp4", "min": 1},
    "TAD":                        {"pattern": "*.mp4", "min": 1},
    "HTV":                        {"pattern": "*.mp4", "min": 250},
    "barbados_traffic_challenge": {"pattern": "normanniles1/*.mp4", "min": 1},
}

TOTAL_ANNOTATIONS = {
    "so-tad": 26232, "TAD-benchmark": 4392, "UCF_Crimes": 4008,
    "HTV": 3048, "TAD": 2292, "Vad-R1": 1572,
    "barbados_traffic_challenge": 1536, "Accident-Bench": 960,
}


def check(videos_dir: Path):
    print(f"\nVideo directory: {videos_dir}\n")
    print(f"{'Dataset':<30} {'mp4s':>6}  {'ann':>6}  Status")
    print("-" * 60)
    total_ann = 0
    ready_ann = 0
    for name, spec in EXPECTED.items():
        d = videos_dir / name
        if not d.is_dir():
            count = 0
        else:
            # glob doesn't support {a,b} — count both patterns for so-tad
            if name == "so-tad":
                count = (
                    sum(1 for _ in (d / "train").glob("*.mp4") if (d / "train").is_dir())
                    + sum(1 for _ in (d / "test").glob("*.mp4") if (d / "test").is_dir())
                )
            else:
                count = sum(1 for _ in d.glob(spec["pattern"]))

        ann = TOTAL_ANNOTATIONS.get(name, 0)
        total_ann += ann
        ok = count >= spec["min"]
        status = "✓ ready" if ok else "✗ needs post-processing"
        if ok:
            ready_ann += ann
        print(f"  {name:<28} {count:>6}  {ann:>6}  {status}")

    pct = 100 * ready_ann / total_ann if total_ann else 0
    print(f"\nAnnotations ready: {ready_ann}/{total_ann} ({pct:.0f}%)")


# ── main ────────────────────────────────────────────────────────────────────

PROCESSORS = {
    "so_tad":  process_sotad,
    "htv":     process_htv,
    "tad":     process_tad,
    "barbados": process_barbados,
}


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--work-dir", required=True,
                    help="$WORK root (e.g. /leonardo_work/AIH4A_syrate)")
    ap.add_argument("--only", nargs="+", choices=list(PROCESSORS),
                    help="Only process these datasets")
    ap.add_argument("--sevenz", default=None,
                    help="Path to 7z binary (e.g. /tmp/7zz). Auto-detected if omitted.")
    ap.add_argument("--check", action="store_true",
                    help="Print status table and exit without processing")
    args = ap.parse_args()

    work_dir = Path(args.work_dir)
    videos_dir = work_dir / "data/track3/videos"
    annotations_dir = work_dir / "data/track3/annotations"

    if args.check:
        check(videos_dir)
        return

    keys = args.only or list(PROCESSORS)
    for key in keys:
        print(f"\n=== {key} ===")
        if key == "so_tad":
            PROCESSORS[key](videos_dir, annotations_dir, args.sevenz)
        else:
            PROCESSORS[key](videos_dir, annotations_dir)

    print("\n--- final status ---")
    check(videos_dir)


if __name__ == "__main__":
    main()
