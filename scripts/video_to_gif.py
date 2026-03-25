"""
video_to_gif.py — Convert a video clip to an optimized GIF.

Usage:
    python scripts/video_to_gif.py <input_video> [options]

Examples:
    python scripts/video_to_gif.py recording.mp4
    python scripts/video_to_gif.py recording.mp4 --start 30 --end 40 --out docs/assets/demo.gif
    python scripts/video_to_gif.py recording.mp4 --start 5 --end 15 --crop-top 55 --crop-bottom 598 --fps 15
"""

import argparse
import os
import sys
import numpy as np
from moviepy import VideoFileClip
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(description="Convert a video segment to an optimized GIF.")
    parser.add_argument("input", help="Path to the input video file")
    parser.add_argument("--start", type=float, default=0, help="Start time in seconds (default: 0)")
    parser.add_argument("--end", type=float, default=None, help="End time in seconds (default: end of video)")
    parser.add_argument("--fps", type=int, default=15, help="Output GIF frame rate (default: 15)")
    parser.add_argument("--crop-top", type=int, default=0, help="Pixels to crop from the top (default: 0)")
    parser.add_argument("--crop-bottom", type=int, default=None, help="Bottom pixel boundary to keep (default: full height)")
    parser.add_argument("--width", type=int, default=None, help="Resize output width in px, preserving aspect ratio (default: native)")
    parser.add_argument("--colors", type=int, default=256, help="GIF palette size, 2–256 (default: 256)")
    parser.add_argument("--out", default=None, help="Output GIF path (default: <input_basename>.gif)")
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.input):
        print(f"Error: file not found — {args.input}")
        sys.exit(1)

    out_path = args.out or os.path.splitext(args.input)[0] + ".gif"

    print(f"Input:  {args.input}")
    print(f"Output: {out_path}")

    clip = VideoFileClip(args.input)
    duration = clip.duration
    end = args.end if args.end is not None else duration
    end = min(end, duration)

    print(f"Video:  {clip.size[0]}x{clip.size[1]}  {duration:.1f}s  {clip.fps}fps")
    print(f"Clip:   {args.start}s to {end}s  ({end - args.start:.1f}s)")

    segment = clip.subclipped(args.start, end)

    crop_bottom = args.crop_bottom if args.crop_bottom is not None else clip.size[1]
    total_frames = int((end - args.start) * args.fps)

    print(f"Crop:   top={args.crop_top}  bottom={crop_bottom}")
    print(f"Frames: {total_frames} @ {args.fps}fps")

    frames = []
    timestamps = np.linspace(0, end - args.start - 1 / args.fps, total_frames)

    for i, t in enumerate(timestamps):
        raw = segment.get_frame(t)
        cropped = raw[args.crop_top:crop_bottom, :, :]
        img = Image.fromarray(cropped)

        if args.width:
            aspect = cropped.shape[0] / cropped.shape[1]
            new_h = int(args.width * aspect)
            img = img.resize((args.width, new_h), Image.LANCZOS)

        img_q = img.quantize(
            colors=args.colors,
            method=Image.Quantize.MEDIANCUT,
            dither=Image.Dither.FLOYDSTEINBERG,
        )
        frames.append(img_q)

        if i % 30 == 0 or i == total_frames - 1:
            print(f"  Processing: {i + 1}/{total_frames}", end="\r")

    clip.close()
    print()

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=int(1000 / args.fps),
        loop=0,
        optimize=True,
    )

    size_mb = os.path.getsize(out_path) / 1024 / 1024
    w, h = frames[0].size
    print(f"Done:   {w}x{h}  {size_mb:.1f} MB  =>  {out_path}")


if __name__ == "__main__":
    main()
