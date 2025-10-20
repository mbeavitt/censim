#!/usr/bin/env python3
"""
Convert numbered PNG heatmap images into a video.
Images are named: {number}generation.out_matrix_heatmap.png
where number ranges from start to end in specified increments.
"""

import subprocess
import os
import sys
import argparse
from pathlib import Path
import glob

def find_images(input_dir, pattern):
    """
    Find all images matching the pattern in the input directory.

    Args:
        input_dir: Directory containing the images
        pattern: Glob pattern to match images

    Returns:
        List of image paths sorted naturally
    """
    search_path = os.path.join(input_dir, pattern)
    images = glob.glob(search_path)

    # Sort images by extracting the numeric part from the filename
    def extract_number(filepath):
        basename = os.path.basename(filepath)
        # Extract number from patterns like "1000generation.out_matrix_heatmap.png"
        try:
            return int(basename.split('generation')[0])
        except (ValueError, IndexError):
            return 0

    return sorted(images, key=extract_number)

def create_video(input_dir, output_file, pattern="*generation.out_matrix_heatmap.jpg",
                 fps=60, width=2386, height=1488, bitrate='10M', preset='fast'):
    """
    Create a video from image files using ffmpeg with NVENC hardware encoding.

    Supports PNG, JPEG, and WebP image formats.

    Args:
        input_dir: Directory containing the input images
        output_file: Path to the output video file
        pattern: Glob pattern to match image files
        fps: Frames per second for the output video
        width: Output video width (must be even)
        height: Output video height (must be even)
        bitrate: Video bitrate (e.g., '10M' for 10 Mbps)
        preset: NVENC encoding preset (slow, medium, fast, hp, hq, bd, ll, llhq, llhp, lossless)
    """
    # Find all matching images
    print(f"Searching for images in {input_dir} matching pattern '{pattern}'...")
    images = find_images(input_dir, pattern)

    if not images:
        print(f"Error: No images found matching pattern '{pattern}' in {input_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(images)} images")

    # Create a temporary file listing all images in order
    filelist = os.path.join(input_dir, ".filelist.txt")

    print("Generating file list...")
    with open(filelist, 'w') as f:
        for img in images:
            # Use absolute path for safety
            abs_path = os.path.abspath(img)
            # ffmpeg concat requires duration or file directive
            f.write(f"file '{abs_path}'\n")
            f.write(f"duration {1/fps}\n")
        # Add the last file again without duration (ffmpeg requirement)
        if images:
            f.write(f"file '{os.path.abspath(images[-1])}'\n")

    print(f"Creating video at {fps} fps using NVENC hardware encoding...")
    # Using ffmpeg to create the video with NVENC hardware acceleration
    # Scale and pad all images to a consistent size with even dimensions
    cmd = [
        'ffmpeg',
        '-f', 'concat',
        '-safe', '0',
        '-i', filelist,
        '-framerate', str(fps),
        '-vf', f'scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2',
        '-c:v', 'h264_nvenc',
        '-preset', preset,
        '-b:v', bitrate,
        '-pix_fmt', 'yuv420p',
        '-y',  # Overwrite output file
        output_file
    ]

    try:
        subprocess.run(cmd, check=True)
        print(f"\n✓ Video created successfully: {output_file}")
        print(f"  Total frames: {len(images)}")
        print(f"  Frame rate: {fps} fps")
        print(f"  Duration: ~{len(images)/fps:.1f} seconds")
        print(f"  Resolution: {width}x{height}")
        print(f"  Bitrate: {bitrate}")
    except subprocess.CalledProcessError as e:
        print(f"Error running ffmpeg: {e}", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError:
        print("Error: ffmpeg not found. Please install ffmpeg.", file=sys.stderr)
        sys.exit(1)
    finally:
        # Clean up the temporary file list
        if os.path.exists(filelist):
            os.remove(filelist)

def main():
    parser = argparse.ArgumentParser(
        description='Convert numbered PNG heatmap images into a video using ffmpeg.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with default settings (60fps, 10Mbps, NVENC fast preset)
  %(prog)s -i ./output/matrix_hm -o heatmap.mp4

  # Custom FPS and bitrate
  %(prog)s -i ./images -o video.mp4 --fps 30 --bitrate 5M

  # Custom pattern and resolution
  %(prog)s -i ./images -o video.mp4 --pattern "*heatmap*.jpg" --width 1920 --height 1080

  # High quality encoding with higher bitrate
  %(prog)s -i ./images -o video.mp4 --bitrate 20M --preset hq
        """)

    parser.add_argument('-i', '--input', required=True,
                        help='Input directory containing the PNG images')
    parser.add_argument('-o', '--output', required=True,
                        help='Output video file path (e.g., heatmap.mp4)')
    parser.add_argument('-p', '--pattern', default='*generation.out_matrix_heatmap.jpg',
                        help='Glob pattern to match image files (default: *generation.out_matrix_heatmap.jpg)')
    parser.add_argument('--fps', type=int, default=60,
                        help='Frames per second for output video (default: 60)')
    parser.add_argument('--width', type=int, default=2386,
                        help='Output video width in pixels, must be even (default: 2386)')
    parser.add_argument('--height', type=int, default=1488,
                        help='Output video height in pixels, must be even (default: 1488)')
    parser.add_argument('--bitrate', default='10M',
                        help='Video bitrate (e.g., 10M for 10 Mbps, 5M for 5 Mbps) (default: 10M)')
    parser.add_argument('--preset', default='fast',
                        choices=['slow', 'medium', 'fast', 'hp', 'hq', 'bd', 'll', 'llhq', 'llhp', 'lossless'],
                        help='NVENC encoding preset (default: fast)')

    args = parser.parse_args()

    # Validate inputs
    if not os.path.isdir(args.input):
        print(f"Error: Input directory '{args.input}' does not exist", file=sys.stderr)
        sys.exit(1)

    # Ensure width and height are even (required for H.264)
    if args.width % 2 != 0 or args.height % 2 != 0:
        print("Error: Width and height must be even numbers for H.264 encoding", file=sys.stderr)
        sys.exit(1)

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    create_video(
        input_dir=args.input,
        output_file=args.output,
        pattern=args.pattern,
        fps=args.fps,
        width=args.width,
        height=args.height,
        bitrate=args.bitrate,
        preset=args.preset
    )

if __name__ == "__main__":
    main()
