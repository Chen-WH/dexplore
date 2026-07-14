import os
import argparse
from moviepy.editor import ImageSequenceClip

parser = argparse.ArgumentParser()
parser.add_argument("--image_path", type=str, required=True)
parser.add_argument("--fps", type=int, default=30)
parser.add_argument("--output", type=str, default=None, help="Output video path")
args = parser.parse_args()

image_folder = args.image_path
image_files = sorted([
    os.path.join(image_folder, img)
    for img in os.listdir(image_folder)
    if img.endswith(".png")
])

clip = ImageSequenceClip(image_files, fps=args.fps)

if args.output:
    out_path = args.output
else:
    out_path = os.path.splitext(image_folder.rstrip('/')) [0] + '.mp4'

os.makedirs(os.path.dirname(out_path), exist_ok=True)
clip.write_videofile(out_path)
