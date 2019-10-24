import sys
import cv2
import numpy as np
from pathlib import Path

def write_img(img: np.ndarray, path: Path):
    cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

def save_video_as_images(video_tensor, path):
    path.mkdir(parents=True, exist_ok=True)

    placeholder = str(path / "{:03d}.png")
    for i, frame in enumerate(video_tensor):
        write_img(frame, placeholder.format(i))



root = Path("data/processed/mock/train")

# color video
red   = [255, 0, 0]
green = [0, 255, 0]
blue  = [0, 0, 255]

colors = [red, green, blue]

for i in range(3):
    video_folder = root / str(i)
    dummy_video = np.ones((16, 64, 64, 3), dtype=np.uint8)

    for j in range(len(dummy_video)):
        dummy_video[j] = dummy_video[j] * colors[(j+i)%len(colors)]

    save_video_as_images(dummy_video, Path(f"data/processed/mock/train/{i+1}/color"))

# depth video
black = 0
gray = 127
white = 255

colors = [black, gray, white]

for i in range(3):
    video_folder = root / str(i)
    dummy_video = np.ones((16, 64, 64, 3), dtype=np.uint8)

    for j in range(len(dummy_video)):
        dummy_video[j] = dummy_video[j] * colors[(j+i)%len(colors)]

    save_video_as_images(dummy_video, Path(f"data/processed/mock/train/{i+1}/depth"))
