import numpy as np
import cv2
from PIL import Image
import skvideo.io

def read_img(path):
    return cv2.imread(str(path))[:,:,::-1]

def write_img(img, path):
    Image.fromarray(img).save(str(path))

def save_video_as_images(video_tensor, path):
    """
    Save video frames into the directory

    Parameters
    ----------
    video_tensor: numpy.array
        video tensor in the shape of (frame, height, width, channel)
        
    path : pathlib.Path
        path to the video
    """
    path.mkdir(parents=True, exist_ok=True)
    
    placeholder = str(path / "{:03d}.jpg")
    for i, frame in enumerate(video_tensor):
        write_img(frame, placeholder.format(i))

def read_video(path):
    """
    read a video

    Parameters
    ----------
    path : string or pathlib.Path
        path to the video
        
    Returns
    -------
    video_tensor : numpy.array
        video tensor in the shape of (frame, height, width, channel)
    """
    videogen = skvideo.io.vreader(str(path))
    video_tensor = np.stack([frame for frame in videogen])

    return video_tensor

def write_video(video_tensor, path):
    """
    save a video

    Parameters
    ----------
    video_tensor: numpy.array
        video tensor in the shape of (frame, height, width, channel)
        
    path : string or pathlib.Path
        path to the video
    """
    writer = skvideo.io.FFmpegWriter(str(path))
    for frame in video_tensor:
        writer.writeFrame(frame)
    writer.close()
