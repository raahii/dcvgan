import numpy as np
from PIL import Image

import face_recognition

def detect_face(video_tensor, num_frames_to_use=6):
    """
    detect human face in a video and return average position.

    Parameters
    ----------
    video_tensor: numpy.array
        video tensor in the shape of (frame, height, width, channel)
        
    num_frames_to_use : int
        num frames to detect face
    """

    frames = np.linspace(0, len(video_tensor), num_frames_to_use, endpoint=False)\
                .astype(np.int)

    locs = []
    for t in frames:
        locations = face_recognition.face_locations(video_tensor[t])
        if len(locations) != 0:
            locs.append(np.asarray(list(locations[0])))

    locs = np.asarray(locs)
    if len(locs) == 0:
        return [-1, -1, -1, -1]
    else:
        mean = locs.mean(axis=0).astype(np.int)
        return mean

def images_to_numpy(tensor):
    """
    convert pytorch tensor to numpy array

    Parameters
    ----------
    tensor: torch or torch.cuda
        pytorch images tensor
    
    Returns
    ---------
    imgs: numpy.array
        numpy images array
    """

    imgs = tensor.data.cpu().numpy()
    imgs = imgs.transpose(0, 2, 3, 1) # (B, C, H, W) -> (B, H, W, C)
    imgs = np.clip(imgs, -1, 1)
    imgs = (imgs + 1) / 2 * 255
    imgs = imgs.astype('uint8')

    return imgs

def videos_to_numpy(tensor):
    """
    convert pytorch tensor to numpy array

    Parameters
    ----------
    tensor: torch or torch.cuda
        pytorch videos tensor
    
    Returns
    ---------
    imgs: numpy.array
        numpy videos array
    """
    videos = tensor.data.cpu().numpy()
    import pdb; pdb.set_trace()
    videos = videos.transpose(0, 1, 2, 3, 4)
    videos = np.clip(videos, -1, 1)
    videos = (videos + 1) / 2 * 255
    videos = videos.astype('uint8')

    return videos
