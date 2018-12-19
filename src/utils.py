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
