import os
import re
import shutil
from pathlib import Path

import numpy as np
import skvideo.io
from joblib import Parallel, delayed
from scipy.misc import imresize

import dataio
import utils


def preprocess_mug_dataset(dataset_path, save_path, mode, length, img_size, n_jobs=-1):
    """
    Preprocessing function for MUG Facial Expression Database
    https://mug.ee.auth.gr/fed/
    """
    raise NotImplementedError
