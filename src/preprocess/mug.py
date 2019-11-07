import os
import re
import shutil
from pathlib import Path

import numpy as np
import skvideo.io
from joblib import Parallel, delayed

import dataio
import util


def preprocess_mug_dataset(
    dataset_path: Path,
    save_path: Path,
    mode: str,
    length: int,
    img_size: int,
    n_jobs: int = -1,
):
    """
    Preprocessing function for MUG Facial Expression Database
    https://mug.ee.auth.gr/fed/
    """
    raise NotImplementedError
