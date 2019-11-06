import os
import subprocess as sp
import time
from pathlib import Path

from halo import Halo

LOCAL_DIR = os.environ.get("LOCAL_DIR")  # <ROOT>/data/processed/isogd/train/
REMOTE_HOST = "labo"
REMOTE_DIR = os.environ.get("REMOTE_DIR")  # <HOST>:<ROOT>/data/processed/isogd/train/
NUM_SAMPLES = 100


def rsync(_from, _to, dry_run=False, verbose=True):
    cmds = ["rsync", "-auvz", str(_from), str(_to)]

    if dry_run:
        cmds.insert(2, "--dry-run")

    text = ""
    if verbose:
        text = f"> {' '.join(cmds)}\n"
    print(text)

    with Halo(text="executing", spinner="dots"):
        res = sp.call(cmds)
        print(f"status: {res}")


def main():
    # fetch train list
    rsync(REMOTE_DIR / "list.txt", str(LOCAL_DIR) + "/")

    with open(LOCAL_DIR / "list.txt") as f:
        lines = f.readlines().copy()
    os.remove(LOCAL_DIR / "list.txt")

    f = open(LOCAL_DIR / "list.txt", "w")
    (LOCAL_DIR / "color").mkdir(exist_ok=True)
    cnt = 0
    for line in lines:
        filename = line.split(" ")[0].strip()
        if filename == "":
            continue

        # dataset sample
        rsync(REMOTE_DIR / filename, str(LOCAL_DIR) + "/", verbose=True)

        # color video
        rsync(
            REMOTE_DIR / "color" / (filename + ".mp4"),
            str(LOCAL_DIR / "color") + "/",
            verbose=True,
        )

        f.write(line)
        cnt += 1
        if cnt > NUM_SAMPLES:
            break
    f.close()


if __name__ == "__main__":
    main()
