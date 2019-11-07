import argparse
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from joblib import Parallel, delayed
from tqdm import tqdm

import dataio
import util


def load_model(model_path: Path, params_path: Path) -> nn.Module:
    model = torch.load(model_path, map_location="cpu")
    params = torch.load(params_path, map_location="cpu")
    model.load_state_dict(params)
    model = model.to(util.current_device())
    model.device = util.current_device()
    model.eval()

    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("iteration", type=int)
    parser.add_argument("save_dir", type=Path)
    parser.add_argument("--n_samples", "-n", type=int, default=10000)
    parser.add_argument("--batchsize", "-b", type=int, default=10)
    args = parser.parse_args()

    # read config file
    with open(args.result_dir / "config.yml") as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)

    # load model with weights
    ggen = load_model(
        args.result_dir / "models" / "ggen_model.pth",
        args.result_dir / "models" / f"ggen_params_{args.iteration:05d}.pth",
    )
    cgen = load_model(
        args.result_dir / "models" / "cgen_model.pth",
        args.result_dir / "models" / f"cgen_params_{args.iteration:05d}.pth",
    )

    # generate samples
    xg, xc = util.generate_samples(ggen, cgen, args.n_samples, args.batchsize)

    # save samples
    color_dir = args.save_dir / "color"
    geo_dir = args.save_dir / configs["geometric_info"]
    color_dir.mkdir(parents=True, exist_ok=True)
    geo_dir.mkdir(parents=True, exist_ok=True)

    Parallel(n_jobs=-1, verbose=0)(
        [
            delayed(dataio.write_video)(d, geo_dir / "{:06d}.mp4".format(i))
            for i, d in enumerate(xg)
        ]
    )
    Parallel(n_jobs=-1, verbose=0)(
        [
            delayed(dataio.write_video)(c, color_dir / "{:06d}.mp4".format(i))
            for i, c in enumerate(xc)
        ]
    )


if __name__ == "__main__":
    main()
