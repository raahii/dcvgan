import argparse
from pathlib import Path

import torch
import yaml
from joblib import Parallel, delayed
from tqdm import tqdm

import dataio
import util


def load_model(model_path, params_path):
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
    color_dir = args.save_dir / "color"
    depth_dir = args.save_dir / "depth"
    color_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)
    for i in tqdm(range(0, args.n_samples, args.batchsize)):
        with torch.no_grad():
            xg = ggen.sample_videos(args.batchsize)
            xc = cgen.forward_videos(xg)

        if configs["geometric_info"] == "depth":
            xg = xg.repeat(1, 3, 1, 1, 1)
        else:
            raise NotImplementedError

        xg = util.videos_to_numpy(xg)
        xg = xg.transpose(0, 2, 3, 4, 1)
        xc = util.videos_to_numpy(xc)
        xc = xc.transpose(0, 2, 3, 4, 1)

        Parallel(n_jobs=10, verbose=0)(
            [
                delayed(dataio.write_video)(d, depth_dir / "{:06d}.mp4".format(i + j))
                for j, d in enumerate(xg)
            ]
        )
        Parallel(n_jobs=10, verbose=0)(
            [
                delayed(dataio.write_video)(c, color_dir / "{:06d}.mp4".format(i + j))
                for j, c in enumerate(xc)
            ]
        )

        # for j, (d, c) in enumerate(zip(dv, cv)):
        #     dataio.write_video(d, depth_dir/"{:06d}.mp4".format(i+j))
        #     dataio.write_video(c, color_dir/"{:06d}.mp4".format(i+j))


if __name__ == "__main__":
    main()
