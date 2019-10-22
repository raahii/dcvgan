import argparse
from pathlib import Path

import torch
import yaml
from joblib import Parallel, delayed
from tqdm import tqdm

import dataio
import util
from models import ColorVideoGenerator, MidFeatureVideoGenerator


def load_weight(model, weight_path):
    if torch.cuda.is_available():
        model.cuda()
        model_data = torch.load(weight_path)
    else:
        model_data = torch.load(weight_path, map_location="cpu")

    model.load_state_dict(model_data)

    return model


def load_genrators(result_dir, configs, iteration):
    ggen = MidFeatureVideoGenerator(
        configs["ggen"]["dim_z_content"],
        configs["ggen"]["dim_z_motion"],
        configs["video_length"],
    )
    cgen = ColorVideoGenerator(configs["cgen"]["dim_z_color"])

    weight_path = result_dir / "ggen_{:05d}.pytorch".format(iteration)
    load_weight(ggen, weight_path)
    ggen.eval()

    weight_path = result_dir / "cgen_{:05d}.pytorch".format(iteration)
    load_weight(cgen, weight_path)
    cgen.eval()

    return ggen, cgen


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
        configs = yaml.load(f)

    # load model with weights
    ggen, cgen = load_genrators(args.result_dir, configs, args.iteration)

    # generate samples
    color_dir = args.save_dir / "color"
    depth_dir = args.save_dir / "depth"
    color_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)
    for i in tqdm(range(0, args.n_samples, args.batchsize)):
        dv = ggen.sample_videos(args.batchsize)
        cv = cgen.forward_videos(dv)
        dv = dv.repeat(1, 3, 1, 1, 1)

        dv = util.videos_to_numpy(dv)
        dv = dv.transpose(0, 2, 3, 4, 1)
        cv = util.videos_to_numpy(cv)
        cv = cv.transpose(0, 2, 3, 4, 1)

        Parallel(n_jobs=10, verbose=0)(
            [
                delayed(dataio.write_video)(d, depth_dir / "{:06d}.mp4".format(i + j))
                for j, d in enumerate(dv)
            ]
        )
        Parallel(n_jobs=10, verbose=0)(
            [
                delayed(dataio.write_video)(c, color_dir / "{:06d}.mp4".format(i + j))
                for j, c in enumerate(cv)
            ]
        )

        # for j, (d, c) in enumerate(zip(dv, cv)):
        #     dataio.write_video(d, depth_dir/"{:06d}.mp4".format(i+j))
        #     dataio.write_video(c, color_dir/"{:06d}.mp4".format(i+j))


if __name__ == "__main__":
    main()
