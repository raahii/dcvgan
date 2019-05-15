import argparse
import random
from pathlib import Path

import torch
import yaml
from dataset import ConditionVariableVideoDataset
from datasets.surreal import preprocess_surreal_dataset, segm_part_colors
from models import (ColorVideoGenerator, DepthVideoGenerator,
                    ImageDiscriminator, SegmentationVideoGenerator,
                    VideoDiscriminator)
from torch.utils.data import DataLoader
from trainer_segm_condition import Trainer


def worker_init_fn(worker_id):
    random.seed(worker_id)


def prepare_dataset(configs):
    if configs["dataset"]["name"] not in ["surreal"]:
        raise NotImplementedError

    return ConditionVariableVideoDataset(
        configs["dataset"]["name"],
        Path(configs["dataset"]["path"]),
        eval(f'preprocess_{configs["dataset"]["name"]}_dataset'),
        configs["video_length"],
        configs["image_size"],
        configs["dataset"]["number_limit"],
        configs["dataset"]["cond"],
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        "-c",
        default="configs/default.yml",
        help="training configuration file",
    )
    args = parser.parse_args()

    # parse config yaml
    with open(args.config) as f:
        configs = yaml.load(f)
    configs["config_path"] = args.config

    # prepare dataset
    dataset = prepare_dataset(configs)
    dataloader = DataLoader(
        dataset,
        batch_size=configs["batchsize"],
        num_workers=configs["dataset"]["n_workers"],
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
    )

    # prepare models
    cond = configs["dataset"]["cond"]
    if cond == "depth":
        in_ch = 1
        ggen = DepthVideoGenerator(
            in_ch,
            configs["gen"]["dim_z_content"],
            configs["gen"]["dim_z_motion"],
            configs["gen"]["ngf"],
            configs["video_length"],
        )
    elif cond == "segm":
        in_ch = len(segm_part_colors)
        ggen = SegmentationVideoGenerator(
            in_ch,
            configs["gen"]["dim_z_content"],
            configs["gen"]["dim_z_motion"],
            configs["gen"]["ngf"],
            configs["video_length"],
        )
    else:
        raise NotImplementedError

    cgen = ColorVideoGenerator(in_ch, 3, configs["gen"]["dim_z_color"])

    idis = ImageDiscriminator(
        in_ch,
        3,
        configs["idis"]["use_noise"],
        configs["idis"]["noise_sigma"],
        configs["idis"]["ndf"],
    )

    vdis = VideoDiscriminator(
        in_ch,
        3,
        configs["vdis"]["use_noise"],
        configs["vdis"]["noise_sigma"],
        configs["vdis"]["ndf"],
    )

    # start training
    trainer = Trainer(dataloader, configs)
    trainer.train(ggen, cgen, idis, vdis)


if __name__ == "__main__":
    main()
