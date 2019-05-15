import argparse
import random
from pathlib import Path

import torch
import yaml
from dataset import VideoDataset
from datasets.isogd import preprocess_isogd_dataset
from datasets.mug import preprocess_mug_dataset
from datasets.surreal import preprocess_surreal_dataset
from models import (ColorVideoGenerator, DepthVideoGenerator,
                    ImageDiscriminator, VideoDiscriminator)
from torch.utils.data import DataLoader
from trainer import Trainer


def worker_init_fn(worker_id):
    random.seed(worker_id)


def prepare_dataset(configs):
    if configs["dataset"]["name"] not in ["mug", "isogd", "surreal"]:
        raise NotImplementedError

    return VideoDataset(
        configs["dataset"]["name"],
        Path(configs["dataset"]["path"]),
        eval(f'preprocess_{configs["dataset"]["name"]}_dataset'),
        configs["video_length"],
        configs["image_size"],
        configs["dataset"]["number_limit"],
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
    dgen = DepthVideoGenerator(
        1,
        configs["gen"]["dim_z_content"],
        configs["gen"]["dim_z_motion"],
        configs["gen"]["ngf"],
        configs["video_length"],
    )

    cgen = ColorVideoGenerator(
        1, 3, configs["gen"]["dim_z_color"], configs["gen"]["ngf"]
    )

    idis = ImageDiscriminator(
        1,
        3,
        configs["idis"]["use_noise"],
        configs["idis"]["noise_sigma"],
        configs["idis"]["ndf"],
    )

    vdis = VideoDiscriminator(
        1,
        3,
        configs["vdis"]["use_noise"],
        configs["vdis"]["noise_sigma"],
        configs["vdis"]["ndf"],
    )

    # start training
    trainer = Trainer(dataloader, configs)
    trainer.train(dgen, cgen, idis, vdis)


if __name__ == "__main__":
    main()
