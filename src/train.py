import argparse
import random
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader

from dataset import VideoDataset
from datasets.isogd import preprocess_isogd_dataset
from datasets.mug import preprocess_mug_dataset
from datasets.surreal import preprocess_surreal_dataset
from logger import Logger
from models import (ColorVideoGenerator, DepthVideoGenerator,
                    ImageDiscriminator, VideoDiscriminator)
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


def create_optimizer(models: List[nn.Module], lr: float, decay: float):
    params: List[torch.Tensor] = []
    for m in models:
        params += list(m.parameters())
    return optim.Adam(params, lr=lr, betas=(0.5, 0.999), weight_decay=decay)


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
        configs = yaml.load(f, Loader=yaml.FullLoader)
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

    # initialize logger
    log_path = Path(configs["log_dir"]) / configs["experiment_name"]
    tb_path = Path(configs["tensorboard_dir"]) / configs["experiment_name"]
    logger = Logger(log_path, tb_path)

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
    models = {"dgen": dgen, "cgen": cgen, "idis": idis, "vdis": vdis}

    # optimizers
    opt_gen = create_optimizer([dgen, cgen], **configs["gen"]["optimizer"])
    opt_idis = create_optimizer([idis], **configs["idis"]["optimizer"])
    opt_vdis = create_optimizer([vdis], **configs["vdis"]["optimizer"])
    optimizers = {"gen": opt_gen, "idis": opt_idis, "vdis": opt_vdis}

    # start training
    trainer = Trainer(dataloader, logger, models, optimizers, configs)
    trainer.train()


if __name__ == "__main__":
    main()
