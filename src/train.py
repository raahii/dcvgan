import argparse
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader

import util
from dataset import VideoDataset
from discriminator import ImageDiscriminator, VideoDiscriminator
from generator import ColorVideoGenerator, new_geometric_generator
from logger import Logger
from preprocess.isogd import preprocess_isogd_dataset
from preprocess.mug import preprocess_mug_dataset
from preprocess.surreal import preprocess_surreal_dataset
from trainer import Trainer


def worker_init_fn(worker_id):
    random.seed(worker_id)


def fix_seed(value):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def new_dataset(configs):
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

    # fix seed
    fix_seed(configs["seed"])

    # prepare dataset
    dataset = new_dataset(configs)
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
    ggen = new_geometric_generator(configs["geometric_info"])(
        configs["ggen"]["dim_z_content"],
        configs["ggen"]["dim_z_motion"],
        configs["ggen"]["ngf"],
        configs["video_length"],
    )

    cgen = ColorVideoGenerator(
        ggen.channel,
        configs["cgen"]["dim_z_color"],
        configs["cgen"]["ngf"],
        configs["video_length"],
    )

    idis = ImageDiscriminator(
        ggen.channel,
        cgen.channel,
        configs["idis"]["use_noise"],
        configs["idis"]["noise_sigma"],
        configs["idis"]["ndf"],
    )

    vdis = VideoDiscriminator(
        ggen.channel,
        cgen.channel,
        configs["vdis"]["use_noise"],
        configs["vdis"]["noise_sigma"],
        configs["vdis"]["ndf"],
    )
    models = {"ggen": ggen, "cgen": cgen, "idis": idis, "vdis": vdis}

    # init weights
    for m in models.values():
        m.apply(util.init_weights)

    # optimizers
    opt_ggen = create_optimizer([ggen], **configs["ggen"]["optimizer"])
    opt_cgen = create_optimizer([cgen], **configs["cgen"]["optimizer"])
    opt_idis = create_optimizer([idis], **configs["idis"]["optimizer"])
    opt_vdis = create_optimizer([vdis], **configs["vdis"]["optimizer"])
    optimizers = {
        "ggen": opt_ggen,
        "cgen": opt_cgen,
        "idis": opt_idis,
        "vdis": opt_vdis,
    }

    # start training
    trainer = Trainer(dataloader, logger, models, optimizers, configs)
    trainer.train()


if __name__ == "__main__":
    main()
