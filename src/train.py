import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml

import util
from dataset import VideoDataLoader, VideoDataset
from discriminator import ImageDiscriminator, VideoDiscriminator
from generator import ColorVideoGenerator, GeometricVideoGenerator
from logger import Logger
from loss import AdversarialLoss, HingeLoss, Loss
from preprocess.isogd import preprocess_isogd_dataset
from preprocess.mug import preprocess_mug_dataset
from preprocess.surreal import preprocess_surreal_dataset
from trainer import Trainer


def _worker_init_fn(worker_id: int):
    random.seed(worker_id)


def fix_seed(value: int):
    """
    Fix every random seed.

    Parameters
    ----------
    value: int
        Seed value.
    """
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


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

    # initialize logger
    log_path = Path(configs["log_dir"]) / configs["experiment_name"]
    tb_path = Path(configs["tensorboard_dir"]) / configs["experiment_name"]
    logger = Logger(log_path, tb_path)
    logger.debug("(experiment)")
    logger.debug(f"name: {configs['experiment_name']}", 1)
    logger.debug(f"directory {configs['log_dir']}", 1)
    logger.debug(f"tensorboard: {configs['tensorboard_dir']}", 1)
    logger.debug(f"geometric_info: {configs['geometric_info']}", 1)
    logger.debug(f"log_interval: {configs['log_interval']}", 1)
    logger.debug(f"log_samples_interval: {configs['log_samples_interval']}", 1)
    logger.debug(f"snapshot_interval: {configs['snapshot_interval']}", 1)
    logger.debug(f"evaluation_interval: {configs['evaluation_interval']}", 1)

    # loss
    loss: Loss
    if configs["loss"] == "adversarial-loss":
        loss = AdversarialLoss()
    elif configs["loss"] == "hinge-loss":
        loss = HingeLoss()
    else:
        logger.error(f"Specified loss is not supported {configs['loss']}")
        sys.exit(1)
    logger.debug(f"loss: {configs['loss']}", 1)

    # prepare dataset
    dataset = VideoDataset(
        configs["dataset"]["name"],
        Path(configs["dataset"]["path"]),
        eval(f'preprocess_{configs["dataset"]["name"]}_dataset'),
        configs["video_length"],
        configs["image_size"],
        configs["dataset"]["number_limit"],
        geometric_info=configs["geometric_info"]["name"],
    )
    dataloader = VideoDataLoader(
        dataset,
        batch_size=configs["batchsize"],
        num_workers=configs["dataset"]["n_workers"],
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        worker_init_fn=_worker_init_fn,
    )
    logger.debug("(dataset)")
    logger.debug(f"name: {dataset.name}", 1)
    logger.debug(f"size: {len(dataset)}", 1)
    logger.debug(f"batchsize: {dataloader.batch_size}", 1)
    logger.debug(f"workers: {dataloader.num_workers}", 1)

    # prepare models
    ggen = GeometricVideoGenerator(
        configs["ggen"]["dim_z_content"],
        configs["ggen"]["dim_z_motion"],
        configs["geometric_info"]["channel"],
        configs["geometric_info"]["name"],
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

    logger.debug("(models)")
    for m in models.values():
        logger.debug(str(m), 1)

    # init weights
    for m in models.values():
        m.apply(util.init_weights)

    # optimizers
    logger.debug("(optimizers)")
    optimizers = {}
    for name, model in models.items():
        lr = configs[name]["optimizer"]["lr"]
        betas = (0.5, 0.999)
        decay = configs[name]["optimizer"]["decay"]
        optimizers[name] = optim.Adam(
            model.parameters(), lr=lr, betas=betas, weight_decay=decay
        )
        logger.debug(
            json.dumps({name: {"betas": betas, "lr": lr, "weight_decay": decay}}), 1
        )

    # start training
    trainer = Trainer(dataloader, logger, models, optimizers, loss, configs)
    trainer.train()


if __name__ == "__main__":
    main()
