import argparse
import yaml
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import VideoDataset
from dataset import preprocess_isogd_dataset

from models import DepthVideoGenerator, ColorVideoGenerator
from models import ImageDiscriminator, VideoDiscriminator

from trainer import Trainer

def prepare_dataset(configs):
    if configs["dataset"]["name"] not in ["isogd"]:
        raise NotImplemented
    
    return VideoDataset(
            Path(configs["dataset"]["path"]),
            eval(f'preprocess_{configs["dataset"]["name"]}_dataset'),
            configs['video_length'], 
            configs['image_size'], 
            configs["dataset"]['number_limit'], 
            )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', default='configs/default.yml',
                        help='training configuration file')
    args = parser.parse_args()
    
    # parse config yaml
    with open(args.config) as f:
        configs = yaml.load(f)
    
    # prepare dataset
    dataset = prepare_dataset(configs)
    dataloader = DataLoader(
                    dataset, 
                    batch_size=configs["batchsize"], 
                    num_workers=configs["dataset"]["n_workers"],
                    shuffle=True, 
                    drop_last=True, 
                    )
    
    # prepare models
    dgen = DepthVideoGenerator(
            configs["dgen"]["n_channels"],
            configs["dgen"]["dim_z_content"],
            0,
            configs["dgen"]["dim_z_motion"],
            configs["video_length"],
            )

    cgen = ColorVideoGenerator(
            configs["cgen"]["in_ch"],
            configs["cgen"]["out_ch"],
            configs["cgen"]["dim_z_color"],
            )

    idis = ImageDiscriminator(
            configs["idis"]["n_channels"],
            configs["idis"]["use_noise"],
            configs["idis"]["noise_sigma"],
            configs["idis"]["ndf"],
            )

    vdis = VideoDiscriminator(
            configs["vdis"]["n_channels"],
            configs["vdis"]["use_noise"],
            configs["vdis"]["noise_sigma"],
            configs["vdis"]["ndf"],
            )
     
    # start training
    trainer = Trainer(dataloader, configs)
    trainer.train(dgen, cgen, idis, vdis)

if __name__ == "__main__":
    main()
