import argparse
import yaml
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import VideoDataset
from dataset import preprocess_isogd_dataset

from models import VideoGenerator, ImageDiscriminator, VideoDiscriminator

from trainer import Trainer

def prepare_dataset(configs):
    if configs["dataset"]["name"] not in ["isogd"]:
        raise NotImplemented
    
    return VideoDataset(
            Path(configs["dataset"]["path"]),
            eval(f'preprocess_{configs["dataset"]["name"]}_dataset'),
            configs['video_length'], 
            configs['image_size'], 
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
    gen = VideoGenerator(
            configs["n_channels"],
            configs["gen"]["dim_z_content"],
            0,
            configs["gen"]["dim_z_motion"],
            configs["video_length"],
            )

    idis = ImageDiscriminator(
            configs["n_channels"],
            configs["idis"]["use_noise"],
            configs["idis"]["noise_sigma"],
            configs["vdis"]["ndf"],
            )

    vdis = VideoDiscriminator(
            configs["n_channels"],
            configs["vdis"]["use_noise"],
            configs["vdis"]["noise_sigma"],
            configs["vdis"]["ndf"],
            )
     
    # start training
    trainer = Trainer(dataloader, configs)
    trainer.train(gen, idis, vdis)

if __name__ == "__main__":
    main()
