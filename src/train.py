import argparse
import yaml
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import VideoDataset
from datasets.isogd import preprocess_isogd_dataset
from datasets.mug import preprocess_mug_dataset
from datasets.surreal import preprocess_surreal_dataset

from models import DepthVideoGenerator, ColorVideoGenerator
from models import ImageDiscriminator, VideoDiscriminator

from trainer import Trainer

def prepare_dataset(configs):
    if configs["dataset"]["name"] not in ["mug", "isogd", "surreal"]:
        raise NotImplementedError
    
    return VideoDataset(
            configs["dataset"]["name"],
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
                    )
    
    # prepare models
    dgen = DepthVideoGenerator(
            configs["gen"]["dim_z_content"],
            configs["gen"]["dim_z_motion"],
            configs["video_length"],
            )

    cgen = ColorVideoGenerator(
            configs["gen"]["dim_z_color"],
            )

    idis = ImageDiscriminator(
            configs["idis"]["use_noise"],
            configs["idis"]["noise_sigma"],
            configs["idis"]["ndf"],
            )

    vdis = VideoDiscriminator(
            configs["vdis"]["use_noise"],
            configs["vdis"]["noise_sigma"],
            configs["vdis"]["ndf"],
            )

    # start training
    trainer = Trainer(dataloader, configs)
    trainer.train(dgen, cgen, idis, vdis)

if __name__ == "__main__":
    main()
