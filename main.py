import argparse
import datetime
import random
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

import datasets
import util.misc as utils
from datasets import build_dataset, get_coco_api_from_dataset

def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    
    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', type=str)
    
    return parser
    
def main(args):
    
    dataset_train = build_dataset(image_set='train', args=args)
    dataset_val = build_dataset(image_set='val', args=args)
    
    dataloader_train = DataLoader(dataset=dataset_train, batch_size=1, shuffle=False, collate_fn=utils.collate_fn)
    dataloader_val = DataLoader(dataset=dataset_val, batch_size=1, shuffle=False, collate_fn=utils.collate_fn)

    image, target = next(iter(dataloader_train))
    print(image.tensors.shape)
    print(target)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)