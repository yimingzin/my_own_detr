import argparse
import datetime
import random
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, DistributedSampler

import datasets
import util.misc as utils
from datasets import build_dataset, get_coco_api_from_dataset

def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    
    # * Train
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    
    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='learned', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    
    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    
    # * Transformer
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--pre_norm', action='store_true')      # 使用 --pre_norm 参数，args.pre_norm 的值将被设置为 True
    parser.add_argument('--num_queries', default=100, type=int,
                        help="Number of query slots")
    
     # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")
    
    # dataset parameters
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', type=str)
    parser.add_argument('--num_workers', default=2, type=int)
    
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    
    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    
    # * Matcher - set cost weight
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    
    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")
    
    return parser
    
def main(args):
    utils.init_distributed_mode(args)   # 初始化分布式训练环境 set args.rank/.world_size(if have)/.gpu/.distributed
    print("git:\n  {}\n".format(utils.get_sha()))
    
    print(args)
    device = torch.device(args.device)
    
    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # initial dataset / Dataloader
    dataset_train = build_dataset(image_set='train', args=args)
    dataset_val = build_dataset(image_set='val', args=args)
    
    if args.distributed:
        # 分布式采样
        sampler_train = DistributedSampler(dataset_train)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    
    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True
    )
    
    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train, 
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)
    
    output_dir = Path(args.output_dir)
    
    image, target = next(iter(data_loader_train))
    print(image.tensors.shape)
    from models.backbone import build_backbone
    model = build_backbone(args)
    out, pos = model(image)
    print("backbone out: ")
    for i in out:
        print(i.tensors.shape)
    
    print("pos out: ")
    for j in pos:
        print(j.shape)
        
    from models.transformer import build_transformer
    model_transofmer = build_transformer(args)
    
    src, mask = out[-1].decompose()
    query_embed = nn.Embedding(args.num_queries, args.hidden_dim)
    input_proj = nn.Conv2d(in_channels=2048, out_channels=args.hidden_dim, kernel_size=1)
    hs, memory = model_transofmer(input_proj(src), mask, query_embed.weight, pos[-1])
    
    print(hs.shape)
    print(memory.shape)
    
    # -----------
    from models.detr import DETR
    model_detr = DETR(build_backbone(args),build_transformer(args), 91,args.num_queries ,args.aux_loss)
    
    out = model_detr(image)
    for k, v in out.items():
        if k == 'aux_outputs':
            for vv in v:
                for idx, value in vv.items():
                    print(f'aux_outputs{idx}: {value.shape}')
            break
        print(f'{k} : {v.shape}')
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)