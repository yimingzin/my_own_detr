"""
Various positional encodings for the transformer. 1. Sine 2. Learn
"""
import math
import torch
import einops
from torch import nn

from util.misc import NestedTensor

class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, tensor_list: NestedTensor):
        x = tensor_list.tensors
        mask = tensor_list.mask
        assert mask is not None
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos
    
class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """
    def __init__(self, num_pos_feats = 256):
        super().__init__()
        self.row_embed = nn.Embedding(50, num_pos_feats)    # 学习最多 50 种不同的行位置嵌入（索引从 0 到 49）
        self.col_embed = nn.Embedding(50, num_pos_feats)    # 学习最多 50 种不同的列位置嵌入
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight.data)
        nn.init.uniform_(self.col_embed.weight.data)
    
    def forward(self, tensor_list: NestedTensor):
        x = tensor_list.tensors     # (batch_size, channels, height, width)的特征图
        h, w = x.shape[-2:]
        i = torch.arange(w, device=x.device)    # 行索引
        j = torch.arange(h, device=x.device)    # 列索引
        
        x_emb = self.col_embed(i)   # (w, num_pos_feats)
        y_emb = self.row_embed(j)   # (h, num_pos_feats)
        
        x_emb_repeated = einops.repeat(x_emb, 'w f -> h w f', h=h)
        y_emb_repeated = einops.repeat(y_emb, 'h f -> h w f', w=w)
        
        pos = torch.cat([x_emb_repeated, y_emb_repeated], dim=-1)   # (h, w, 2*f)
        pos = pos.permute(2, 0, 1).unsqueeze(dim=0).repeat(x.shape[0], 1, 1, 1) # (batch_size, 2*f, h, w)
        
        return pos
    
def build_position_encoding(args):
    N_steps = args.hidden_dim // 2          # * Transformer
    
    if args.position_embedding in ('v2', 'sine'):   # * Model parameters
        position_embedding = PositionEmbeddingSine(N_steps, normalize=True)
    elif args.position_embedding in ('v3', 'learned'):
        position_embedding = PositionEmbeddingLearned(N_steps)
    else:
        raise ValueError(f"not supported {args.position_embedding}")
    
    return position_embedding