"""
DETR Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
from typing import Optional, List
from einops import rearrange
import torch
import torch.nn.functional as F
from torch import nn, Tensor

class Transformer(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation='relu', normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()
        
        # if normalize_before 则应用前置归一化并对最后的输出归一化
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None  # 是否对最后的输出归一化
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        # decoder
        decoder_layer = TransformerDecoderLayer(d_model,nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm, 
                                          return_intermediate=return_intermediate_dec)
        self._reset_parameters()
        self.d_model = d_model
        self.nhead = nhead
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos_embed):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        mask = mask.flatten(1)

        tgt = torch.zeros_like(query_embed)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        hs = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                          pos=pos_embed, query_pos=query_embed)
        '''
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = rearrange(src, 'b c h w ->(h w) b c')                 # HW, B, C
        pos_embed = rearrange(pos_embed, 'b c h w ->(h w) b c')
        query_embed = rearrange(query_embed, 'n c -> n b c', b=bs)
        mask = mask.flatten(1)
        # mask = torch.flatten(mask, start_dim=1, end_dim=-1)
        
        tgt = torch.zeros_like(query_embed)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)    #HW, B, C
        hs = self.decoder(tgt, memory, memory_key_padding_mask=mask, pos=pos_embed, query_pos=query_embed) # n, b, c
        '''
        return hs.transpose(1, 2), memory.permute(1, 2, 0).view(bs, c, h, w)    # (n, c, bs)    (bs, c, h, w)
        
        
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu", normaliza_before=False):
        super().__init__()
        
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        self.activation = _get_activation_fn(activation=activation)
        self.normalize_before = normaliza_before
    
    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        src,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None
    ):
        ''' original Transformer take post forward
        
        out_1 = Norm1(Dropout(Attn(src_+_pos, src_+_pos, src)) + src)
        out_2 = Norm2(Dropout(FFD(out_1)) + out_1)
        '''
        # 位置信息帮助模型判断关注哪里(query, key) value是关注的内容
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]    # return tuple (attn_output, attn_weights)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))   # feedforward
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src
    
    def forward_pre(
        self,
        src,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None
    ):
        
        '''
        preNorm: for deeper Network
        
        out_1 = src + Dropout(Attn(q=pos_+_Norm(src), k=pos_+_Norm(src), v=Norm(sr_+_query_posc)))Norm(obj_query)_+_query_pos, V=Norm(obj_query)
        out_2 = out_1 + Dropout(FFD(Norm(out__+_query_pos1)))Norm(obj_query)_+_query_pos, V=Norm(obj_query)
        '''
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src
    
    def forward(
        self,
        src,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None
    ):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)
    
class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
    
    def forward(
        self,
        src,
        mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None
    ):
        
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)
        return output

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normaliza_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout)
        
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        
        self.activation = _get_activation_fn(activation=activation)
        self.normalize_before = normaliza_before
        
    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos
    
    def forward_post(
        self,
        tgt,        # object query
        memory,     # enc_out
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,       # 编码器输出的位置嵌入
        query_pos: Optional[Tensor] = None  # 
    ):
        '''
        out_1 = Norm[Dropout(Attention(Q=obj_query+pos_query, K=obj_query+pos_query, V=obj_query)) + obj_qu_+_query_posery]Norm(obj_query)_+_query_pos, V=Norm(obj_query)
        
        out_2 = Norm[Dropout(Attention(Q=out_1+pos_query, K=encoder_out(memory) + pos, V=encoder_out)) + ou_+_query_post_1]Norm(obj_query)_+_query_pos, V=Norm(obj_query)
        
        out_3 = Norm[FFD(out_2) + ou_+_query_post_2]Norm(obj_query)_+_query_pos, V=Norm(obj_query)
        '''
        # attention_self
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        
        # attention_cross
        tgt2 = self.multihead_attn(
            query = self.with_pos_embed(tgt, query_pos),
            key = self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask = memory_mask,
            key_padding_mask = memory_key_padding_mask
        )[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        
        # FFD
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt
    
    def forward_pre(
        self,
        tgt,        # object query
        memory,     # enc_out
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,       # 编码器输出的位置嵌入
        query_pos: Optional[Tensor] = None  #
    ):
        '''
        out_1 = Dropout[SELF_ATTN(Q=Norm(obj_query)_+_query_pos, K=Norm(obj_query)_+_query_pos, V=Norm(obj_query))] + obj_query
        out_2 = Dropout[CROSS_ATTN(Q=Norm(out_1)_+_query_pos, K=enc_out + pos, V=enc_out)] + out_1
        out_3 = Dropout[FFD(Norm(out_2))] + out_2
        '''
        # self attention
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        
        # CROSS Attention
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt2, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask
        )[0]
        tgt = tgt + self.dropout2(tgt2)
        
        # FFD
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        
        return tgt

    
    def forward(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None
    ):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
    

class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate  # 是否返回中间层, if True将中间层输出存入list
        
    def forward(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memeory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None
    ):
        
        output = tgt
        intermediate = []
        
        for layer in self.layers:
            output = layer(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memeory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                pos=pos,
                query_pos=query_pos
            )
            if self.return_intermediate:
                intermediate.append(self.norm(output))
        
        # 如果存在归一化层，在遍历完所有解码器之后对最终的outpu再进行归一化
        if self.norm is not None: 
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()      # 弹出最后一个元素
                intermediate.append(output)   # 添加

        if self.return_intermediate:
            return torch.stack(intermediate)
        
        return output.unsqueeze(dim=0)
    
def _get_activation_fn(activation):
    """return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

def _get_clones(module, N):
    '''
    given one pytorch module and repeat for N times
    '''    
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

# * Transformer
def build_transformer(args):
    return Transformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
    )