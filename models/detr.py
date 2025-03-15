"""
DETR model and criterion classes.
"""
import torch
import torch.nn.functional as F
from torch import nn

from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)

from .backbone import build_backbone
from .matcher import build_matcher
# from .segmentation import (DETRsegm, PostProcessPanoptic, PostProcessSegm,
#                            dice_loss, sigmoid_focal_loss)
from .transformer import build_transformer

class DETR(nn.Module):
    """ This is the DETR module that performs object detection """
    def __init__(self, backbone, transformer, num_classes, num_queries, aux_loss=False):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.num_queries = num_queries  # 100
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone
        self.aux_loss = aux_loss
        
    def forward(self, samples: NestedTensor):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if isinstance(samples, (list, torch.Tensor)):                       # 如果是列表 or tensor就转为nestedTensor
            samples = nested_tensor_from_tensor_list(samples)
        
        features, pos = self.backbone(samples)
        # 假设batch_size为2，其中一个图片为(3, 1167, 1039), 另一个为(3, 480, 480)
        # 把他们转成nestedTensor就会把(3, 480, 480)填充到最大的那一个(3, 1167, 1039)
        # 然后合成一个batch变成(2, 3, 1167, 1039)
        # return src: 个形状为 (2, 3, 1167, 1039) 的张量，其中第二个图像的原始 (3, 480, 480) 区域之外的部分是填充的值
        # mask: 形状为 (2, 1167, 1039) 的二进制张量（布尔型）。对于每个图像，掩码会在填充的像素位置上标记为 True(1)，在原始的图像像素位置上标记为 False(0)
        src, mask = features[-1].decompose()    
        assert mask is not None
        hs = self.transformer(self.input_proj(src), mask, self.query_embed.weight, pos[-1])[0]  # (num_decoder_layers, batch_size, num_queries, hidden_dim)
        
        outputs_class = self.class_embed(hs)        # (num_decoder_layers, batch_size, num_queries, num_classes+1)
        outputs_coord = self.bbox_embed(hs).sigmoid()   # (num_decoder_layers, batch_size, num_queries, 4)
        
        # 取最后一个decoder的输出
        out = {
            'pred_logits': outputs_class[-1],
            'pred_boxes': outputs_coord[-1]
        }
        
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)
        
        return out
    
    # 假设有4层decoder，存储第一层，第二层，第三层的输出
    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b} 
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

class MLP(nn.Module):
    """ Simple multi-layer perceptron (FFN)
    eg: hidden_dim = 256, num_layers = 3 => h = [256, 256] => zip([input_dim, 256, 256], [256, 256, output_dim])
        Linear(input_dim, 256) => Linear(256, 256) => Linear(256, output_dim)
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x