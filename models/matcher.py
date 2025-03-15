"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""

import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou

class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """
    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1):
        '''Create the matcher

        Args:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        '''
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"
    
    @torch.no_grad()
    def forward(self, outputs, targets):
        ''' Performs the matching

        Args:
            otuputs: This is a dict that contains at least these entries:
            "pred_logits": Tensor of dim (batch_size, num_queries, num_classes) with the classification logits
            "pred_boxes": Tensor of dim (batch_size, num_queries, 4) with the predicted box coordinates
            
            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                           
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates
            
        Returns:
            返回一个列表，其长度等于批次大小，列表中每个元素是一个元组，包含两个值(index_i, index_j)
            假设`num_queries` = 100, 而图像中只有4个真实的框, HungarianMatcher主要是从这 100 个预测框中找出与这4个真实框最匹配的4个预测框
            
            - index_i: 告诉你从 100 个预测框中选出了哪 4 个框（比如第 68 个，第 12 个，第 95 个，第 3 个）。
            - index_j: 告诉你这 4 个被选中的预测框分别对应的是图像中的哪 4 个真实框（比如第 2 个，第 1 个，第 4 个，第 3 个）
            
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        '''
        
        bs, num_queries = outputs["pred_logits"].shape[:2]
        
        # We flatten to compute the cost matrices in a batch
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)     # (batch_size * num_queries, num_classes)   当前预测框对应索引类别的概率分布
        out_bbox = outputs["pred_boxes"].flatten(0, 1)                  # (batch_size * num_queries, 4)
        
        # Also concat the target labels and boxes 
        # targets本来是由有batchsize个图像，每个图像有labels个标签，boxes个锚框，单独把标签和锚框取出来拉成一个长tensor
        tgt_ids = torch.cat([v['labels'] for v in targets])     # (total_num_target_boxes) - 单个图像中真实目标框的数量。数量在同一个批次的不同图像之间会不同
        tgt_bbox = torch.cat([v['boxes'] for v in targets])     # (total_num_target_boxes, 4)
        
        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        # 匈牙利算法的目标是找到代价最小的匹配。这里构建一个(batch_size * num_queries, total_num_target_boxes)的代价矩阵
        # 矩阵中每个元素 C[i][j] 即为第i个预测框匹配到第j个真实目标框的代价是多少
        # 我们希望匹配那些预测为目标类别的概率高的预测框。因此，概率越高，代价应该越低。通过取负数，高概率变成了低代价
        cost_class = -out_prob[:, tgt_ids]
        
        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)    # (batch_size * num_queries, total_num_target_boxes)
        
        # Compute the giou cost between boxes
        # 把batch_size * num_queries个预测框和全部真实的框去计算iou，最后得到(batch_size * num_queries, total_num_target_boxes)的代价矩阵
        # GIoU 的值越高（越接近 1），表示预测框和真实目标框的重叠程度越高，形状也越相似，因此认为这是一个更好的匹配。
        # 在构建代价矩阵时，希望好的匹配具有较低的代价。通过在 GIoU 的结果前加上负号，就将高 GIoU 值（好的匹配）转换成了低代价（更希望被匹配）
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))
        
        # Final cost matrix
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).cpu()
        
        sizes = [len(v["boxes"]) for v in targets]   # eg: [2, 5, 1] 第一个图像有2个目标框，第二个有5个.. len(size) = batch_size
        
        # C: (bs, num_queries, total_num_target_boxes) 对最后一个维度按照size中的值来划分
        # C.split(sizes, -1) 沿着最后一个维度分割，每个段的长度由 sizes 列表指定  -> (bs, num_queries, sizes[i])
        # c[i] -> (num_queries, sizes[i])
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]   # (array([68, 12, 95, 3]), array([1, 0, 3, 2]))
        
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

def build_matcher(args):
    return HungarianMatcher(cost_class=args.set_cost_class, cost_bbox=args.set_cost_bbox, cost_giou=args.set_cost_giou)