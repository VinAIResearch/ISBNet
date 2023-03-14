# ------------------------------------------------------------------------
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import nn

from .model_utils import batch_giou_cross


# from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou, generalized_box_cdist
# import functools


def compute_dice_cost(inputs, targets):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    # inputs = inputs.sigmoid()
    # # inputs = inputs.flatten(1)
    # numerator = 2 * (inputs * targets).sum(1)
    # denominator = inputs.sum(-1) + targets.sum(-1)
    # loss = 1 - (numerator + 1) / (denominator + 1)
    # return loss

    inputs = inputs.sigmoid()
    # inputs = inputs.flatten(1)
    numerator = 2 * torch.einsum("nc,mc->nm", inputs, targets)
    denominator = inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss


# def compute_bce_cost(inputs, targets):
#     n_pred = inputs.shape[0]
#     n_target = targets.shape[0]
#     n_point = inputs.shape[1]

#     bce_cost = F.binary_cross_entropy_with_logits(
#         inputs[:, None, :].repeat(1, n_target, 1),
#         targets[None, :, :].repeat(n_pred, 1, 1),
#         reduction="none")

#     bce_cost = bce_cost.sum(dim=-1) / n_point

#     return bce_cost

# @torch.jit.script
def sigmoid_bce_loss(inputs: torch.Tensor, targets: torch.Tensor):
    """
    Args:
        inputs: (num_querys, N)
        targets: (num_inst, N)
    Returns:
        Loss tensor
    """
    N = inputs.shape[1]

    pos = F.binary_cross_entropy_with_logits(inputs, torch.ones_like(inputs), reduction="none")
    neg = F.binary_cross_entropy_with_logits(inputs, torch.zeros_like(inputs), reduction="none")

    loss = torch.einsum("nc,mc->nm", pos, targets) + torch.einsum("nc,mc->nm", neg, (1 - targets))

    return loss / N


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, ignore_label=-100, cost_class=1, cost_bbox=1, cost_giou=1):
        """Creates the matcher
        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()

        self.ignore_label = ignore_label

        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    def get_gt(self, instance_labels_b, instance_cls, instance_box):
        n_points = instance_labels_b.shape[0]

        # unique_query_inst = torch.unique(query_instance_labels_b)
        # unique_query_inst = unique_query_inst[unique_query_inst != self.ignore_label]

        unique_inst = torch.unique(instance_labels_b)
        unique_inst = unique_inst[unique_inst != self.ignore_label]

        # exist_query = torch.zeros_like(unique_inst)
        # for i in range(len(unique_inst)):
        #     if (unique_query_inst == unique_inst[i]).sum() > 0:
        #          exist_query[i] = 1
        # unique_inst = unique_inst[exist_query.bool()]

        cls_label = instance_cls[unique_inst]  # n_inst
        cls_label_cond = cls_label >= 0
        cls_labels_b = cls_label[cls_label_cond]
        fg_unique_inst = unique_inst[cls_label_cond]

        box_label = instance_box[unique_inst]
        box_labels_b = box_label[cls_label_cond]

        n_inst_gt = fg_unique_inst.shape[0]

        if n_inst_gt == 0:
            return None

        mask_labels_b = torch.zeros((n_inst_gt, n_points), device=instance_labels_b.device, dtype=torch.float)
        for i in range(n_inst_gt):
            inst_id = fg_unique_inst[i]
            mask_labels_b[i] = (instance_labels_b == inst_id).float()

        # breakpoint()

        return cls_labels_b, mask_labels_b, box_labels_b

    def get_match(
        self,
        cls_labels_b,
        mask_labels_b,
        box_labels_b,
        cls_preds_b,
        mask_logits_preds_b,
        conf_preds_b,
        box_preds_b,
        dup_gt=6,
    ):

        n_inst_gt, n_points = mask_labels_b.shape[:2]

        # n_queries = cls_preds_b.shape[0]

        dice_cost = compute_dice_cost(
            mask_logits_preds_b,
            mask_labels_b,
        )
        # dice_cost = compute_dice(
        #     mask_logits_preds_b.reshape(-1, 1, n_points).repeat(1, n_inst_gt, 1).flatten(0, 1),
        #     mask_labels_b.reshape(1, -1, n_points).repeat(n_queries, 1, 1).flatten(0, 1),
        # )

        bce_cost = sigmoid_bce_loss(mask_logits_preds_b, mask_labels_b)

        # dice_cost = dice_cost.reshape(n_queries, n_inst_gt)

        cls_preds_b_sm = F.softmax(cls_preds_b, dim=-1)

        class_cost = -cls_preds_b_sm[:, cls_labels_b]

        conf_cost = -conf_preds_b[:, None].repeat(1, n_inst_gt)

        # box_preds_b = (
        #     box_preds_b[:, None, :].repeat(1, n_inst_gt, 1).reshape(n_queries * n_inst_gt, -1)
        # )  # n_queries * n_inst, 6
        # box_labels_b = (
        #     box_labels_b[None, :, :].repeat(n_queries, 1, 1).reshape(n_queries * n_inst_gt, -1)
        # )  # n_queries * n_inst, 6
        # box_preds_b = box_preds_b.reshape(n_queries*n_inst_gt, -1)
        # instance_box_b = instance_box_b.reshape(n_queries*n_inst_gt, -1)
        _, giou_b = batch_giou_cross(box_preds_b, box_labels_b)
        # giou_b = giou_b.reshape(n_queries, n_inst_gt)
        giou_cost = -giou_b

        # final_cost = 1 * class_cost + 5 * dice_cost + 5 * bce_cost + 1 * conf_cost + 1 * giou_cost
        final_cost = 0.5 * class_cost + 1 * dice_cost + 1 * bce_cost + 0.2 * conf_cost + 0.2 * giou_cost

        final_cost = final_cost.detach()

        final_cost[torch.isnan(final_cost)] = 100000
        final_cost[torch.isinf(final_cost)] = 100000

        main_final_cost = final_cost.cpu().numpy()

        row_inds, col_inds = linear_sum_assignment(main_final_cost)

        aux_final_cost = final_cost.repeat(1, dup_gt).cpu().numpy()
        aux_row_inds, aux_col_inds = linear_sum_assignment(aux_final_cost)

        return row_inds, col_inds, aux_row_inds, aux_col_inds

    @torch.no_grad()
    def forward_dup(
        self,
        cls_preds,
        mask_logits_preds,
        conf_preds,
        box_preds,
        dc_inst_mask_arr,
        dup_gt=1,
    ):
        # cls_preds : batch x classes x n_queries
        batch_size, n_queries, _ = cls_preds.shape

        gt_dict = dict(
            row_indices=[],
            inst_labels=[],
            cls_labels=[],
            box_labels=[],
        )

        aux_gt_dict = dict(
            row_indices=[],
            inst_labels=[],
            cls_labels=[],
            box_labels=[],
        )

        for b in range(batch_size):

            if dc_inst_mask_arr[b] is None:
                gt_dict["row_indices"].append(None)
                gt_dict["inst_labels"].append(None)
                gt_dict["cls_labels"].append(None)
                gt_dict["box_labels"].append(None)

                aux_gt_dict["row_indices"].append(None)
                aux_gt_dict["inst_labels"].append(None)
                aux_gt_dict["cls_labels"].append(None)
                aux_gt_dict["box_labels"].append(None)
                continue

            # NOTE gt
            cls_labels_b, mask_labels_b, box_labels_b = (
                dc_inst_mask_arr[b]["cls"],
                dc_inst_mask_arr[b]["mask"],
                dc_inst_mask_arr[b]["box"],
            )

            row_inds, col_inds, aux_row_inds, aux_col_inds = self.get_match(
                cls_labels_b,
                mask_labels_b,
                box_labels_b,
                cls_preds[b],
                mask_logits_preds[b],
                conf_preds[b],
                box_preds[b],
                dup_gt=dup_gt,
            )

            gt_dict["row_indices"].append(row_inds)
            gt_dict["inst_labels"].append(mask_labels_b[col_inds])
            gt_dict["cls_labels"].append(cls_labels_b[col_inds])
            gt_dict["box_labels"].append(box_labels_b[col_inds])

            # NOTE aux gt
            aux_cls_labels_b = cls_labels_b.repeat(dup_gt)
            aux_mask_labels_b = mask_labels_b.repeat(dup_gt, 1)
            aux_box_labels_b = box_labels_b.repeat(dup_gt, 1)

            # aux_row_inds, aux_col_inds = self.get_match(aux_cls_labels_b, aux_mask_labels_b, cls_preds[b], mask_logits_preds[b], conf_preds[b])

            aux_gt_dict["row_indices"].append(aux_row_inds)
            aux_gt_dict["inst_labels"].append(aux_mask_labels_b[aux_col_inds])
            aux_gt_dict["cls_labels"].append(aux_cls_labels_b[aux_col_inds])
            aux_gt_dict["box_labels"].append(aux_box_labels_b[aux_col_inds])

        return gt_dict, aux_gt_dict


def build_matcher(args):
    return HungarianMatcher(cost_class=args.set_cost_class, cost_bbox=args.set_cost_bbox, cost_giou=args.set_cost_giou)
