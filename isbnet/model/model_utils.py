import numpy as np
import torch
import torch.nn.functional as F

import torch_scatter
from ..ops import ballquery_batchflat


def calc_square_dist(a, b, norm=True):
    """
    Calculating square distance between a and b
    a: [bs, n, c]
    b: [bs, m, c]
    """
    n = a.shape[1]
    m = b.shape[1]
    a_square = a.unsqueeze(dim=2)  # [bs, n, 1, c]
    b_square = b.unsqueeze(dim=1)  # [bs, 1, m, c]
    a_square = torch.sum(a_square * a_square, dim=-1)  # [bs, n, 1]
    b_square = torch.sum(b_square * b_square, dim=-1)  # [bs, 1, m]
    a_square = a_square.repeat((1, 1, m))  # [bs, n, m]
    b_square = b_square.repeat((1, n, 1))  # [bs, n, m]

    coor = torch.matmul(a, b.transpose(1, 2))  # [bs, n, m]

    if norm:
        dist = a_square + b_square - 2.0 * coor  # [bs, npoint, ndataset]
        # dist = torch.sqrt(dist)
    else:
        dist = a_square + b_square - 2 * coor
        # dist = torch.sqrt(dist)
    return dist


def nms_and_merge(proposals_pred, scores, classes, threshold):
    proposals_pred = proposals_pred.float()  # (nProposal, N), float, cuda
    intersection = torch.mm(proposals_pred, proposals_pred.t())  # (nProposal, nProposal), float, cuda
    proposals_pointnum = proposals_pred.sum(1)  # (nProposal), float, cuda
    proposals_pn_h = proposals_pointnum.unsqueeze(-1).repeat(1, proposals_pointnum.shape[0])
    proposals_pn_v = proposals_pointnum.unsqueeze(0).repeat(proposals_pointnum.shape[0], 1)
    ious = intersection / (proposals_pn_h + proposals_pn_v - intersection)

    ixs = torch.argsort(scores, descending=True)

    pick = []
    proposals = []
    while len(ixs) > 0:
        i = ixs[0]
        pick.append(i)

        pivot_cls = classes[i]

        iou = ious[i, ixs[1:]]
        other_cls = classes[ixs[1:]]

        condition = (iou > threshold) & (other_cls == pivot_cls)
        remove_ixs = torch.nonzero(condition).view(-1) + 1

        remove_ixs = torch.cat([remove_ixs, torch.tensor([0], device=remove_ixs.device)]).long()

        idx_to_merge = ixs[remove_ixs]
        proposals_to_merge = proposals_pred[idx_to_merge, :]
        n_proposals_to_merge = len(remove_ixs)
        proposal_merged = torch.sum(proposals_to_merge, dim=0) >= n_proposals_to_merge * 0.5

        proposals.append(proposal_merged)

        mask = torch.ones_like(ixs, device=ixs.device, dtype=torch.bool)
        mask[remove_ixs] = False
        ixs = ixs[mask]

    pick = torch.tensor(pick, dtype=torch.long, device=scores.device)
    proposals = torch.stack(proposals, dim=0).bool()
    return pick, proposals


def standard_nms(proposals_pred, categories, scores, boxes, threshold=0.2):
    ixs = torch.argsort(scores, descending=True)
    # n_samples = len(ixs)

    intersection = torch.einsum("nc,mc->nm", proposals_pred.type(scores.dtype), proposals_pred.type(scores.dtype))
    proposals_pointnum = proposals_pred.sum(1)  # (nProposal), float, cuda
    ious = intersection / (proposals_pointnum[None, :] + proposals_pointnum[:, None] - intersection)

    pick = []
    while len(ixs) > 0:
        i = ixs[0]
        pick.append(i)

        pivot_cls = categories[i]

        iou = ious[i, ixs[1:]]
        other_cls = categories[ixs[1:]]

        condition = (iou > threshold) & (other_cls == pivot_cls)
        # condition = (iou > threshold)
        remove_ixs = torch.nonzero(condition).view(-1) + 1

        remove_ixs = torch.cat([remove_ixs, torch.tensor([0], device=remove_ixs.device)]).long()

        mask = torch.ones_like(ixs, device=ixs.device, dtype=torch.bool)
        mask[remove_ixs] = False
        ixs = ixs[mask]
    get_idxs = torch.tensor(pick, dtype=torch.long, device=scores.device)

    return proposals_pred[get_idxs], categories[get_idxs], scores[get_idxs], boxes[get_idxs]


def matrix_nms(proposals_pred, categories, scores, boxes, final_score_thresh=0.1, topk=-1):
    if len(categories) == 0:
        return proposals_pred, categories, scores, boxes

    ixs = torch.argsort(scores, descending=True)
    n_samples = len(ixs)

    categories_sorted = categories[ixs]
    proposals_pred_sorted = proposals_pred[ixs]
    scores_sorted = scores[ixs]
    boxes_sorted = boxes[ixs]

    # (nProposal, N), float, cuda
    intersection = torch.einsum(
        "nc,mc->nm", proposals_pred_sorted.type(scores.dtype), proposals_pred_sorted.type(scores.dtype)
    )
    proposals_pointnum = proposals_pred_sorted.sum(1)  # (nProposal), float, cuda
    ious = intersection / (proposals_pointnum[None, :] + proposals_pointnum[:, None] - intersection)

    # label_specific matrix.
    categories_x = categories_sorted[None, :].expand(n_samples, n_samples)
    label_matrix = (categories_x == categories_x.transpose(1, 0)).float().triu(diagonal=1)

    # IoU compensation
    compensate_iou, _ = (ious * label_matrix).max(0)
    compensate_iou = compensate_iou.expand(n_samples, n_samples).transpose(1, 0)

    # IoU decay
    decay_iou = ious * label_matrix

    # matrix nms
    sigma = 2.0
    decay_matrix = torch.exp(-1 * sigma * (decay_iou**2))
    compensate_matrix = torch.exp(-1 * sigma * (compensate_iou**2))
    decay_coefficient, _ = (decay_matrix / compensate_matrix).min(0)

    # update the score.
    cate_scores_update = scores_sorted * decay_coefficient

    if topk != -1:
        _, get_idxs = torch.topk(
            cate_scores_update, k=min(topk, cate_scores_update.shape[0]), largest=True, sorted=False
        )
    else:
        get_idxs = torch.nonzero(cate_scores_update >= final_score_thresh).view(-1)

    return (
        proposals_pred_sorted[get_idxs],
        categories_sorted[get_idxs],
        cate_scores_update[get_idxs],
        boxes_sorted[get_idxs],
    )


def nms(proposals_pred, categories, scores, boxes, test_cfg):
    if test_cfg.type_nms == "matrix":
        return matrix_nms(proposals_pred, categories, scores, boxes, topk=test_cfg.topk)
    elif test_cfg.type_nms == "standard":
        return standard_nms(proposals_pred, categories, scores, boxes, threshold=test_cfg.nms_threshold)
    else:
        raise RuntimeError("Invalid nms type")


def compute_dice_loss(inputs, targets):
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
    # inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum()
    denominator = inputs.sum() + targets.sum()
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss


def sigmoid_focal_loss(inputs, targets, weights, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    # return loss.sum()

    loss = (loss * weights).sum()
    return loss


def compute_sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / (num_boxes + 1e-6)


@torch.no_grad()
def iou_aabb(pt_offsets_vertices=None, pt_offset_vertices_labels=None, coords=None, box_preds=None, box_gt=None):
    if coords is not None:
        coords_min_pred = coords + pt_offsets_vertices[:, 0:3]  # N x 3
        coords_max_pred = coords + pt_offsets_vertices[:, 3:6]  # N x 3

        coords_min_gt = coords + pt_offset_vertices_labels[:, 0:3]  # N x 3
        coords_max_gt = coords + pt_offset_vertices_labels[:, 3:6]  # N x 3
    else:
        coords_min_pred = box_preds[:, 0:3]  # n_queries x 3
        coords_max_pred = box_preds[:, 3:6]  # n_queries x 3

        coords_min_gt = box_gt[:, 0:3]  # n_inst x 3
        coords_max_gt = box_gt[:, 3:6]  # n_inst x 3

    upper = torch.min(coords_max_pred, coords_max_gt)  # Nx3
    lower = torch.max(coords_min_pred, coords_min_gt)  # Nx3

    intersection = torch.prod(torch.clamp((upper - lower), min=0.0), -1)  # N

    gt_volumes = torch.prod(torch.clamp((coords_max_gt - coords_min_gt), min=0.0), -1)
    pred_volumes = torch.prod(torch.clamp((coords_max_pred - coords_min_pred), min=0.0), -1)

    union = gt_volumes + pred_volumes - intersection
    iou = intersection / (union + 1e-6)
    return iou


def giou_aabb(pt_offsets_vertices, pt_offset_vertices_labels, coords=None):
    if coords is None:
        # coords = torch.zeros((pt_offsets_vertices.shape[0], 3), dtype=torch.float, device=pt_offsets_vertices.device)
        coords = 0.0

    coords_min_pred = coords + pt_offsets_vertices[:, 0:3]  # N x 3
    coords_max_pred = coords + pt_offsets_vertices[:, 3:6]  # N x 3

    coords_min_gt = coords + pt_offset_vertices_labels[:, 0:3]  # N x 3
    coords_max_gt = coords + pt_offset_vertices_labels[:, 3:6]  # N x 3

    upper = torch.min(coords_max_pred, coords_max_gt)  # Nx3
    lower = torch.max(coords_min_pred, coords_min_gt)  # Nx3

    intersection = torch.prod(torch.clamp((upper - lower), min=0.0), -1)  # N

    gt_volumes = torch.prod(torch.clamp((coords_max_gt - coords_min_gt), min=0.0), -1)
    pred_volumes = torch.prod(torch.clamp((coords_max_pred - coords_min_pred), min=0.0), -1)

    union = gt_volumes + pred_volumes - intersection
    iou = intersection / (union + 1e-6)

    upper_bound = torch.max(coords_max_pred, coords_max_gt)
    lower_bound = torch.min(coords_min_pred, coords_min_gt)

    volumes_bound = torch.prod(torch.clamp((upper_bound - lower_bound), min=0.0), -1)  # N

    giou = iou - (volumes_bound - union) / (volumes_bound + 1e-6)

    return iou, giou


def cal_iou(volumes, x1, y1, z1, x2, y2, z2, sort_indices, index):
    rem_volumes = torch.index_select(volumes, dim=0, index=sort_indices)

    xx1 = torch.index_select(x1, dim=0, index=sort_indices)
    xx2 = torch.index_select(x2, dim=0, index=sort_indices)
    yy1 = torch.index_select(y1, dim=0, index=sort_indices)
    yy2 = torch.index_select(y2, dim=0, index=sort_indices)
    zz1 = torch.index_select(z1, dim=0, index=sort_indices)
    zz2 = torch.index_select(z2, dim=0, index=sort_indices)

    # centroid_ = torch.index_select(centroid, dim=0, index=sort_indices)
    # pivot = centroid[[index]]

    xx1 = torch.max(xx1, x1[index])
    yy1 = torch.max(yy1, y1[index])
    zz1 = torch.max(zz1, z1[index])
    xx2 = torch.min(xx2, x2[index])
    yy2 = torch.min(yy2, y2[index])
    zz2 = torch.min(zz2, z2[index])

    l = torch.clamp(xx2 - xx1, min=0.0)
    w = torch.clamp(yy2 - yy1, min=0.0)
    h = torch.clamp(zz2 - zz1, min=0.0)

    inter = w * h * l

    union = (rem_volumes - inter) + volumes[index]

    IoU = inter / union

    return IoU


def cal_giou(volumes, x1, y1, z1, x2, y2, z2, sort_indices, index):
    rem_volumes = torch.index_select(volumes, dim=0, index=sort_indices)

    xx1 = torch.index_select(x1, dim=0, index=sort_indices)
    xx2 = torch.index_select(x2, dim=0, index=sort_indices)
    yy1 = torch.index_select(y1, dim=0, index=sort_indices)
    yy2 = torch.index_select(y2, dim=0, index=sort_indices)
    zz1 = torch.index_select(z1, dim=0, index=sort_indices)
    zz2 = torch.index_select(z2, dim=0, index=sort_indices)

    # centroid_ = torch.index_select(centroid, dim=0, index=sort_indices)
    # pivot = centroid[[index]]

    xx1 = torch.max(xx1, x1[index])
    yy1 = torch.max(yy1, y1[index])
    zz1 = torch.max(zz1, z1[index])
    xx2 = torch.min(xx2, x2[index])
    yy2 = torch.min(yy2, y2[index])
    zz2 = torch.min(zz2, z2[index])

    l = torch.clamp(xx2 - xx1, min=0.0)
    w = torch.clamp(yy2 - yy1, min=0.0)
    h = torch.clamp(zz2 - zz1, min=0.0)

    inter = w * h * l

    union = (rem_volumes - inter) + volumes[index]

    IoU = inter / union

    x_min_bound = torch.min(xx1, x1[index])
    y_min_bound = torch.min(yy1, y1[index])
    z_min_bound = torch.min(zz1, z1[index])
    x_max_bound = torch.max(xx2, x2[index])
    y_max_bound = torch.max(yy2, y2[index])
    z_max_bound = torch.max(zz2, z2[index])

    convex_area = (x_max_bound - x_min_bound) * (y_max_bound - y_min_bound) * (z_max_bound - z_min_bound)
    gIoU = IoU - (convex_area - union) / (convex_area + 1e-6)

    return IoU, gIoU


def batch_giou_cross(boxes1, boxes2):
    # boxes1: N, 6
    # boxes2: M, 6
    # out: N, M
    boxes1 = boxes1[:, None, :]
    boxes2 = boxes2[None, :, :]
    intersection = torch.prod(
        torch.clamp(
            (torch.min(boxes1[..., 3:], boxes2[..., 3:]) - torch.max(boxes1[..., :3], boxes2[..., :3])), min=0.0
        ),
        -1,
    )  # N

    boxes1_volumes = torch.prod(torch.clamp((boxes1[..., 3:] - boxes1[..., :3]), min=0.0), -1)
    boxes2_volumes = torch.prod(torch.clamp((boxes2[..., 3:] - boxes2[..., :3]), min=0.0), -1)

    union = boxes1_volumes + boxes2_volumes - intersection
    iou = intersection / (union + 1e-6)

    volumes_bound = torch.prod(
        torch.clamp(
            (torch.max(boxes1[..., 3:], boxes2[..., 3:]) - torch.min(boxes1[..., :3], boxes2[..., :3])), min=0.0
        ),
        -1,
    )  # N

    giou = iou - (volumes_bound - union) / (volumes_bound + 1e-6)

    return iou, giou


def batch_giou_corres(boxes1, boxes2):
    # boxes1: N, 6
    # boxes2: N, 6
    # out: N, M
    # boxes1 = boxes1[:, None, :]
    # boxes2 = boxes2[None, :, :]
    intersection = torch.prod(
        torch.clamp(
            (torch.min(boxes1[..., 3:], boxes2[..., 3:]) - torch.max(boxes1[..., :3], boxes2[..., :3])), min=0.0
        ),
        -1,
    )  # N

    boxes1_volumes = torch.prod(torch.clamp((boxes1[..., 3:] - boxes1[..., :3]), min=0.0), -1)
    boxes2_volumes = torch.prod(torch.clamp((boxes2[..., 3:] - boxes2[..., :3]), min=0.0), -1)

    union = boxes1_volumes + boxes2_volumes - intersection
    iou = intersection / (union + 1e-6)

    volumes_bound = torch.prod(
        torch.clamp(
            (torch.max(boxes1[..., 3:], boxes2[..., 3:]) - torch.min(boxes1[..., :3], boxes2[..., :3])), min=0.0
        ),
        -1,
    )  # N

    giou = iou - (volumes_bound - union) / (volumes_bound + 1e-6)

    return iou, giou


def superpoint_align(spp, proposals_pred):
    if len(proposals_pred.shape) == 2:
        n_inst, n_points = proposals_pred.shape[:2]
        spp_unique, spp_ids = torch.unique(spp, return_inverse=True)

        mean_spp_inst = torch_scatter.scatter(
            proposals_pred.float(), spp_ids.expand(n_inst, n_points), dim=-1, reduce="mean"
        )  # n_inst, n_spp
        spp_mask = mean_spp_inst >= 0.5

        refine_proposals_pred = torch.gather(spp_mask, dim=-1, index=spp_ids.expand(n_inst, n_points))

        return refine_proposals_pred

    if len(proposals_pred.shape) == 1:
        proposals_pred = proposals_pred.float()
        spp_unique, spp_ids = torch.unique(spp, return_inverse=True)

        mean_spp_inst = torch_scatter.scatter(proposals_pred.float(), spp_ids, dim=-1, reduce="mean")  # n_inst, n_spp
        spp_mask = mean_spp_inst >= 0.5

        refine_proposals_pred = spp_mask[spp_ids]
        refine_proposals_pred = refine_proposals_pred.bool()
        return refine_proposals_pred


def gen_boundary_gt(
    semantic_labels,
    instance_labels,
    coords_float,
    batch_idxs,
    radius=0.2,
    neighbor=48,
    ignore_label=255,
    label_shift=2,
):
    boundary = torch.zeros((instance_labels.shape[0]), dtype=torch.float, device=instance_labels.device)
    # condition = (instance_labels != ignore_label) & (semantic_labels >= label_shift)

    # condition = (semantic_labels >= 0)
    condition = torch.ones_like(semantic_labels).bool()
    object_idxs = torch.nonzero(condition).view(-1)

    if len(object_idxs) == 0:
        return boundary

    coords_float_ = coords_float[object_idxs]
    instance_ = instance_labels[object_idxs][:, None]

    batch_size = len(torch.unique(batch_idxs))
    batch_offsets = get_batch_offsets(batch_idxs, batch_size)
    batch_idxs_ = batch_idxs[object_idxs]
    batch_offsets_ = get_batch_offsets(batch_idxs_, batch_size)
    # batch_offsets_ = torch.tensor([0, coords_float_.shape[0]], dtype=torch.int).cuda()

    neighbor_inds = ballquery_batchflat(radius, neighbor, coords_float, coords_float_, batch_offsets, batch_offsets_)
    # neighbor_inds, _ = knnquery(neighbor, coords_float, coords_float_, batch_offsets, batch_offsets_)

    # print(neighbor_inds.shape, coords_float.shape)
    neighbor_inds = neighbor_inds.view(-1).long()
    neighbor_instance = instance_labels[neighbor_inds].view(coords_float_.shape[0], neighbor)

    diff_ins = torch.any((neighbor_instance != instance_), dim=-1)  # mpoints
    # boundary_labels[object_idxs.long()] = (diff_sem | diff_ins).long()

    # boundary_labels = boundary_labels.cpu()

    boundary[object_idxs.long()] = diff_ins.float()

    return boundary


def get_instance_info(coords_float, instance_labels, semantic_labels, label_shift=2):
    instance_pointnum = []
    instance_cls = []
    instance_box = []
    instance_num = int(instance_labels.max()) + 1

    centroid_offset_labels = (
        torch.ones((coords_float.shape[0], 3), dtype=coords_float.dtype, device=coords_float.device) * -100.0
    )
    corners_offset_labels = (
        torch.ones((coords_float.shape[0], 3 * 2), dtype=coords_float.dtype, device=coords_float.device) * -100.0
    )

    for i_ in range(instance_num):
        inst_idx_i = torch.nonzero(instance_labels == i_).view(-1)
        coords_float_i = coords_float[inst_idx_i]

        centroid = coords_float_i.mean(dim=0)

        min_xyz_i = coords_float_i.min(dim=0)[0]
        max_xyz_i = coords_float_i.max(dim=0)[0]

        centroid_offset_labels[inst_idx_i] = centroid - coords_float_i
        corners_offset_labels[inst_idx_i, 0:3] = min_xyz_i - coords_float_i
        corners_offset_labels[inst_idx_i, 3:6] = max_xyz_i - coords_float_i

        instance_box.append(torch.cat([min_xyz_i, max_xyz_i], axis=0))

        instance_pointnum.append(len(inst_idx_i))
        cls_idx = inst_idx_i[0]
        instance_cls.append(semantic_labels[cls_idx])

    instance_cls = torch.tensor(instance_cls, device=coords_float.device)
    instance_box = torch.stack(instance_box, dim=0)  # N, 6
    instance_cls[instance_cls != -100] = instance_cls[instance_cls != -100] - label_shift

    return instance_cls, instance_box, centroid_offset_labels, corners_offset_labels


def get_batch_offsets(batch_idxs, bs):
    batch_offsets = torch.zeros((bs + 1), dtype=torch.int, device=batch_idxs.device)
    for i in range(bs):
        batch_offsets[i + 1] = batch_offsets[i] + (batch_idxs == i).sum()
    assert batch_offsets[-1] == batch_idxs.shape[0]
    return batch_offsets


def random_downsample(batch_offsets, batch_size, n_subsample=30000):
    idxs_subsample = []
    for b in range(batch_size):
        start, end = batch_offsets[b], batch_offsets[b + 1]
        num_points_b = (end - start).cpu()

        if n_subsample == -1 or n_subsample >= num_points_b:
            new_inds = torch.arange(num_points_b, dtype=torch.long, device=batch_offsets.device) + start
        else:
            new_inds = (
                torch.tensor(
                    np.random.choice(num_points_b, n_subsample, replace=False),
                    dtype=torch.long,
                    device=batch_offsets.device,
                )
                + start
            )
        idxs_subsample.append(new_inds)
    idxs_subsample = torch.cat(idxs_subsample)  # N_subsample: batch x 20000

    return idxs_subsample


def get_cropped_instance_label(instance_label, valid_idxs=None):
    if valid_idxs is not None:
        instance_label = instance_label[valid_idxs]
    j = 0
    while j < instance_label.max():
        if (instance_label == j).sum() == 0:
            instance_label[instance_label == instance_label.max()] = j
        j += 1
    return instance_label


def custom_scatter_mean(input_feats, indices, dim=0, pool=True, output_type=None):
    if not pool:
        return input_feats

    original_type = input_feats.dtype
    with torch.cuda.amp.autocast(enabled=False):
        out_feats = torch_scatter.scatter_mean(input_feats.to(torch.float32), indices, dim=dim)

    if output_type is None:
        out_feats = out_feats.to(original_type)
    else:
        out_feats = out_feats.to(output_type)

    return out_feats


def superpoint_major_voting(
    labels, superpoint, n_classes, has_ignore_label=False, ignore_label=-100, return_full=True
):
    if has_ignore_label:
        labels = torch.where(labels >= 0, labels + 1, 0)
        n_classes += 1

    n_points = len(labels)
    # semantic_preds = voting_semantic_segmentation(semantic_preds, superpoint, num_semantic=self.classes)
    onehot_semantic_preds = F.one_hot(labels.long(), num_classes=n_classes)

    # breakpoint()
    count_onehot_semantic_preds = torch_scatter.scatter(
        onehot_semantic_preds, superpoint[:, None].expand(n_points, n_classes), dim=0, reduce="sum"
    )  # n_labels, n_spp

    label_spp = torch.argmax(count_onehot_semantic_preds, dim=1)  # n_spp
    score_label_spp = count_onehot_semantic_preds / torch.sum(count_onehot_semantic_preds, dim=1, keepdim=True)

    if has_ignore_label:
        label_spp = torch.where(label_spp >= 1, label_spp - 1, ignore_label)

    if return_full:
        refined_labels = label_spp[superpoint]
        refine_scores = score_label_spp[superpoint]

        return refined_labels, refine_scores

    return label_spp, score_label_spp


def get_subsample_gt(
    instance_labels, subsample_idxs, instance_cls, instance_box, subsample_batch_offsets, batch_size, ignore_label=-100
):
    subsample_inst_mask_arr = []

    subsample_instance_labels = instance_labels[subsample_idxs]
    for b in range(batch_size):
        start, end = subsample_batch_offsets[b], subsample_batch_offsets[b + 1]
        n_subsample = end - start

        instance_labels_b = subsample_instance_labels[start:end]

        unique_inst = torch.unique(instance_labels_b)
        unique_inst = unique_inst[unique_inst != ignore_label]

        unique_inst = unique_inst[instance_cls[unique_inst] >= 0]

        n_inst_gt = len(unique_inst)

        if n_inst_gt == 0:
            subsample_inst_mask_arr.append(None)
            continue

        instance_cls_b = instance_cls[unique_inst]
        instance_box_b = instance_box[unique_inst]

        subsample_mask_labels_b = torch.zeros(
            (n_inst_gt, n_subsample), device=subsample_instance_labels.device, dtype=torch.float
        )
        # breakpoint()
        for i, uni_id in enumerate(unique_inst):
            mask_ = instance_labels_b == uni_id
            subsample_mask_labels_b[i] = mask_.float()

        subsample_inst_mask_arr.append(
            {
                "mask": subsample_mask_labels_b,
                "cls": instance_cls_b,
                "box": instance_box_b,
            }
        )

    return subsample_inst_mask_arr


def get_spp_gt(
    instance_labels, spps, instance_cls, instance_box, batch_offsets, batch_size, ignore_label=-100, pool=True
):
    # original_type = instance_box.dtype
    spp_inst_mask_arr = []
    for b in range(batch_size):
        start, end = batch_offsets[b], batch_offsets[b + 1]

        instance_labels_b = instance_labels[start:end]
        spp_b = spps[start:end]
        spp_b_unique, spp_b = torch.unique(spp_b, return_inverse=True)

        # n_points = instance_labels_b.shape[0]
        n_spp = len(spp_b_unique)

        unique_inst = torch.unique(instance_labels_b)
        unique_inst = unique_inst[unique_inst != ignore_label]

        unique_inst = unique_inst[instance_cls[unique_inst] >= 0]

        n_inst_gt = len(unique_inst)

        if n_inst_gt == 0:
            spp_inst_mask_arr.append(None)
            continue

        instance_cls_b = instance_cls[unique_inst]
        instance_box_b = instance_box[unique_inst]

        spp_mask_labels_b = torch.zeros((n_inst_gt, n_spp), device=instance_labels.device, dtype=torch.float)

        for i, uni_id in enumerate(unique_inst):
            mask_ = instance_labels_b == uni_id
            spp_mask_ = custom_scatter_mean(mask_, spp_b, pool=pool, output_type=torch.float32)

            cond_ = spp_mask_ >= 0.5
            spp_mask_labels_b[i] = cond_.float()

        spp_inst_mask_arr.append(
            {
                "mask": spp_mask_labels_b,
                "cls": instance_cls_b,
                "box": instance_box_b,
            }
        )

    return spp_inst_mask_arr
