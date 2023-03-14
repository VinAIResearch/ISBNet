import numpy as np


def evaluate_semantic_acc(pred_list, gt_list, ignore_label=-100, logger=None):
    gt = np.concatenate(gt_list, axis=0)
    pred = np.concatenate(pred_list, axis=0)
    assert gt.shape == pred.shape
    correct = (gt[gt != ignore_label] == pred[gt != ignore_label]).sum()
    whole = (gt != ignore_label).sum()
    acc = correct.astype(float) / whole * 100
    logger.info(f"Acc: {acc:.1f}")
    return acc


def evaluate_semantic_miou(pred_list, gt_list, ignore_label=-100, logger=None):
    gt = np.concatenate(gt_list, axis=0)
    pred = np.concatenate(pred_list, axis=0)
    pos_inds = gt != ignore_label
    gt = gt[pos_inds]
    pred = pred[pos_inds]
    assert gt.shape == pred.shape
    iou_list = []
    for _index in np.unique(gt):
        if _index != ignore_label:
            intersection = ((gt == _index) & (pred == _index)).sum()
            union = ((gt == _index) | (pred == _index)).sum()
            iou = intersection.astype(float) / union * 100
            iou_list.append(iou)
    miou = np.mean(iou_list)
    logger.info("Class-wise mIoU: " + " ".join(f"{x:.1f}" for x in iou_list))
    logger.info(f"mIoU: {miou:.1f}")
    return miou


def evaluate_offset_mae(pred_list, gt_list, gt_instance_list, ignore_label=-100, logger=None):
    gt = np.concatenate(gt_list, axis=0)
    pred = np.concatenate(pred_list, axis=0)
    gt_instance = np.concatenate(gt_instance_list, axis=0)
    pos_inds = gt_instance != ignore_label
    gt = gt[pos_inds]
    pred = pred[pos_inds]
    mae = np.abs(gt - pred).sum() / pos_inds.sum()
    logger.info(f"Offset MAE: {mae:.3f}")
    return mae


class PointWiseEval(object):
    def __init__(self, num_classes=20, ignore_label=-100) -> None:
        self.ignore_label = ignore_label
        self.pos_inds_arr = []
        self.mae_arr = []
        self.mae_vertices_arr = []

        self.pos_inds_inst_arr = []
        self.sem_acc_arr = []

        self.conf_metric = np.ndarray((num_classes, num_classes), dtype=np.int32)
        self.conf_metric.fill(0)
        self.num_classes = num_classes

        self.debug_acc = []
        self.debug_acc_num_pos = []

    def update(self, pred_sem, pred_offset, pred_vertices_offset, gt_sem, gt_offset, gt_vertices_offset, gt_instance):
        pos_inds = gt_sem != self.ignore_label

        pred_sem = pred_sem[pos_inds]
        gt_sem = gt_sem[pos_inds]

        correct = (gt_sem == pred_sem).sum()
        self.sem_acc_arr.append(correct)
        self.pos_inds_arr.append(pos_inds.sum())

        # hack for bin counting 2 arrays together
        x = pred_sem + self.num_classes * gt_sem
        bincount_2d = np.bincount(x.astype(np.int32), minlength=self.num_classes**2)
        assert bincount_2d.size == self.num_classes**2
        conf = bincount_2d.reshape((self.num_classes, self.num_classes))
        self.conf_metric += conf
        # print(conf)

        pos_inds_inst = gt_instance != self.ignore_label

        gt_offset = gt_offset[pos_inds_inst]
        pred_offset = pred_offset[pos_inds_inst]
        self.pos_inds_inst_arr.append(pos_inds_inst.sum())
        self.mae_arr.append(np.abs(gt_offset - pred_offset).sum())

        gt_vertices_offset = gt_vertices_offset[pos_inds_inst]
        pred_vertices_offset = pred_vertices_offset[pos_inds_inst]
        self.mae_vertices_arr.append(np.abs(gt_vertices_offset - pred_vertices_offset).sum())

    def update_debug_acc(self, acc, num_pos):
        self.debug_acc.append(acc)
        self.debug_acc_num_pos.append(num_pos)

    def get_eval(self, logger):
        # mIoU
        true_positive = np.diag(self.conf_metric)
        false_positive = np.sum(self.conf_metric, 0) - true_positive
        false_negative = np.sum(self.conf_metric, 1) - true_positive

        # Just in case we get a division by 0, ignore/hide the error
        with np.errstate(divide="ignore", invalid="ignore"):
            iou = true_positive / (true_positive + false_positive + false_negative)
            iou = iou * 100
        miou = np.nanmean(iou)
        logger.info("Class-wise mIoU: " + " ".join(f"{iou[i]:.1f}" for i in range(iou.shape[0])))
        logger.info(f"mIoU: {miou:.1f}")

        # semantic accuracy
        acc = np.sum(np.array(self.sem_acc_arr)) / np.sum(np.array(self.pos_inds_arr)) * 100
        logger.info(f"Acc: {acc:.1f}")

        # offset mae
        mae = np.sum(np.array(self.mae_arr)) / np.sum(np.array(self.pos_inds_inst_arr))
        logger.info(f"Offset MAE: {mae:.3f}")

        mae_vertices = np.sum(np.array(self.mae_vertices_arr)) / np.sum(np.array(self.pos_inds_inst_arr))
        logger.info(f"Offset vertices MAE: {mae_vertices:.3f}")

        if len(self.debug_acc) > 0:
            debug_acc = np.sum(np.array(self.debug_acc)) / np.sum(np.array(self.debug_acc_num_pos))
            logger.info("Mean accuracy of classification: {:.3f}".format(debug_acc))

        return miou, acc, mae
