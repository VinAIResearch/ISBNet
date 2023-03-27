import numpy as np
from scipy import stats

import multiprocessing as mp
import warnings
from ..util import rle_decode


warnings.simplefilter(action="ignore", category=FutureWarning)


class S3DISEval(object):
    CLASSES = (
        "ceiling",
        "floor",
        "wall",
        "beam",
        "column",
        "window",
        "door",
        "chair",
        "table",
        "bookcase",
        "sofa",
        "board",
        "clutter",
    )

    def __init__(self, num_classes=13):
        self.num_classes = num_classes

        # Initialize...
        # precision & recall
        self.total_gt_ins = np.zeros(num_classes)
        self.at = 0.5
        self.tpsins = [[] for _ in range(num_classes)]
        self.fpsins = [[] for _ in range(num_classes)]
        # mucov and mwcov
        self.all_mean_cov = [[] for _ in range(num_classes)]
        self.all_mean_weighted_cov = [[] for _ in range(num_classes)]

    def single_process(self, preds, gt_sem, gt_ins):

        ignore_inds = (gt_ins < 0) | (gt_sem < 0)
        gt_sem[ignore_inds] = -1
        gt_ins[ignore_inds] = -1

        mean_cov_arr = [-1 for _ in range(self.num_classes)]
        mean_weighted_cov_arr = [-1 for _ in range(self.num_classes)]

        tp_arr = [-1 for _ in range(self.num_classes)]
        fp_arr = [-1 for _ in range(self.num_classes)]

        total_gt_ins = [0 for _ in range(self.num_classes)]

        pred_sem = np.zeros(gt_sem.shape[0], dtype=np.int)
        pred_ins = np.zeros(gt_sem.shape[0], dtype=np.int)

        pred_masks, pred_confs, pred_labels = [], [], []
        for pred in preds:
            pred_masks.append(rle_decode(pred["pred_mask"]))
            pred_confs.append(pred["conf"])
            pred_labels.append(pred["label_id"])

        pred_confs = np.array(pred_confs)
        sorted_inds = np.argsort(pred_confs)  # ascendent
        for i, s_id in enumerate(sorted_inds):
            point_ids = pred_masks[s_id] == 1
            pred_ins[point_ids] = i + 1
            pred_sem[point_ids] = pred_labels[s_id] - 1

        un = np.unique(gt_ins)
        pts_in_gt = [[] for itmp in range(self.num_classes)]
        for ig, g in enumerate(un):
            if g == -1:
                continue
            tmp = gt_ins == g
            sem_seg_i = int(stats.mode(gt_sem[tmp], axis=0)[0])
            pts_in_gt[sem_seg_i] += [tmp]

        # instance
        un = np.unique(pred_ins)
        pts_in_pred = [[] for _ in range(self.num_classes)]
        for ig, g in enumerate(un):  # each object in prediction
            if g == -1:
                continue
            tmp = pred_ins == g
            sem_seg_i = int(stats.mode(pred_sem[tmp])[0])
            pts_in_pred[sem_seg_i] += [tmp]

        # instance mucov & mwcov
        for i_sem in range(self.num_classes):
            sum_cov = 0
            mean_cov = 0
            mean_weighted_cov = 0
            num_gt_point = 0
            for ig, ins_gt in enumerate(pts_in_gt[i_sem]):
                ovmax = 0.0
                num_ins_gt_point = np.sum(ins_gt)
                num_gt_point += num_ins_gt_point
                for ip, ins_pred in enumerate(pts_in_pred[i_sem]):
                    union = ins_pred | ins_gt
                    intersect = ins_pred & ins_gt
                    iou = float(np.sum(intersect)) / np.sum(union)

                    if iou > ovmax:
                        ovmax = iou
                        # ipmax = ip

                sum_cov += ovmax
                mean_weighted_cov += ovmax * num_ins_gt_point

            if len(pts_in_gt[i_sem]) != 0:
                mean_cov = sum_cov / len(pts_in_gt[i_sem])
                mean_cov_arr[i_sem] = mean_cov

                mean_weighted_cov /= num_gt_point
                mean_weighted_cov_arr[i_sem] = mean_weighted_cov

        # instance precision & recall
        for i_sem in range(self.num_classes):
            tp = [0.0] * len(pts_in_pred[i_sem])
            fp = [0.0] * len(pts_in_pred[i_sem])
            gtflag = np.zeros(len(pts_in_gt[i_sem]))
            total_gt_ins[i_sem] = len(pts_in_gt[i_sem])

            for ip, ins_pred in enumerate(pts_in_pred[i_sem]):
                ovmax = -1.0

                for ig, ins_gt in enumerate(pts_in_gt[i_sem]):

                    union = ins_pred | ins_gt
                    intersect = ins_pred & ins_gt
                    iou = float(np.sum(intersect)) / np.sum(union)

                    if iou > ovmax:
                        ovmax = iou
                        igmax = ig

                if ovmax >= self.at:
                    tp[ip] = 1  # true
                    gtflag[igmax] = 1
                else:
                    fp[ip] = 1  # false positive

            tp_arr[i_sem] = tp
            fp_arr[i_sem] = fp

        return mean_cov_arr, mean_weighted_cov_arr, tp_arr, fp_arr, total_gt_ins

    def evaluate(self, pred_list, gt_sem_list, gt_ins_list):

        pool = mp.Pool(processes=16)
        results = pool.starmap(self.single_process, zip(pred_list, gt_sem_list, gt_ins_list))
        pool.close()
        pool.join()

        for i, result in enumerate(results):
            mean_cov_arr, mean_weighted_cov_arr, tp_arr, fp_arr, total_gt_ins = result
            for i_sem in range(self.num_classes):
                if mean_cov_arr[i_sem] != -1:
                    self.all_mean_cov[i_sem].append(mean_cov_arr[i_sem])

                if mean_weighted_cov_arr[i_sem] != -1:
                    self.all_mean_weighted_cov[i_sem].append(mean_weighted_cov_arr[i_sem])

                self.tpsins[i_sem] += tp_arr[i_sem]
                self.fpsins[i_sem] += fp_arr[i_sem]

                self.total_gt_ins[i_sem] += total_gt_ins[i_sem]

        MUCov = np.zeros(self.num_classes)
        MWCov = np.zeros(self.num_classes)
        for i_sem in range(self.num_classes):
            MUCov[i_sem] = np.mean(self.all_mean_cov[i_sem])
            MWCov[i_sem] = np.mean(self.all_mean_weighted_cov[i_sem])

        precision = np.zeros(self.num_classes)
        recall = np.zeros(self.num_classes)
        for i_sem in range(self.num_classes):
            tp = np.asarray(self.tpsins[i_sem]).astype(np.float)
            fp = np.asarray(self.fpsins[i_sem]).astype(np.float)
            tp = np.sum(tp)
            fp = np.sum(fp)
            rec = np.minimum(1.0, tp / self.total_gt_ins[i_sem])
            prec = tp / (tp + fp)

            precision[i_sem] = prec
            recall[i_sem] = rec

        sep = ""
        col1 = ":"
        lineLen = 48

        print()
        print("#" * lineLen)
        line = ""
        line += "{:<15}".format("what") + sep + col1
        line += "{:>8}".format("MUCov") + sep
        line += "{:>8}".format("MWCov") + sep
        line += "{:>8}".format("Prec") + sep
        line += "{:>8}".format("Rec") + sep

        print(line)
        print("#" * lineLen)

        for (li, label_name) in enumerate(self.CLASSES):
            line = "{:<15}".format(label_name) + sep + col1
            line += sep + "{:>8.3f}".format(MUCov[li]) + sep
            line += sep + "{:>8.3f}".format(MWCov[li]) + sep
            line += sep + "{:>8.3f}".format(precision[li]) + sep
            line += sep + "{:>8.3f}".format(recall[li]) + sep
            print(line)
        print("#" * lineLen)
        print()

        mMUCov, mMWCov, mPrec, mRec = np.nanmean(MUCov), np.nanmean(MWCov), np.nanmean(precision), np.nanmean(recall)

        print(f"mMUCov: {mMUCov}")
        print(f"mMWCov: {mMWCov}")
        print(f"mPrecision: {mPrec}")
        print(f"mRecall: {mRec}")
        print()

        return mMUCov, mMWCov, mPrec, mRec
