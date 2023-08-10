import numpy as np
import torch

import argparse
import os
import os.path as osp
import pyviz3d.visualizer as viz


COLOR_DETECTRON2 = (
    np.array(
        [
            0.000,
            0.447,
            0.741,
            0.850,
            0.325,
            0.098,
            0.929,
            0.694,
            0.125,
            0.494,
            0.184,
            0.556,
            0.466,
            0.674,
            0.188,
            0.301,
            0.745,
            0.933,
            0.635,
            0.078,
            0.184,
            # 0.300, 0.300, 0.300,
            0.600,
            0.600,
            0.600,
            1.000,
            0.000,
            0.000,
            1.000,
            0.500,
            0.000,
            0.749,
            0.749,
            0.000,
            0.000,
            1.000,
            0.000,
            0.000,
            0.000,
            1.000,
            0.667,
            0.000,
            1.000,
            0.333,
            0.333,
            0.000,
            0.333,
            0.667,
            0.000,
            0.333,
            1.000,
            0.000,
            0.667,
            0.333,
            0.000,
            0.667,
            0.667,
            0.000,
            0.667,
            1.000,
            0.000,
            1.000,
            0.333,
            0.000,
            1.000,
            0.667,
            0.000,
            1.000,
            1.000,
            0.000,
            0.000,
            0.333,
            0.500,
            0.000,
            0.667,
            0.500,
            0.000,
            1.000,
            0.500,
            0.333,
            0.000,
            0.500,
            0.333,
            0.333,
            0.500,
            0.333,
            0.667,
            0.500,
            0.333,
            1.000,
            0.500,
            0.667,
            0.000,
            0.500,
            0.667,
            0.333,
            0.500,
            0.667,
            0.667,
            0.500,
            0.667,
            1.000,
            0.500,
            1.000,
            0.000,
            0.500,
            1.000,
            0.333,
            0.500,
            1.000,
            0.667,
            0.500,
            1.000,
            1.000,
            0.500,
            0.000,
            0.333,
            1.000,
            0.000,
            0.667,
            1.000,
            0.000,
            1.000,
            1.000,
            0.333,
            0.000,
            1.000,
            0.333,
            0.333,
            1.000,
            0.333,
            0.667,
            1.000,
            0.333,
            1.000,
            1.000,
            0.667,
            0.000,
            1.000,
            0.667,
            0.333,
            1.000,
            0.667,
            0.667,
            1.000,
            0.667,
            1.000,
            1.000,
            1.000,
            0.000,
            1.000,
            1.000,
            0.333,
            1.000,
            1.000,
            0.667,
            1.000,
            # 0.333, 0.000, 0.000,
            0.500,
            0.000,
            0.000,
            0.667,
            0.000,
            0.000,
            0.833,
            0.000,
            0.000,
            1.000,
            0.000,
            0.000,
            0.000,
            0.167,
            0.000,
            # 0.000, 0.333, 0.000,
            0.000,
            0.500,
            0.000,
            0.000,
            0.667,
            0.000,
            0.000,
            0.833,
            0.000,
            0.000,
            1.000,
            0.000,
            0.000,
            0.000,
            0.167,
            # 0.000, 0.000, 0.333,
            0.000,
            0.000,
            0.500,
            0.000,
            0.000,
            0.667,
            0.000,
            0.000,
            0.833,
            0.000,
            0.000,
            1.000,
            # 0.000, 0.000, 0.000,
            0.143,
            0.143,
            0.143,
            0.857,
            0.857,
            0.857,
            # 1.000, 1.000, 1.000
        ]
    )
    .astype(np.float32)
    .reshape(-1, 3)
    * 255
)

CLASS_LABELS_STPLS3D = (
    "background",
    "building",
    "low vegetation",
    "med. vegetation",
    "high vegetation",
    "vehicle",
    "truck",
    "aircraft",
    "militaryVehicle",
    "bike",
    "motorcycle",
    "light pole",
    "street sign",
    "clutter",
    "fence",
)

COLOR_MAP = {
    0: (0.0, 0.0, 0.0),
    1: (174.0, 199.0, 232.0),
    2: (152.0, 223.0, 138.0),
    3: (31.0, 119.0, 180.0),
    4: (255.0, 187.0, 120.0),
    5: (188.0, 189.0, 34.0),
    6: (140.0, 86.0, 75.0),
    7: (255.0, 152.0, 150.0),
    8: (214.0, 39.0, 40.0),
    9: (197.0, 176.0, 213.0),
    10: (148.0, 103.0, 189.0),
    11: (196.0, 156.0, 148.0),
    12: (23.0, 190.0, 207.0),
    14: (247.0, 182.0, 210.0),
    15: (66.0, 188.0, 102.0),
    16: (219.0, 219.0, 141.0),
    17: (140.0, 57.0, 197.0),
    18: (202.0, 185.0, 52.0),
    19: (51.0, 176.0, 203.0),
    20: (200.0, 54.0, 131.0),
    21: (92.0, 193.0, 61.0),
    22: (78.0, 71.0, 183.0),
    23: (172.0, 114.0, 82.0),
    24: (255.0, 127.0, 14.0),
    25: (91.0, 163.0, 138.0),
    26: (153.0, 98.0, 156.0),
    27: (140.0, 153.0, 101.0),
    28: (158.0, 218.0, 229.0),
    29: (100.0, 125.0, 154.0),
    30: (178.0, 127.0, 135.0),
    32: (146.0, 111.0, 194.0),
    33: (44.0, 160.0, 44.0),
    34: (112.0, 128.0, 144.0),
    35: (96.0, 207.0, 209.0),
    36: (227.0, 119.0, 194.0),
    37: (213.0, 92.0, 176.0),
    38: (94.0, 106.0, 211.0),
    39: (82.0, 84.0, 163.0),
    40: (100.0, 85.0, 144.0),
}

SEMANTIC_IDX2NAME = {k: v for k, v in enumerate(CLASS_LABELS_STPLS3D)}


def get_pred_color(scene_name, mask_valid, dir):
    instance_file = osp.join(dir, "pred_instance", scene_name + ".txt")

    f = open(instance_file, "r")
    masks = f.readlines()
    masks = [mask.rstrip().split() for mask in masks]
    inst_label_pred_rgb = np.zeros((mask_valid.sum(), 3))  # np.ones(rgb.shape) * 255 #

    # FIXME
    ins_num = len(masks)
    ins_pointnum = np.zeros(ins_num)
    inst_label = -100 * np.ones(mask_valid.sum()).astype(np.int)

    # sort score such that high score has high priority for visualization
    scores = np.array([float(x[-1]) for x in masks])
    sort_inds = np.argsort(scores)[::-1]
    for i_ in range(len(masks) - 1, -1, -1):
        i = sort_inds[i_]
        # mask_path = os.path.join(opt.prediction_path, "pred_instance", masks[i][0])
        mask_path = osp.join(dir, "pred_instance", masks[i][0])
        assert osp.isfile(mask_path), mask_path
        if float(masks[i][2]) < 0.1:
            continue

        mask = np.loadtxt(mask_path).astype(np.int)
        mask = mask[mask_valid]

        cls = SEMANTIC_IDX2NAME[int(masks[i][1])]

        print("{} {}: {} pointnum: {}".format(i, masks[i], cls, mask.sum()))
        ins_pointnum[i] = mask.sum()
        inst_label[mask == 1] = i

    sort_idx = np.argsort(ins_pointnum)[::-1]
    for _sort_id in range(ins_num):
        color = COLOR_DETECTRON2[_sort_id % len(COLOR_DETECTRON2)]
        inst_label_pred_rgb[inst_label == sort_idx[_sort_id]] = color

    return inst_label_pred_rgb


def main():
    parser = argparse.ArgumentParser("STPLS3D-Vis")

    parser.add_argument("--data_root", type=str, default="dataset/stpls3d")
    parser.add_argument("--scene_name", type=str, default="5_points_GTv3_00")
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument(
        "--prediction_path", help="path to the prediction results", default="results/isbnet_stpls3d_val"
    )
    parser.add_argument("--point_size", type=float, default=15.0)
    parser.add_argument(
        "--task",
        help="all/input/sem_gt/inst_gt/superpoint/inst_pred",
        default="all",
    )
    args = parser.parse_args()

    # First, we set up a visualizer
    v = viz.Visualizer()

    if args.task == "all":
        vis_tasks = ["input", "sem_gt", "inst_gt", "superpoint" "inst_pred"]
    else:
        vis_tasks = [args.task]

    xyz, rgb, semantic_label, instance_label = torch.load(
        f"{args.data_root}/{args.split}/{args.scene_name}_inst_nostuff.pth"
    )
    xyz = xyz.astype(np.float32)
    rgb = rgb.astype(np.float32)
    semantic_label = semantic_label.astype(np.int)
    instance_label = instance_label.astype(np.int)

    rgb = (rgb + 1.0) * 127.5

    mask_valid = semantic_label != -100
    xyz = xyz[mask_valid]
    rgb = rgb[mask_valid]
    semantic_label = semantic_label[mask_valid]
    instance_label = instance_label[mask_valid]

    if "input" in vis_tasks:
        v.add_points(f"input", xyz, rgb, point_size=args.point_size)

    if "sem_gt" in vis_tasks:
        sem_label_rgb = np.zeros_like(rgb)
        sem_unique = np.unique(semantic_label)
        for i, sem in enumerate(sem_unique):
            if sem == -100:
                continue
            remap_sem_id = sem + 1
            color_ = COLOR_MAP[remap_sem_id]
            sem_label_rgb[semantic_label == sem] = color_

        v.add_points(f"sem_gt", xyz, sem_label_rgb, point_size=args.point_size)

    if "inst_gt" in vis_tasks:
        inst_unique = np.unique(instance_label)
        inst_label_rgb = np.zeros_like(rgb)
        for i, ind in enumerate(inst_unique):
            if ind == -100:
                continue
            inst_label_rgb[instance_label == ind] = COLOR_DETECTRON2[ind % 68]

        v.add_points(f"inst_gt", xyz, inst_label_rgb, point_size=args.point_size)

    if "superpoint" in vis_tasks:
        # NOTE currently STPLS3D does not have superpoint
        spp = np.arange((mask_valid.shape[0]), dtype=np.long)
        spp = spp[mask_valid]
        superpoint_rgb = np.zeros_like(rgb)
        unique_spp = np.unique(spp)

        for i, u_ in enumerate(unique_spp):
            superpoint_rgb[spp == u_] = COLOR_DETECTRON2[i % 68]

        v.add_points(f"superpoint", xyz, superpoint_rgb, point_size=args.point_size)

    if "inst_pred" in vis_tasks:
        pred_rgb = get_pred_color(args.scene_name, mask_valid, args.prediction_path)
        v.add_points(f"inst_pred", xyz, pred_rgb, point_size=args.point_size)

    v.save("visualization/pyviz3d")


if __name__ == "__main__":
    main()
