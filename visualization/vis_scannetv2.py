import numpy as np
import pyviz3d.visualizer as viz
import torch
import os.path as osp
import argparse

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

### ScanNet Benchmark constants ###
VALID_CLASS_IDS_20 = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39)

CLASS_LABELS_20 = ('wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window',
                   'bookshelf', 'picture', 'counter', 'desk', 'curtain', 'refrigerator',
                   'shower curtain', 'toilet', 'sink', 'bathtub', 'otherfurniture')

COLOR_MAP = {
    0: (0., 0., 0.),
    1: (174., 199., 232.),
    2: (152., 223., 138.),
    3: (31., 119., 180.),
    4: (255., 187., 120.),
    5: (188., 189., 34.),
    6: (140., 86., 75.),
    7: (255., 152., 150.),
    8: (214., 39., 40.),
    9: (197., 176., 213.),
    10: (148., 103., 189.),
    11: (196., 156., 148.),
    12: (23., 190., 207.),
    14: (247., 182., 210.),
    15: (66., 188., 102.),
    16: (219., 219., 141.),
    17: (140., 57., 197.),
    18: (202., 185., 52.),
    19: (51., 176., 203.),
    20: (200., 54., 131.),
    21: (92., 193., 61.),
    22: (78., 71., 183.),
    23: (172., 114., 82.),
    24: (255., 127., 14.),
    25: (91., 163., 138.),
    26: (153., 98., 156.),
    27: (140., 153., 101.),
    28: (158., 218., 229.),
    29: (100., 125., 154.),
    30: (178., 127., 135.),
    32: (146., 111., 194.),
    33: (44., 160., 44.),
    34: (112., 128., 144.),
    35: (96., 207., 209.),
    36: (227., 119., 194.),
    37: (213., 92., 176.),
    38: (94., 106., 211.),
    39: (82., 84., 163.),
    40: (100., 85., 144.),
}

SEMANTIC_IDXS = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39])
SEMANTIC_IDX2NAME = {
    1: "wall",
    2: "floor",
    3: "cabinet",
    4: "bed",
    5: "chair",
    6: "sofa",
    7: "table",
    8: "door",
    9: "window",
    10: "bookshelf",
    11: "picture",
    12: "counter",
    14: "desk",
    16: "curtain",
    24: "refridgerator",
    28: "shower curtain",
    33: "toilet",
    34: "sink",
    36: "bathtub",
    39: "otherfurniture",
}

reverse_map = {i:inv_i for i, inv_i in enumerate(SEMANTIC_IDXS)}

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
    parser = argparse.ArgumentParser("ScanNetV2-Vis")

    parser.add_argument("--data_root", type=str, default='dataset/scannetv2')
    parser.add_argument("--scene_name", type=str, default='scene0011_00')
    parser.add_argument("--split", type=str, default='val')
    parser.add_argument("--prediction_path", help="path to the prediction results", 
                        default="results/isbnet_scannetv2_val")
    parser.add_argument("--point_size", type=float, default=15.0)
    parser.add_argument(
        "--task",
        help="all/input/sem_gt/inst_gt/superpoint/inst_pred",
        default="all",
    )
    args = parser.parse_args()

    # First, we set up a visualizer
    v = viz.Visualizer()

    if args.task == 'all':
        vis_tasks = ['input', 'sem_gt', 'inst_gt', 'superpoint' 'inst_pred']
    else:
        vis_tasks = [args.task]

    xyz, rgb, semantic_label, instance_label = torch.load(f'{args.data_root}/{args.split}/{args.scene_name}_inst_nostuff.pth')
    xyz = xyz.astype(np.float32)
    rgb = rgb.astype(np.float32)
    semantic_label = semantic_label.astype(np.int)
    instance_label = instance_label.astype(np.int)

    rgb = (rgb + 1.0) * 127.5

    mask_valid = (semantic_label != -100)
    xyz = xyz[mask_valid]
    rgb = rgb[mask_valid]
    semantic_label = semantic_label[mask_valid]
    instance_label = instance_label[mask_valid]
    
    if 'input' in vis_tasks:
        v.add_points(f'input', xyz, rgb, point_size=args.point_size)

    if 'sem_gt' in vis_tasks:
        sem_label_rgb = np.zeros_like(rgb)
        sem_unique = np.unique(semantic_label)
        for i, sem in enumerate(sem_unique):
            if sem == -100:
                continue
            remap_sem_id = reverse_map[sem]
            color_ = COLOR_MAP[remap_sem_id]
            sem_label_rgb[semantic_label == sem] = color_

        v.add_points(f'sem_gt', xyz, sem_label_rgb, point_size=args.point_size)

    if 'inst_gt' in vis_tasks:
        inst_unique = np.unique(instance_label)
        inst_label_rgb = np.zeros_like(rgb)
        for i, ind in enumerate(inst_unique):
            if ind == -100: continue
            inst_label_rgb[instance_label == ind] = COLOR_DETECTRON2[ind % 68]

        v.add_points(f'inst_gt', xyz, inst_label_rgb, point_size=args.point_size)

    if 'superpoint' in vis_tasks:
        spp = torch.load(f'{args.data_root}/superpoints/{args.scene_name}.pth')
        spp = spp[mask_valid]
        superpoint_rgb = np.zeros_like(rgb)
        unique_spp = np.unique(spp)

        for i, u_ in enumerate(unique_spp):
            superpoint_rgb[spp == u_] = COLOR_DETECTRON2[i % 68]

        v.add_points(f'superpoint', xyz, superpoint_rgb, point_size=args.point_size)

    if 'inst_pred' in vis_tasks:
        pred_rgb = get_pred_color(args.scene_name, mask_valid, args.prediction_path)
        v.add_points(f'inst_pred', xyz, pred_rgb, point_size=args.point_size)

    v.save('visualization/pyviz3d')


if __name__ == '__main__':
    main()