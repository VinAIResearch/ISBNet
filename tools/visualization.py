import numpy as np
import torch

import argparse
import math
import open3d as o3d
import os
from operator import itemgetter


# yapf:disable
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
# yapf:enable

SEMANTIC_IDXS = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39])
SEMANTIC_NAMES = np.array(
    [
        "wall",
        "floor",
        "cabinet",
        "bed",
        "chair",
        "sofa",
        "table",
        "door",
        "window",
        "bookshelf",
        "picture",
        "counter",
        "desk",
        "curtain",
        "refridgerator",
        "shower curtain",
        "toilet",
        "sink",
        "bathtub",
        "otherfurniture",
    ]
)
CLASS_COLOR = {
    "unannotated": [0, 0, 0],
    "floor": [143, 223, 142],
    "wall": [171, 198, 230],
    "cabinet": [0, 120, 177],
    "bed": [255, 188, 126],
    "chair": [189, 189, 57],
    "sofa": [144, 86, 76],
    "table": [255, 152, 153],
    "door": [222, 40, 47],
    "window": [197, 176, 212],
    "bookshelf": [150, 103, 185],
    "picture": [200, 156, 149],
    "counter": [0, 190, 206],
    "desk": [252, 183, 210],
    "curtain": [219, 219, 146],
    "refridgerator": [255, 127, 43],
    "bathtub": [234, 119, 192],
    "shower curtain": [150, 218, 228],
    "toilet": [0, 160, 55],
    "sink": [110, 128, 143],
    "otherfurniture": [80, 83, 160],
}
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


def get_coords_color(opt):
    if opt.dataset == "s3dis":
        assert opt.data_split in [
            "Area_1",
            "Area_2",
            "Area_3",
            "Area_4",
            "Area_5",
            "Area_6",
        ], "data_split for s3dis should be one of [Area_1, Area_2, Area_3, Area_4, Area_5, Area_6]"
        input_file = os.path.join("dataset", opt.dataset, "preprocess", opt.room_name + "_inst_nostuff.pth")
        assert os.path.isfile(input_file), "File not exist - {}.".format(input_file)
        xyz, rgb, label, inst_label, _, _ = torch.load(input_file)
        # update variable to match scannet format
        opt.data_split = os.path.join("val", opt.data_split)
    else:
        input_file = os.path.join("dataset", opt.dataset, opt.data_split, opt.room_name + "_inst_nostuff.pth")
        assert os.path.isfile(input_file), "File not exist - {}.".format(input_file)
        if opt.data_split == "test":
            xyz, rgb = torch.load(input_file)
        else:
            xyz, rgb, label, inst_label = torch.load(input_file)

    rgb = (rgb + 1) * 127.5

    m = np.eye(3)
    theta = 0.35 * math.pi
    m = np.matmul(m, [[math.cos(theta), math.sin(theta), 0], [-math.sin(theta), math.cos(theta), 0], [0, 0, 1]])

    xyz = np.matmul(xyz, m)

    boxes = []

    if opt.task == "semantic_gt":
        assert opt.data_split != "test"
        label = label.astype(np.int)
        label_rgb = np.zeros(rgb.shape)
        label_rgb[label >= 0] = np.array(itemgetter(*SEMANTIC_NAMES[label[label >= 0]])(CLASS_COLOR))
        rgb = label_rgb

    elif opt.task == "semantic_pred":
        assert opt.data_split != "train"
        semantic_file = os.path.join(opt.prediction_path, "semantic_pred", opt.room_name + ".npy")
        assert os.path.isfile(semantic_file), "No semantic result - {}.".format(semantic_file)
        label_pred = np.load(semantic_file).astype(np.int)  # 0~19
        label_pred_rgb = np.array(itemgetter(*SEMANTIC_NAMES[label_pred])(CLASS_COLOR))
        rgb = label_pred_rgb

    elif opt.task == "object_pred":
        assert opt.data_split != "train"
        obj_file = os.path.join(opt.prediction_path, "object_conditions", opt.room_name + ".npy")
        assert os.path.isfile(obj_file), "No semantic result - {}.".format(obj_file)
        obj_pred = np.load(obj_file).astype(np.int)  # 0~19

        # label_pred_rgb = np.array(itemgetter(*SEMANTIC_NAMES[label_pred])(CLASS_COLOR))
        # rgb = label_pred_rgb
        print("bug")
        obj_pred_rgb = np.zeros_like(rgb)
        obj_pred_rgb[obj_pred == 1] = CLASS_COLOR["floor"]
        obj_pred_rgb[obj_pred == 0] = CLASS_COLOR["unannotated"]
        rgb = obj_pred_rgb

    elif opt.task == "offset_semantic_pred":
        assert opt.data_split != "train"
        semantic_file = os.path.join(opt.prediction_path, "semantic_pred", opt.room_name + ".npy")
        assert os.path.isfile(semantic_file), "No semantic result - {}.".format(semantic_file)
        label_pred = np.load(semantic_file).astype(np.int)  # 0~19
        label_pred_rgb = np.array(itemgetter(*SEMANTIC_NAMES[label_pred])(CLASS_COLOR))
        rgb = label_pred_rgb

        offset_file = os.path.join(opt.prediction_path, "offset_pred", opt.room_name + ".npy")
        assert os.path.isfile(offset_file), "No offset result - {}.".format(offset_file)
        offset_coords = np.load(offset_file)

        # xyz = offset_coords[:, :3] + offset_coords[:, 3:]
        # xyz[label_pred > 1] += offset_coords[label_pred > 1]
        xyz += offset_coords

    elif opt.task == "superpoint":
        superpoint_file = os.path.join("dataset", opt.dataset, "superpoints", opt.room_name + ".pth")
        spp = torch.load(superpoint_file)
        spp_label_rgb = np.zeros(rgb.shape)
        unique_spp = torch.unique(spp)

        for i, uni_idx in enumerate(unique_spp):
            if uni_idx == -100:
                continue
            spp_label_rgb[spp == uni_idx] = COLOR_DETECTRON2[i % len(COLOR_DETECTRON2)]

        rgb = spp_label_rgb

    # same color order according to instance pointnum
    elif opt.task == "instance_gt":
        assert opt.data_split != "test"
        inst_label = inst_label.astype(np.int)

        # unique_labels = np.unique(inst_label)
        # unique_labels = [ins for ins in unique_labels if ins != -100]

        inst_label_rgb = np.zeros(rgb.shape)
        ins_num = inst_label.max() + 1
        ins_pointnum = np.zeros(ins_num)
        for _ins_id in range(ins_num):
            ins_pointnum[_ins_id] = (inst_label == _ins_id).sum()
        sort_idx = np.argsort(ins_pointnum)[::-1]

        count_ins = 0
        for _sort_id in range(ins_num):
            inds = np.nonzero(inst_label == sort_idx[_sort_id])[0]
            # print(inds)
            if label[inds[0]] == -100:
                continue

            # inst_label_rgb[inds] = [255,0,0]

            inst_label_rgb[inds] = COLOR_DETECTRON2[_sort_id % len(COLOR_DETECTRON2)]

            print("Instance {}: pointnum: {}".format(_sort_id, ins_pointnum[_sort_id]))
            count_ins += 1

        print("Instance number: {}".format(count_ins))
        rgb = inst_label_rgb

        # xyz = xyz[inst_label_rgb==[255,0,0]]
        # rgb = rgb[inst_label_rgb==[255,0,0]]

    # same color order according to instance pointnum
    elif opt.task == "instance_pred":
        assert opt.data_split != "train"
        instance_file = os.path.join(opt.prediction_path, "pred_instance", opt.room_name + ".txt")
        assert os.path.isfile(instance_file), "No instance result - {}.".format(instance_file)
        f = open(instance_file, "r")
        masks = f.readlines()
        masks = [mask.rstrip().split() for mask in masks]
        inst_label_pred_rgb = np.zeros(rgb.shape)  # np.ones(rgb.shape) * 255 #

        # FIXME
        # masks = masks[24:30]
        ins_num = len(masks)
        ins_pointnum = np.zeros(ins_num)
        box_pred = np.zeros((ins_num, 6))
        inst_label = -100 * np.ones(rgb.shape[0]).astype(np.int)

        # sort score such that high score has high priority for visualization
        scores = np.array([float(x[-1]) for x in masks])
        sort_inds = np.argsort(scores)[::-1]
        for i_ in range(len(masks) - 1, -1, -1):
            i = sort_inds[i_]
            mask_path = os.path.join(opt.prediction_path, "pred_instance", masks[i][0])
            assert os.path.isfile(mask_path), mask_path
            if float(masks[i][2]) < 0.09:
                continue
            mask = np.loadtxt(mask_path).astype(np.int)
            if opt.dataset == "scannet":
                print("{} {}: {} pointnum: {}".format(i, masks[i], SEMANTIC_IDX2NAME[int(masks[i][1])], mask.sum()))
            else:
                print("{} {}: pointnum: {}".format(i, masks[i], mask.sum()))
            ins_pointnum[i] = mask.sum()
            inst_label[mask == 1] = i

            box = masks[i][3:9]
            box_pred[i] = np.array(box)

        sort_idx = np.argsort(ins_pointnum)[::-1]
        for _sort_id in range(ins_num):
            color = COLOR_DETECTRON2[_sort_id % len(COLOR_DETECTRON2)]
            inst_label_pred_rgb[inst_label == sort_idx[_sort_id]] = color

            box = o3d.geometry.AxisAlignedBoundingBox(
                min_bound=box_pred[sort_idx[_sort_id], 0:3], max_bound=box_pred[sort_idx[_sort_id], 3:6]
            )
            box.color = color / 255.0
            boxes.append(box)
        rgb = inst_label_pred_rgb

        # # NOTE plot box of context points
        # offset_file = os.path.join(opt.prediction_path, 'offset_vertices_pred',
        #                            opt.room_name + '.npy')
        # assert os.path.isfile(offset_file), 'No offset result - {}.'.format(offset_file)
        # offset_coords = np.load(offset_file)

        # semantic_file = os.path.join(opt.prediction_path, "semantic_pred", opt.room_name + ".npy")
        # assert os.path.isfile(semantic_file), "No semantic result - {}.".format(semantic_file)
        # label_pred = np.load(semantic_file).astype(np.int)  # 0~19

        # for _sort_id in range(ins_num):
        #     color = COLOR_DETECTRON2[_sort_id % len(COLOR_DETECTRON2)]
        #     inst_label_rgb[inst_label == sort_idx[_sort_id]] = color

        #     # min_coord = np.mean(offset_coords_min[inst_label == sort_idx[_sort_id]], 0)
        #     # max_coord = np.mean(offset_coords_max[inst_label == sort_idx[_sort_id]], 0)
        #     # box = o3d.geometry.AxisAlignedBoundingBox(min_bound=min_coord, max_bound=max_coord)
        #     # box.color = color/255.
        #     # boxes.append(box)
        #     indices = np.nonzero(inst_label == sort_idx[_sort_id])[0][::500]
        #     for ind in indices:
        #         box = o3d.geometry.AxisAlignedBoundingBox(min_bound=offset_coords_min[ind], max_bound=offset_coords_max[ind])
        #         box.color = color/255.
        #         boxes.append(box)

    elif opt.task == "offset_vertices_pred":
        offset_file = os.path.join(opt.prediction_path, "offset_vertices_pred", opt.room_name + ".npy")
        assert os.path.isfile(offset_file), "No offset result - {}.".format(offset_file)
        offset_coords = np.load(offset_file)

        semantic_file = os.path.join(opt.prediction_path, "semantic_pred", opt.room_name + ".npy")
        assert os.path.isfile(semantic_file), "No semantic result - {}.".format(semantic_file)
        label_pred = np.load(semantic_file).astype(np.int)  # 0~19

        # print(offset_coords.shape)

        offset_coords_min = offset_coords[:, :3] + xyz
        offset_coords_max = offset_coords[:, 3:] + xyz

        assert opt.data_split != "test"
        inst_label = inst_label.astype(np.int)
        print("Instance number: {}".format(inst_label.max() + 1))
        inst_label_rgb = np.zeros(rgb.shape)
        ins_num = inst_label.max() + 1
        ins_pointnum = np.zeros(ins_num)
        for _ins_id in range(ins_num):
            ins_pointnum[_ins_id] = (inst_label == _ins_id).sum()
        sort_idx = np.argsort(ins_pointnum)[::-1]
        for _sort_id in range(ins_num):
            color = COLOR_DETECTRON2[_sort_id % len(COLOR_DETECTRON2)]
            inst_label_rgb[inst_label == sort_idx[_sort_id]] = color

            # min_coord = np.mean(offset_coords_min[inst_label == sort_idx[_sort_id]], 0)
            # max_coord = np.mean(offset_coords_max[inst_label == sort_idx[_sort_id]], 0)
            # box = o3d.geometry.AxisAlignedBoundingBox(min_bound=min_coord, max_bound=max_coord)
            # box.color = color/255.
            # boxes.append(box)
            indices = np.nonzero(inst_label == sort_idx[_sort_id])[0][::500]
            for ind in indices:
                box = o3d.geometry.AxisAlignedBoundingBox(
                    min_bound=offset_coords_min[ind], max_bound=offset_coords_max[ind]
                )
                box.color = color / 255.0
                boxes.append(box)

        print("total boxes:", len(boxes))
        rgb = inst_label_rgb

    if opt.data_split != "test":
        sem_valid = label != -100
        xyz = xyz[sem_valid]
        rgb = rgb[sem_valid]

    return xyz, rgb, boxes


def write_ply(verts, colors, indices, output_file):
    if colors is None:
        colors = np.zeros_like(verts)
    if indices is None:
        indices = []

    file = open(output_file, "w")
    file.write("ply \n")
    file.write("format ascii 1.0\n")
    file.write("element vertex {:d}\n".format(len(verts)))
    file.write("property float x\n")
    file.write("property float y\n")
    file.write("property float z\n")
    file.write("property uchar red\n")
    file.write("property uchar green\n")
    file.write("property uchar blue\n")
    file.write("element face {:d}\n".format(len(indices)))
    file.write("property list uchar uint vertex_indices\n")
    file.write("end_header\n")
    for vert, color in zip(verts, colors):
        file.write(
            "{:f} {:f} {:f} {:d} {:d} {:d}\n".format(
                vert[0], vert[1], vert[2], int(color[0] * 255), int(color[1] * 255), int(color[2] * 255)
            )
        )
    for ind in indices:
        file.write("3 {:d} {:d} {:d}\n".format(ind[0], ind[1], ind[2]))
    file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", choices=["scannetv2", "s3dis"], help="dataset for visualization", default="scannetv2"
    )
    parser.add_argument(
        "--prediction_path",
        help="path to the prediction results",
        default="./results/pointnext_val",
    )
    parser.add_argument("--data_split", help="train/val/test for scannet or Area_ID for s3dis", default="val")
    parser.add_argument("--room_name", help="room_name", default="scene0011_00")
    parser.add_argument(
        "--task",
        help="input/semantic_gt/semantic_pred/offset_semantic_pred/instance_gt/instance_pred",
        default="instance_pred",
    )
    parser.add_argument("--out", default="", help="output point cloud file in FILE.ply format")
    opt = parser.parse_args()

    xyz, rgb, boxes = get_coords_color(opt)
    points = xyz[:, :3]
    colors = rgb / 255

    if opt.out != "":
        assert ".ply" in opt.out, "output cloud file should be in FILE.ply format"
        write_ply(points, colors, None, opt.out)
    else:

        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(points)
        pc.colors = o3d.utility.Vector3dVector(colors)

        vis = o3d.visualization.Visualizer()
        vis.create_window()

        if len(boxes) > 0:
            for box in boxes:
                # box.color = (1,0,0)
                vis.add_geometry(box)

        vis.add_geometry(pc)
        vis.get_render_option().point_size = 1.5
        vis.run()
        vis.destroy_window()
