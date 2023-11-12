import numpy as np
import torch

import argparse
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

CLASS_LABELS_S3DIS = (
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
    13: (23.0, 190.0, 207.0),
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

SEMANTIC_IDX2NAME = {
    0: "unannotated",
    1: "ceiling",
    2: "floor",
    3: "wall",
    4: "beam",
    5: "column",
    6: "window",
    7: "door",
    8: "chair",
    9: "table",
    10: "bookcase",
    11: "sofa",
    12: "board",
    13: "clutter",
}


SCANNET_COLOR_MAP_200 = {
    0: (0.0, 0.0, 0.0),
    1: (174.0, 199.0, 232.0),
    2: (188.0, 189.0, 34.0),
    3: (152.0, 223.0, 138.0),
    4: (255.0, 152.0, 150.0),
    5: (214.0, 39.0, 40.0),
    6: (91.0, 135.0, 229.0),
    7: (31.0, 119.0, 180.0),
    8: (229.0, 91.0, 104.0),
    9: (247.0, 182.0, 210.0),
    10: (91.0, 229.0, 110.0),
    11: (255.0, 187.0, 120.0),
    13: (141.0, 91.0, 229.0),
    14: (112.0, 128.0, 144.0),
    15: (196.0, 156.0, 148.0),
    16: (197.0, 176.0, 213.0),
    17: (44.0, 160.0, 44.0),
    18: (148.0, 103.0, 189.0),
    19: (229.0, 91.0, 223.0),
    21: (219.0, 219.0, 141.0),
    22: (192.0, 229.0, 91.0),
    23: (88.0, 218.0, 137.0),
    24: (58.0, 98.0, 137.0),
    26: (177.0, 82.0, 239.0),
    27: (255.0, 127.0, 14.0),
    28: (237.0, 204.0, 37.0),
    29: (41.0, 206.0, 32.0),
    31: (62.0, 143.0, 148.0),
    32: (34.0, 14.0, 130.0),
    33: (143.0, 45.0, 115.0),
    34: (137.0, 63.0, 14.0),
    35: (23.0, 190.0, 207.0),
    36: (16.0, 212.0, 139.0),
    38: (90.0, 119.0, 201.0),
    39: (125.0, 30.0, 141.0),
    40: (150.0, 53.0, 56.0),
    41: (186.0, 197.0, 62.0),
    42: (227.0, 119.0, 194.0),
    44: (38.0, 100.0, 128.0),
    45: (120.0, 31.0, 243.0),
    46: (154.0, 59.0, 103.0),
    47: (169.0, 137.0, 78.0),
    48: (143.0, 245.0, 111.0),
    49: (37.0, 230.0, 205.0),
    50: (14.0, 16.0, 155.0),
    51: (196.0, 51.0, 182.0),
    52: (237.0, 80.0, 38.0),
    54: (138.0, 175.0, 62.0),
    55: (158.0, 218.0, 229.0),
    56: (38.0, 96.0, 167.0),
    57: (190.0, 77.0, 246.0),
    58: (208.0, 49.0, 84.0),
    59: (208.0, 193.0, 72.0),
    62: (55.0, 220.0, 57.0),
    63: (10.0, 125.0, 140.0),
    64: (76.0, 38.0, 202.0),
    65: (191.0, 28.0, 135.0),
    66: (211.0, 120.0, 42.0),
    67: (118.0, 174.0, 76.0),
    68: (17.0, 242.0, 171.0),
    69: (20.0, 65.0, 247.0),
    70: (208.0, 61.0, 222.0),
    71: (162.0, 62.0, 60.0),
    72: (210.0, 235.0, 62.0),
    73: (45.0, 152.0, 72.0),
    74: (35.0, 107.0, 149.0),
    75: (160.0, 89.0, 237.0),
    76: (227.0, 56.0, 125.0),
    77: (169.0, 143.0, 81.0),
    78: (42.0, 143.0, 20.0),
    79: (25.0, 160.0, 151.0),
    80: (82.0, 75.0, 227.0),
    82: (253.0, 59.0, 222.0),
    84: (240.0, 130.0, 89.0),
    86: (123.0, 172.0, 47.0),
    87: (71.0, 194.0, 133.0),
    88: (24.0, 94.0, 205.0),
    89: (134.0, 16.0, 179.0),
    90: (159.0, 32.0, 52.0),
    93: (213.0, 208.0, 88.0),
    95: (64.0, 158.0, 70.0),
    96: (18.0, 163.0, 194.0),
    97: (65.0, 29.0, 153.0),
    98: (177.0, 10.0, 109.0),
    99: (152.0, 83.0, 7.0),
    100: (83.0, 175.0, 30.0),
    101: (18.0, 199.0, 153.0),
    102: (61.0, 81.0, 208.0),
    103: (213.0, 85.0, 216.0),
    104: (170.0, 53.0, 42.0),
    105: (161.0, 192.0, 38.0),
    106: (23.0, 241.0, 91.0),
    107: (12.0, 103.0, 170.0),
    110: (151.0, 41.0, 245.0),
    112: (133.0, 51.0, 80.0),
    115: (184.0, 162.0, 91.0),
    116: (50.0, 138.0, 38.0),
    118: (31.0, 237.0, 236.0),
    120: (39.0, 19.0, 208.0),
    121: (223.0, 27.0, 180.0),
    122: (254.0, 141.0, 85.0),
    125: (97.0, 144.0, 39.0),
    128: (106.0, 231.0, 176.0),
    130: (12.0, 61.0, 162.0),
    131: (124.0, 66.0, 140.0),
    132: (137.0, 66.0, 73.0),
    134: (250.0, 253.0, 26.0),
    136: (55.0, 191.0, 73.0),
    138: (60.0, 126.0, 146.0),
    139: (153.0, 108.0, 234.0),
    140: (184.0, 58.0, 125.0),
    141: (135.0, 84.0, 14.0),
    145: (139.0, 248.0, 91.0),
    148: (53.0, 200.0, 172.0),
    154: (63.0, 69.0, 134.0),
    155: (190.0, 75.0, 186.0),
    156: (127.0, 63.0, 52.0),
    157: (141.0, 182.0, 25.0),
    159: (56.0, 144.0, 89.0),
    161: (64.0, 160.0, 250.0),
    163: (182.0, 86.0, 245.0),
    165: (139.0, 18.0, 53.0),
    166: (134.0, 120.0, 54.0),
    168: (49.0, 165.0, 42.0),
    169: (51.0, 128.0, 133.0),
    170: (44.0, 21.0, 163.0),
    177: (232.0, 93.0, 193.0),
    180: (176.0, 102.0, 54.0),
    185: (116.0, 217.0, 17.0),
    188: (54.0, 209.0, 150.0),
    191: (60.0, 99.0, 204.0),
    193: (129.0, 43.0, 144.0),
    195: (252.0, 100.0, 106.0),
    202: (187.0, 196.0, 73.0),
    208: (13.0, 158.0, 40.0),
    213: (52.0, 122.0, 152.0),
    214: (128.0, 76.0, 202.0),
    221: (187.0, 50.0, 115.0),
    229: (180.0, 141.0, 71.0),
    230: (77.0, 208.0, 35.0),
    232: (72.0, 183.0, 168.0),
    233: (97.0, 99.0, 203.0),
    242: (172.0, 22.0, 158.0),
    250: (155.0, 64.0, 40.0),
    261: (118.0, 159.0, 30.0),
    264: (69.0, 252.0, 148.0),
    276: (45.0, 103.0, 173.0),
    283: (111.0, 38.0, 149.0),
    286: (184.0, 9.0, 49.0),
    300: (188.0, 174.0, 67.0),
    304: (53.0, 206.0, 53.0),
    312: (97.0, 235.0, 252.0),
    323: (66.0, 32.0, 182.0),
    325: (236.0, 114.0, 195.0),
    331: (241.0, 154.0, 83.0),
    342: (133.0, 240.0, 52.0),
    356: (16.0, 205.0, 144.0),
    370: (75.0, 101.0, 198.0),
    392: (237.0, 95.0, 251.0),
    395: (191.0, 52.0, 49.0),
    399: (227.0, 254.0, 54.0),
    408: (49.0, 206.0, 87.0),
    417: (48.0, 113.0, 150.0),
    488: (125.0, 73.0, 182.0),
    540: (229.0, 32.0, 114.0),
    562: (158.0, 119.0, 28.0),
    570: (60.0, 205.0, 27.0),
    572: (18.0, 215.0, 201.0),
    581: (79.0, 76.0, 153.0),
    609: (134.0, 13.0, 116.0),
    748: (192.0, 97.0, 63.0),
    776: (108.0, 163.0, 18.0),
    1156: (95.0, 220.0, 156.0),
    1163: (98.0, 141.0, 208.0),
    1164: (144.0, 19.0, 193.0),
    1165: (166.0, 36.0, 57.0),
    1166: (212.0, 202.0, 34.0),
    1167: (23.0, 206.0, 34.0),
    1168: (91.0, 211.0, 236.0),
    1169: (79.0, 55.0, 137.0),
    1170: (182.0, 19.0, 117.0),
    1171: (134.0, 76.0, 14.0),
    1172: (87.0, 185.0, 28.0),
    1173: (82.0, 224.0, 187.0),
    1174: (92.0, 110.0, 214.0),
    1175: (168.0, 80.0, 171.0),
    1176: (197.0, 63.0, 51.0),
    1178: (175.0, 199.0, 77.0),
    1179: (62.0, 180.0, 98.0),
    1180: (8.0, 91.0, 150.0),
    1181: (77.0, 15.0, 130.0),
    1182: (154.0, 65.0, 96.0),
    1183: (197.0, 152.0, 11.0),
    1184: (59.0, 155.0, 45.0),
    1185: (12.0, 147.0, 145.0),
    1186: (54.0, 35.0, 219.0),
    1187: (210.0, 73.0, 181.0),
    1188: (221.0, 124.0, 77.0),
    1189: (149.0, 214.0, 66.0),
    1190: (72.0, 185.0, 134.0),
    1191: (42.0, 94.0, 198.0),
}

COLOR_200 = [c for _, c in SCANNET_COLOR_MAP_200.items()]


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

        cls = SEMANTIC_IDX2NAME[int(masks[i][1]) - 1]

        print("{} {}: {} pointnum: {}".format(i, masks[i], cls, mask.sum()))
        ins_pointnum[i] = mask.sum()
        inst_label[mask == 1] = i

    sort_idx = np.argsort(ins_pointnum)[::-1]
    for _sort_id in range(ins_num):
        color = COLOR_DETECTRON2[_sort_id % len(COLOR_DETECTRON2)]
        inst_label_pred_rgb[inst_label == sort_idx[_sort_id]] = color

    return inst_label_pred_rgb

def get_pred_color2(scene_name, mask_valid, dir):
    instance_file = osp.join("/home/tdngo/Workspace/3dis_ws/Open3DInstanceSegmentation/data/s3dis/version1/fuse3d", scene_name + ".pth")

    # instance_file = osp.join("/home/tdngo/Workspace/3dis_ws/Open3DInstanceSegmentation/data/s3dis/version1/mergedsam3d", scene_name + ".pth")
    # instance_file = osp.join("/home/tdngo/Workspace/3dis_ws/ISBNet/results/s3dis_area4_cls_agnostic", scene_name + ".pth")
    
    
    data = torch.load(instance_file)
    # breakpoint()
    # masks = torch.from_numpy(data['ins']).long()
    masks = torch.stack([m for m in data['ins']], dim=0)
    # if len(masks.shape == 1):

    # mask_torch = masks + 1
    # mask_torch = torch.nn.functional.one_hot(mask_torch)[:, 2:].permute(1,0)
    # # breakpoint()
    # masks = mask_torch.numpy()

    print(masks.shape)
    try:
        scores = data['conf']
    except:
        scores = np.ones((masks.shape[0]))
    # f = open(instance_file, "r")
    # masks = f.readlines()
    # masks = [mask.rstrip().split() for mask in masks]
    inst_label_pred_rgb = np.zeros((mask_valid.sum(), 3))  # np.ones(rgb.shape) * 255 #


    # masks = masks.T

    # inds = np.arange(masks.shape[0])

    # p1 = inds[::4]
    # p2 = inds[1::4]
    # p3 = inds[2::4]
    # p4 = inds[3::4]
    # ps = [p1, p2, p3, p4]
    # # breakpoint()
    # len_arr = [len(p) for p in ps]
    # p12 = len_arr[0] + len_arr[1]
    # p123 = p12 + len_arr[2]

    # masks_split = [masks[:len_arr[0]], masks[len_arr[0]:p12], masks[p12:p123], masks[p123:, :]]
    # # np.split(xyz2, [len(p) for p in ps], axis=0)
    # # breakpoint()
    # masks3 = np.zeros_like(masks)
    # for i, p in enumerate(ps):
    #     masks3[p] = masks_split[i]

    # masks = masks3.T

    # breakpoint()
    # FIXME
    ins_num = len(masks)
    ins_pointnum = np.zeros(ins_num)
    inst_label = -100 * np.ones(mask_valid.sum()).astype(np.int)

    # sort score such that high score has high priority for visualization
    # scores = np.array([float(x[-1]) for x in masks])
    sort_inds = np.argsort(scores)[::-1]
    for i_ in range(len(masks) - 1, -1, -1):
        i = sort_inds[i_]
        # mask_path = os.path.join(opt.prediction_path, "pred_instance", masks[i][0])
        # mask_path = osp.join(dir, "pred_instance", masks[i][0])
        # assert osp.isfile(mask_path), mask_path
        # if float(masks[i][2]) < 0.1:
        #     continue

        # mask = np.loadtxt(mask_path).astype(np.int)
        mask = masks[i]
        mask = mask[mask_valid]

        # cls = SEMANTIC_IDX2NAME[int(masks[i][1]) - 1]

        # print("{} {}: {} pointnum: {}".format(i, masks[i], cls, mask.sum()))
        ins_pointnum[i] = mask.sum()
        inst_label[mask == 1] = i

    sort_idx = np.argsort(ins_pointnum)[::-1]
    for _sort_id in range(ins_num):
        color = COLOR_DETECTRON2[_sort_id % len(COLOR_DETECTRON2)]
        inst_label_pred_rgb[inst_label == sort_idx[_sort_id]] = color

    return inst_label_pred_rgb


def main():
    parser = argparse.ArgumentParser("S3DIS-Vis")

    parser.add_argument("--data_root", type=str, default="dataset/s3dis")
    parser.add_argument("--scene_name", type=str, default="Area_4_office_20")
    parser.add_argument("--split", type=str, default="preprocess_notalign")
    parser.add_argument(
        "--prediction_path", help="path to the prediction results", default="results/s3dis_area5_hardfilter_spp"
    )
    parser.add_argument("--point_size", type=float, default=30.0)
    parser.add_argument(
        "--task",
        help="all/input/sem_gt/inst_gt/superpoint/inst_pred",
        default="all",
    )
    args = parser.parse_args()

    # First, we set up a visualizer
    v = viz.Visualizer()

    if args.task == "all":
        # vis_tasks = ["input", "sem_gt", "inst_gt", "superpoint", "inst_pred"]
        vis_tasks = ["input", "sem_gt", "inst_gt", "superpoint"]
    else:
        vis_tasks = [args.task]

    xyz, rgb, semantic_label, instance_label = torch.load(
        f"{args.data_root}/{args.split}/{args.scene_name}_inst_nostuff.pth"
    )
    # _, rgb, semantic_label, instance_label = torch.load(
    #     f"{args.data_root}/preprocess/{args.scene_name}_inst_nostuff.pth"
    # )
    xyz = xyz.astype(np.float32)
    rgb = rgb.astype(np.float32)
    semantic_label = semantic_label.astype(np.int)
    instance_label = instance_label.astype(np.int)

    # xyz, rgb, semantic_label, instance_label, _, fps_ind = torch.load(
    #     osp.join('/home/tdngo/Workspace/3dis_ws/ISBNet/dataset/s3dis/preprocess_notalign_subsample', args.scene_name + '_inst_nostuff.pth')
    # )

    # xyz = xyz.astype(np.float32)
    # rgb = rgb.astype(np.float32)
    # semantic_label = semantic_label.astype(np.int)
    # instance_label = instance_label.astype(np.int)


    # NOTE split 4 to match with the order of model's prediction
    # inds = np.arange(xyz.shape[0])
    # xyz_list = []
    # rgb_list = []
    # semantic_label_list = []
    # instance_label_list = []
    # for i in range(4):
    #     piece = inds[i::4]
    #     semantic_label_list.append(semantic_label[piece])
    #     instance_label_list.append(instance_label[piece])
    #     xyz_list.append(xyz[piece])
    #     rgb_list.append(rgb[piece])

    # xyz = np.concatenate(xyz_list, 0)
    # rgb = np.concatenate(rgb_list, 0)
    # semantic_label = np.concatenate(semantic_label_list, 0)
    # instance_label = np.concatenate(instance_label_list, 0)

    
    # breakpoint()

    xyz = xyz - np.min(xyz, axis=0)
    rgb = (rgb + 1) * 255.0

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
        spp = torch.load(f"{args.data_root}/superpoints_notalign/{args.scene_name}.pth")
        # spp_list = []
        # for i in range(4):
        #     piece = inds[i::4]
        #     spp_list.append(spp[piece])
        # spp = np.concatenate(spp_list, axis=0)

        spp = spp[mask_valid]
        superpoint_rgb = np.zeros_like(rgb)
        unique_spp = np.unique(spp)

        for i, u_ in enumerate(unique_spp):
            c = np.random.randint(0, 400)
            superpoint_rgb[spp == u_] = COLOR_200[(c*10) % len(COLOR_200)]

        v.add_points(f"superpoint", xyz, superpoint_rgb, point_size=args.point_size)

    # print(vis_tasks)
    if "inst_pred" in vis_tasks:
        pred_rgb = get_pred_color2(args.scene_name, mask_valid, args.prediction_path)
        v.add_points(f"inst_pred", xyz, pred_rgb, point_size=args.point_size)

    v.save("visualization/pyviz3d")


if __name__ == "__main__":
    main()