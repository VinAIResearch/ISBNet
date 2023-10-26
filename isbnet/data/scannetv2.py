import numpy as np
import torch

import os.path as osp
from .custom import CustomDataset

base_class_idx_8 = [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 13 ]
novel_class_idx_8 = [ 9, 10, 11, 12, 14, 15, 16, 17, 18 , 19]

base_class_idx_10 = [ 0, 1, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16 ]
novel_class_idx_10 = [ 3, 4, 6, 9, 10, 17, 18 , 19]

base_class_idx_13 = [ 0, 1, 2, 3, 4, 6, 7, 8, 10, 11, 13, 14, 15, 17, 18 ]
novel_class_idx_13 =  [ 5, 9, 12, 16, 19]


def build_class_mapper(ignore_idx=-100, squeeze_label=True):
    remapper = np.ones(256, dtype=np.int64) * ignore_idx
    for (i, x) in enumerate(base_class_idx_10):
        if squeeze_label:
            remapper[x] = i
        else:
            remapper[x] = x
    return remapper

class ScanNetDataset(CustomDataset):

    CLASSES = (
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
        "refrigerator",
        "shower curtain",
        "toilet",
        "sink",
        "bathtub",
        "otherfurniture",
    )
    BENCHMARK_SEMANTIC_IDXS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]
    CLASS_MAPPER = build_class_mapper()


    @staticmethod
    def build_class_mapper(class_idx, ignore_idx, squeeze_label=True):
        remapper = np.ones(256, dtype=np.int64) * ignore_idx
        for (i, x) in enumerate(class_idx):
            if squeeze_label:
                remapper[x] = i
            else:
                remapper[x] = x
        return remapper
    
    def load(self, filename):
        scan_id = osp.basename(filename).replace(self.suffix, "")

        if self.prefix == "test":
            xyz, rgb = torch.load(filename)
            semantic_label = np.zeros(xyz.shape[0], dtype=np.long)
            instance_label = np.zeros(xyz.shape[0], dtype=np.long)
        else:
            xyz, rgb, semantic_label, instance_label = torch.load(filename)

        semantic_label = ScanNetDataset.CLASS_MAPPER[semantic_label.astype(np.int64)]

        spp_filename = osp.join(self.data_root, "superpoints", scan_id + ".pth")
        spp = torch.load(spp_filename)

        return xyz, rgb, semantic_label, instance_label, spp
