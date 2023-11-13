import numpy as np
import torch

import os.path as osp
from .custom import CustomDataset


class ReplicaDataset(CustomDataset):

    CLASSES = ["basket", "bed", "bench", "bin", "blanket", "blinds", "book", "bottle", "box", "bowl", "camera", "cabinet", "candle", "chair", "clock",
    "cloth", "comforter", "cushion", "desk", "desk-organizer", "door", "indoor-plant", "lamp", "monitor", "nightstand",
    "panel", "picture", "pillar", "pillow", "pipe", "plant-stand", "plate", "pot", "sculpture", "shelf", "sofa", "stool", "switch", "table",
    "tablet", "tissue-paper", "tv-screen", "tv-stand", "vase", "vent", "wall-plug", "window", "rug"]

    BENCHMARK_SEMANTIC_IDXS = [i+1 for i in range(len(CLASSES))] 

    def load(self, filename):
        scan_id = osp.basename(filename).replace(self.suffix, "")

        xyz, rgb, semantic_label, instance_label = torch.load(filename)
        xyz = xyz - np.min(xyz, axis=0)
        
        spp = np.arange(xyz.shape[0], dtype=np.long)

        return xyz, rgb, semantic_label, instance_label, spp
