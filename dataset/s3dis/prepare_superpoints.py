import numpy as np
import torch

import glob


files = sorted(glob.glob("dataset/s3dis/learned_superpoint_graph_segmentations/*.npy"))

for file in files:
    chunks = file.split("/")[-1].split(".")
    area = chunks[0]
    room = chunks[1]

    spp = np.load(file, allow_pickle=True).item()["segments"]
    torch.save((spp), f"dataset/s3dis/superpoints/{area}_{room}.pth")
