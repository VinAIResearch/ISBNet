import numpy as np
import torch

from .custom import CustomDataset


class STPLS3DDataset(CustomDataset):

    CLASSES = (
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

    BENCHMARK_SEMANTIC_IDXS = [i for i in range(20)]  # NOTE DUMMY values just for save results

    def getInstanceInfo(self, xyz, instance_label, semantic_label):
        ret = super().getInstanceInfo(xyz, instance_label, semantic_label)
        instance_num, instance_pointnum, instance_cls, pt_offset_label = ret
        # ignore instance of class 0 and reorder class id
        instance_cls = [x - 1 if x != -100 else x for x in instance_cls]
        return instance_num, instance_pointnum, instance_cls, pt_offset_label

    def load(self, filename):
        if self.prefix == "test":
            xyz, rgb = torch.load(filename)
            semantic_label = np.zeros(xyz.shape[0], dtype=np.long)
            instance_label = np.zeros(xyz.shape[0], dtype=np.long)
        else:
            xyz, rgb, semantic_label, instance_label = torch.load(filename)

        # NOTE currently stpls3d does not have spps, we will add later
        # spp = np.zeros(xyz.shape[0], dtype=np.long)
        spp = np.arange(xyz.shape[0], dtype=np.long)

        return xyz, rgb, semantic_label, instance_label, spp
