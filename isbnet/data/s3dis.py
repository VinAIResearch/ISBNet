import numpy as np
import torch

import os.path as osp
from glob import glob
from ..ops import voxelization_idx
from .custom import CustomDataset

base_class_idx_6 = [0, 1, 2, 3, 4, 8]
novel_class_idx_6 = [5, 6, 7, 9, 10, 11, 12]

base_class_idx_8 = [ 0, 1, 2, 3, 4, 6, 8, 11 ]
novel_class_idx_8 = [ 5, 7, 9, 10, 12]

base_class_idx = {
    8: base_class_idx_8,
    6: base_class_idx_6,
}

novel_class_idx = {
    8: novel_class_idx_8,
    6: novel_class_idx_6,
}


def build_class_mapper(class_idx, ignore_idx=-100, squeeze_label=True):
    remapper = np.ones(256, dtype=np.int64) * ignore_idx
    for (i, x) in enumerate(class_idx):
        if squeeze_label:
            remapper[x] = i
        else:
            remapper[x] = x
    return remapper
class S3DISDataset(CustomDataset):

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
    BENCHMARK_SEMANTIC_IDXS = [i for i in range(15)]  # NOTE DUMMY values just for save results

    def __init__(self, data_root, prefix, suffix, voxel_cfg=None, training=True, repeat=1, logger=None, base=8):
        super().__init__(data_root, prefix, suffix, voxel_cfg=voxel_cfg, training=training, repeat=repeat, logger=logger)
        self.CLASS_MAPPER = build_class_mapper(base_class_idx[base])

        self.CLASS_NAME_BASE = []
        for cls in base_class_idx[base]:
            # print(cls, len(ScanNetDataset.CLASSES))
            self.CLASS_NAME_BASE.append(S3DISDataset.CLASSES[cls])

    

    def get_filenames(self):
        if isinstance(self.prefix, str):
            self.prefix = [self.prefix]
        filenames_all = []
        for p in self.prefix:
            filenames = glob(osp.join(self.data_root, "preprocess", p + "*" + self.suffix))
            assert len(filenames) > 0, f"Empty {p}"
            filenames_all.extend(filenames)

        filenames_all = sorted(filenames_all * self.repeat)
        return filenames_all

    def load(self, filename):
        scan_id = osp.basename(filename).replace(self.suffix, "")

        xyz, rgb, semantic_label, instance_label = torch.load(filename)

        spp_filename = osp.join(self.data_root, "superpoints", scan_id + ".pth")
        spp = torch.load(spp_filename)

        semantic_label[semantic_label!=-100] = self.CLASS_MAPPER[semantic_label[semantic_label!=-100].astype(np.int64)]
        instance_label[semantic_label == -100] = -100

        N = xyz.shape[0]
        if self.training:
            inds = np.random.choice(N, int(N * 0.25), replace=False)
            xyz = xyz[inds]
            rgb = rgb[inds]
            spp = spp[inds]

            spp = np.unique(spp, return_inverse=True)[1]

            semantic_label = semantic_label[inds]
            instance_label = self.getCroppedInstLabel(instance_label, inds)
        elif N > 5000000:  # NOTE Avoid OOM
            print(f"Downsample scene {scan_id} with original num_points: {N}")
            inds = np.arange(N)[::4]

            xyz = xyz[inds]
            rgb = rgb[inds]
            spp = spp[inds]

            spp = np.unique(spp, return_inverse=True)[1]

            semantic_label = semantic_label[inds]
            instance_label = self.getCroppedInstLabel(instance_label, inds)

        return xyz, rgb, semantic_label, instance_label, spp

    def crop(self, xyz, step=64):
        return super().crop(xyz, step=step)

    def transform_test(self, xyz, rgb, semantic_label, instance_label, spp):
        # devide into 4 piecies
        inds = np.arange(xyz.shape[0])
        piece_1 = inds[::4]
        piece_2 = inds[1::4]
        piece_3 = inds[2::4]
        piece_4 = inds[3::4]
        xyz_aug = self.dataAugment(xyz, False, False, False)

        xyz_list = []
        xyz_middle_list = []
        rgb_list = []
        semantic_label_list = []
        instance_label_list = []
        spp_list = []

        for batch, piece in enumerate([piece_1, piece_2, piece_3, piece_4]):
            xyz_middle = xyz_aug[piece]
            xyz = xyz_middle * self.voxel_cfg.scale
            xyz -= xyz.min(0)
            xyz_list.append(np.concatenate([np.full((xyz.shape[0], 1), batch), xyz], 1))
            xyz_middle_list.append(xyz_middle)
            rgb_list.append(rgb[piece])
            semantic_label_list.append(semantic_label[piece])
            instance_label_list.append(instance_label[piece])
            spp_list.append(spp[piece])

        xyz = np.concatenate(xyz_list, 0)
        xyz_middle = np.concatenate(xyz_middle_list, 0)
        rgb = np.concatenate(rgb_list, 0)

        semantic_label = np.concatenate(semantic_label_list, 0)
        instance_label = np.concatenate(instance_label_list, 0)
        spp = np.concatenate(spp_list, 0)

        valid_idxs = np.ones(xyz.shape[0], dtype=bool)
        instance_label = self.getCroppedInstLabel(instance_label, valid_idxs)  # TODO remove this
        return xyz, xyz_middle, rgb, semantic_label, instance_label, spp

    def collate_fn(self, batch):
        if self.training:
            return super().collate_fn(batch)

        # assume 1 scan only
        (
            scan_id,
            coord,
            coord_float,
            feat,
            semantic_label,
            instance_label,
            spp,
            inst_num,
        ) = batch[0]

        scan_ids = [scan_id]
        coords = coord.long()
        batch_idxs = torch.zeros_like(coord[:, 0].int())
        coords_float = coord_float.float()
        feats = feat.float()
        semantic_labels = semantic_label.long()
        instance_labels = instance_label.long()
        spps = spp.long()

        instance_batch_offsets = torch.tensor([0, inst_num], dtype=torch.long)

        spatial_shape = np.clip((coords.max(0)[0][1:] + 1).numpy(), self.voxel_cfg.spatial_shape[0], None)
        voxel_coords, v2p_map, p2v_map = voxelization_idx(coords, 4)
        return {
            "scan_ids": scan_ids,
            "batch_idxs": batch_idxs,
            "voxel_coords": voxel_coords,
            "p2v_map": p2v_map,
            "v2p_map": v2p_map,
            "coords_float": coords_float,
            "feats": feats,
            "semantic_labels": semantic_labels,
            "instance_labels": instance_labels,
            "spps": spps,
            "instance_batch_offsets": instance_batch_offsets,
            "spatial_shape": spatial_shape,
            "batch_size": 1,
        }
