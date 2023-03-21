import numpy as np
import torch

import open3d as o3d
import os
import segmentator


def get_superpoint(mesh_file):
    mesh = o3d.io.read_triangle_mesh(mesh_file)
    vertices = torch.from_numpy(np.array(mesh.vertices).astype(np.float32))
    faces = torch.from_numpy(np.array(mesh.triangles).astype(np.int64))
    superpoint = segmentator.segment_mesh(vertices, faces)
    # self.superpoints[scene] = superpoint
    # print(mesh_file, superpoint.shape)
    return superpoint


if __name__ == "__main__":
    os.makedirs("dataset/scannetv2/superpoints", exist_ok=True)
    scans_trainval = os.listdir("dataset/scannetv2/scans/*")
    for scan in scans_trainval:
        ply_file = os.path.join("dataset/scannetv2/scans", scan, f"{scan}_vh_clean_2.ply")
        spp = get_superpoint(ply_file)
        spp = spp.numpy()

        torch.save(spp, os.path.join("dataset/scannetv2/superpoints", f"{scan}.pth"))

    scans_test = os.listdir("dataset/scannetv2/scans_test/*")
    for scan in scans_test:
        ply_file = os.path.join("dataset/scannetv2/scans_test", scan, f"{scan}_vh_clean_2.ply")
        spp = get_superpoint(ply_file)
        spp = spp.numpy()

        torch.save(spp, os.path.join("dataset/scannetv2/superpoints", f"{scan}.pth"))
