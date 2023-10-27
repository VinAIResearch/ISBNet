import numpy as np
import torch
from scipy.spatial import KDTree

import configargparse
import glob
import natsort
import os
import sys
from isbnet.pointnet2.pointnet2_utils import ball_query, furthest_point_sample
import math


scene_list = os.listdir('preprocess_notalign')
scene_list = sorted(scene_list)
save_dir = "./preprocess_notalign"
spp_dir = './superpoints'


for scene_name in scene_list:
    # area = scene_name.split(".")[0]
    # name = scene_name.split(".")[1]
    scene_pth = os.path.join(save_dir, scene_name)
    # print('debug', scene_pth)
    xyz, rgb, sem_label, inst_label = torch.load(scene_pth)
    spp = torch.load(os.path.join(spp_dir, scene_name.replace('_inst_nostuff', '')))
    
    # xyz = torch.from_numpy(xyz).cuda()

    if xyz.shape[0] < 300000:
        n_sample = xyz.shape[0]
    else:
        n_sample = min(300000, xyz.shape[0]//4)

    slide = math.floor(xyz.shape[0]/n_sample)
    print(f'sample {xyz.shape[0]}, {n_sample}')
    # fps_inds = furthest_point_sample(xyz[None, :], n_sample).long()
    # fps_inds = fps_inds[0].cpu().numpy()
    fps_inds = np.arange(0,xyz.shape[0], slide)
    # breakpoint()
    # xyz = 
    xyz, rgb, sem_label, inst_label = xyz[fps_inds], rgb[fps_inds], sem_label[fps_inds], inst_label[fps_inds]
    spp = spp[fps_inds]
    save_path = os.path.join('/home/tdngo/Workspace/3dis_ws/ISBNet/dataset/s3dis/preprocess_notalign_subsample',
                             scene_name)
    # save_ind_path = os.path.join('/home/tdngo/Workspace/3dis_ws/ISBNet/dataset/s3dis/subsample_indices',
    #                          scene_name.replace('_inst_nostuff', 'indices'))
    # save_spp_path = os.path.join('/home/tdngo/Workspace/3dis_ws/ISBNet/dataset/s3dis/superpoints_subsample',
    #                          scene_name.replace('_inst_nostuff', ''))
    
    torch.save((xyz, rgb, sem_label, inst_label, spp, fps_inds), save_path)
    # torch.save((fps_inds), save_ind_path)
    # torch.save((spp), save_spp_path)
    # break