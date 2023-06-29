import spconv.pytorch as spconv
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

import functools
import math
from ..ops import voxelization
from ..util import cuda_cast, rle_encode_gpu, rle_encode_gpu_batch
from .aggregator import LocalAggregator
from .blocks import MLP, GenericMLP, ResidualBlock, UBlock, conv_with_kaiming_uniform
from .model_utils import (
    custom_scatter_mean,
    get_batch_offsets,
    get_cropped_instance_label,
    get_instance_info,
    get_spp_gt,
    get_subsample_gt,
    nms,
    random_downsample,
    superpoint_align,
)


class ISBNet(nn.Module):
    def __init__(
        self,
        channels=32,
        num_blocks=7,
        semantic_only=False,
        semantic_classes=20,
        instance_classes=18,
        semantic_weight=None,
        sem2ins_classes=[],
        ignore_label=-100,
        with_coords=True,
        instance_head_cfg=None,
        test_cfg=None,
        fixed_modules=[],
        criterion=None,
        dataset_name="scannetv2",
        trainall=False,
        voxel_scale=50,
        use_spp_pool=True,
        filter_bg_thresh=0.1,
        iterative_sampling=True,
        mask_dim_out=32,
    ):
        super().__init__()
        self.channels = channels
        self.num_blocks = num_blocks
        self.semantic_only = semantic_only
        self.semantic_classes = semantic_classes
        self.instance_classes = instance_classes
        self.semantic_weight = semantic_weight
        self.sem2ins_classes = sem2ins_classes
        self.ignore_label = ignore_label
        self.with_coords = with_coords
        self.instance_head_cfg = instance_head_cfg
        self.test_cfg = test_cfg
        self.fixed_modules = fixed_modules
        self.voxel_scale = voxel_scale

        self.criterion = criterion
        self.dataset_name = dataset_name

        self.label_shift = semantic_classes - instance_classes

        self.trainall = trainall

        self.use_spp_pool = use_spp_pool
        self.filter_bg_thresh = filter_bg_thresh

        # NOTE iterative sampling
        self.iterative_sampling = iterative_sampling

        self.mask_dim_out = mask_dim_out

        in_channels = 6 if with_coords else 3

        block = ResidualBlock
        norm_fn = functools.partial(nn.BatchNorm1d, eps=1e-4, momentum=0.1)

        # NOTE backbone
        self.input_conv = spconv.SparseSequential(
            spconv.SubMConv3d(in_channels, channels, kernel_size=3, padding=1, bias=False, indice_key="subm1")
        )
        block_channels = [channels * (i + 1) for i in range(num_blocks)]
        self.unet = UBlock(block_channels, norm_fn, 2, block, indice_key_id=1)
        self.output_layer = spconv.SparseSequential(norm_fn(channels), nn.ReLU())

        # # NOTE point-wise prediction
        self.semantic_linear = MLP(channels, semantic_classes, norm_fn=norm_fn, num_layers=2)
        self.offset_linear = MLP(channels, 3, norm_fn=norm_fn, num_layers=2)

        # NOTE BBox
        self.offset_vertices_linear = MLP(channels, 3 * 2, norm_fn=norm_fn, num_layers=2)
        self.box_conf_linear = MLP(channels, 1, norm_fn=norm_fn, num_layers=2)

        if not self.semantic_only:
            self.point_aggregator1 = LocalAggregator(
                mlp_dim=self.channels,
                n_sample=instance_head_cfg.n_sample_pa1,
                radius=0.2 * instance_head_cfg.radius_scale,
                n_neighbor=instance_head_cfg.neighbor,
                n_neighbor_post=instance_head_cfg.neighbor * 2,
            )

            self.point_aggregator2 = LocalAggregator(
                mlp_dim=self.channels * 2,
                n_sample=instance_head_cfg.n_queries,
                radius=0.4 * instance_head_cfg.radius_scale,
                n_neighbor=instance_head_cfg.neighbor,
                n_neighbor_post=instance_head_cfg.neighbor,
            )

            self.inst_shared_mlp = GenericMLP(
                input_dim=self.channels * 4,
                hidden_dims=[self.channels * 4],
                output_dim=instance_head_cfg.dec_dim,
                norm_fn_name="bn1d",
                activation="relu",
                use_conv=True,
                output_use_activation=True,
                output_use_norm=True,
            )

            self.inst_sem_head = GenericMLP(
                input_dim=instance_head_cfg.dec_dim,
                hidden_dims=[instance_head_cfg.dec_dim, instance_head_cfg.dec_dim],
                norm_fn_name="bn1d",
                activation="relu",
                use_conv=True,
                output_dim=self.instance_classes + 1,
            )

            self.inst_conf_head = GenericMLP(
                input_dim=instance_head_cfg.dec_dim,
                hidden_dims=[instance_head_cfg.dec_dim, instance_head_cfg.dec_dim],
                norm_fn_name="bn1d",
                activation="relu",
                use_conv=True,
                output_dim=1,
            )

            self.inst_box_head = GenericMLP(
                input_dim=instance_head_cfg.dec_dim,
                hidden_dims=[instance_head_cfg.dec_dim, instance_head_cfg.dec_dim],
                norm_fn_name="bn1d",
                activation="relu",
                use_conv=True,
                output_dim=6,  # xyz_min, xyz_max
            )

            # NOTE dyco
            self.init_dyco()

        self.init_weights()

        for mod in fixed_modules:
            mod = getattr(self, mod)
            for param in mod.parameters():
                param.requires_grad = False

        if "input_conv" in self.fixed_modules and "unet" in self.fixed_modules:
            self.freeze_backbone = True
        else:
            self.freeze_backbone = False

    def init_dyco(self):
        conv_block = conv_with_kaiming_uniform("BN", activation=True)

        mask_tower = [
            conv_block(self.channels, self.channels),
            conv_block(self.channels, self.channels),
            conv_block(self.channels, self.channels),
            nn.Conv1d(self.channels, self.mask_dim_out, 1),
        ]
        self.add_module("mask_tower", nn.Sequential(*mask_tower))

        weight_nums = [(self.mask_dim_out + 3 + 3) * self.mask_dim_out, self.mask_dim_out * (self.mask_dim_out//2), (self.mask_dim_out//2) * 1]
        bias_nums = [self.mask_dim_out, (self.mask_dim_out//2), 1]

        self.weight_nums = weight_nums
        self.bias_nums = bias_nums
        self.num_gen_params = sum(weight_nums) + sum(bias_nums)

        # NOTE convolution before the condinst take place (convolution num before the generated parameters take place)
        inst_mask_head = [
            conv_block(self.instance_head_cfg.dec_dim, self.instance_head_cfg.dec_dim),
            conv_block(self.instance_head_cfg.dec_dim, self.instance_head_cfg.dec_dim),
        ]
        controller_head = nn.Conv1d(self.instance_head_cfg.dec_dim, self.num_gen_params, kernel_size=1)
        torch.nn.init.normal_(controller_head.weight, std=0.01)
        torch.nn.init.constant_(controller_head.bias, 0)
        inst_mask_head.append(controller_head)
        self.add_module("inst_mask_head", nn.Sequential(*inst_mask_head))

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, MLP):
                m.init_weights()

    def train(self, mode=True):
        super().train(mode)
        for mod in self.fixed_modules:
            mod = getattr(self, mod)
            for m in mod.modules():
                if isinstance(m, nn.BatchNorm1d):
                    m.eval()

    def forward(self, batch, return_loss=False, epoch=0):
        if return_loss:
            return self.forward_train(**batch, epoch=epoch)
        else:
            return self.forward_test(**batch)

    @cuda_cast
    def forward_train(
        self,
        batch_idxs,
        voxel_coords,
        p2v_map,
        v2p_map,
        coords_float,
        feats,
        semantic_labels,
        instance_labels,
        spps,
        spatial_shape,
        batch_size,
        epoch,
        **kwargs
    ):
        # NOTE voxelization
        voxel_semantic_labels = semantic_labels[p2v_map[:, 1].long()]
        voxel_instance_labels = instance_labels[p2v_map[:, 1].long()]
        voxel_instance_labels = get_cropped_instance_label(voxel_instance_labels)

        voxel_spps = spps[p2v_map[:, 1].long()]
        voxel_spps = torch.unique(voxel_spps, return_inverse=True)[1]

        voxel_coords_float = voxelization(coords_float, p2v_map)

        instance_cls, instance_box, centroid_offset_labels, corners_offset_labels = get_instance_info(
            voxel_coords_float, voxel_instance_labels, voxel_semantic_labels, label_shift=self.label_shift
        )
        ###################

        batch_inputs = dict(
            semantic_labels=voxel_semantic_labels,
            instance_labels=voxel_instance_labels,
        )

        model_outputs = {}

        voxel_output_feats, voxel_batch_idxs = self.forward_backbone(
            feats, coords_float, voxel_coords, spatial_shape, batch_size, p2v_map
        )
        voxel_batch_offsets = get_batch_offsets(voxel_batch_idxs, batch_size)

        (
            voxel_semantic_scores,
            voxel_centroid_offset,
            voxel_corners_offset,
            voxel_box_conf,
        ) = self.forward_pointwise_head(voxel_output_feats)
        voxel_box_preds = voxel_corners_offset + voxel_coords_float.repeat(1, 2)  # n_points, 6

        if self.semantic_only or self.trainall:
            batch_inputs.update(
                dict(
                    coords_float=voxel_coords_float,
                    centroid_offset_labels=centroid_offset_labels,
                    corners_offset_labels=corners_offset_labels,
                )
            )

            model_outputs.update(
                dict(
                    semantic_scores=voxel_semantic_scores,
                    centroid_offset=voxel_centroid_offset,
                    corners_offset=voxel_corners_offset,
                    box_conf=voxel_box_conf,
                )
            )

            if self.semantic_only:
                # NOTE cal loss
                losses = self.criterion(batch_inputs, model_outputs)

                return self.parse_losses(losses)

        # NOTE Point Aggregator
        voxel_semantic_scores_sm = F.softmax(voxel_semantic_scores, dim=1)
        spp_semantic_scores_sm = custom_scatter_mean(
            voxel_semantic_scores_sm, voxel_spps, dim=0, pool=self.use_spp_pool
        )
        spp_object_conditions = torch.any(
            spp_semantic_scores_sm[:, self.label_shift :] >= self.filter_bg_thresh, dim=-1
        )
        object_conditions = spp_object_conditions[voxel_spps]
        object_idxs = torch.nonzero(object_conditions).view(-1)

        if len(object_idxs) <= 100:
            loss_dict = {
                "placeholder": torch.tensor(0.0, requires_grad=True, device=semantic_labels.device, dtype=torch.float)
            }
            return self.parse_losses(loss_dict)

        voxel_batch_idxs_ = voxel_batch_idxs[object_idxs]
        voxel_output_feats_ = voxel_output_feats[object_idxs]
        voxel_coords_float_ = voxel_coords_float[object_idxs]
        voxel_box_preds_ = voxel_box_preds[object_idxs]
        voxel_batch_offsets_ = get_batch_offsets(voxel_batch_idxs_, batch_size)

        query_locs, query_feats, query_boxes, query_inds1 = self.point_aggregator1(
            voxel_coords_float_,
            voxel_output_feats_,
            voxel_box_preds_,
            batch_offsets=voxel_batch_offsets_,
            batch_size=batch_size,
            sampled_before=False,
        )
        query_locs, query_feats, query_boxes, query_inds = self.point_aggregator2(
            query_locs, query_feats, query_boxes, batch_offsets=None, batch_size=batch_size, sampled_before=True
        )
        # -------------------------------

        # NOTE Dynamic conv
        if self.use_spp_pool:
            dc_coords_float, dc_output_feats, dc_box_preds, dc_batch_offsets = self.spp_pool(
                voxel_coords_float, voxel_output_feats, voxel_box_preds, voxel_spps, voxel_batch_offsets
            )

            # NOTE Get GT and loss
            dc_inst_mask_arr = get_spp_gt(
                voxel_instance_labels,
                voxel_spps,
                instance_cls,
                instance_box,
                voxel_batch_offsets,
                batch_size,
                pool=self.use_spp_pool,
            )

        else:
            idxs_subsample = random_downsample(voxel_batch_offsets_, batch_size, n_subsample=15000)
            dc_coords_float = voxel_coords_float_[idxs_subsample]
            dc_box_preds = voxel_box_preds_[idxs_subsample]
            dc_output_feats = voxel_output_feats_[idxs_subsample]
            dc_batch_offsets = get_batch_offsets(voxel_batch_idxs_[idxs_subsample], batch_size)

            subsample_idxs = object_idxs[idxs_subsample]
            dc_inst_mask_arr = get_subsample_gt(
                voxel_instance_labels, subsample_idxs, instance_cls, instance_box, dc_batch_offsets, batch_size
            )

        dc_mask_features = self.mask_tower(torch.unsqueeze(dc_output_feats, dim=2).permute(2, 1, 0)).permute(2, 1, 0)

        # -------------------------------

        cls_logits, mask_logits, conf_logits, box_preds = self.forward_head(
            query_feats, query_locs, dc_mask_features, dc_coords_float, dc_box_preds, dc_batch_offsets
        )
        # -------------------------------

        model_outputs.update(
            dict(
                dc_inst_mask_arr=dc_inst_mask_arr,
                dc_batch_offsets=dc_batch_offsets,
                cls_logits=cls_logits,
                mask_logits=mask_logits,
                conf_logits=conf_logits,
                box_preds=box_preds,
            )
        )

        losses = self.criterion(batch_inputs, model_outputs)
        # -------------------------------

        return self.parse_losses(losses)

    def parse_losses(self, losses):
        loss = sum(v for v in losses.values())
        losses["loss"] = loss
        for loss_name, loss_value in losses.items():
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            losses[loss_name] = loss_value.item()
        return loss, losses

    @cuda_cast
    def forward_test(
        self,
        batch_idxs,
        voxel_coords,
        p2v_map,
        v2p_map,
        coords_float,
        feats,
        semantic_labels,
        instance_labels,
        spps,
        spatial_shape,
        batch_size,
        scan_ids,
        **kwargs
    ):

        assert batch_size == 1

        voxel_spps = spps[p2v_map[:, 1].long()]
        voxel_spps = torch.unique(voxel_spps, return_inverse=True)[1]

        ret = dict(
            scan_id=scan_ids[0],
            semantic_labels=semantic_labels.cpu().numpy(),
            instance_labels=instance_labels.cpu().numpy(),
            coords_float=coords_float.cpu().numpy(),
        )

        voxel_coords_float = voxelization(coords_float, p2v_map)

        voxel_output_feats, voxel_batch_idxs = self.forward_backbone(
            feats, coords_float, voxel_coords, spatial_shape, batch_size, p2v_map, x4_split=self.test_cfg.x4_split
        )

        voxel_batch_offsets = get_batch_offsets(voxel_batch_idxs, batch_size)

        (
            voxel_semantic_scores,
            voxel_centroid_offset,
            voxel_corners_offset,
            voxel_box_conf,
        ) = self.forward_pointwise_head(voxel_output_feats)
        voxel_box_preds = voxel_corners_offset + voxel_coords_float.repeat(1, 2)  # n_points, 6
        voxel_semantic_scores_pred = torch.argmax(voxel_semantic_scores, dim=1)  # N_points
        voxel_semantic_scores_sm = F.softmax(voxel_semantic_scores, dim=1)

        if self.semantic_only:

            _, _, centroid_offset_labels, corners_offset_labels = get_instance_info(
                coords_float, instance_labels, semantic_labels, label_shift=self.label_shift
            )

            ret.update(
                semantic_preds=voxel_semantic_scores_pred[v2p_map.long()].cpu().numpy(),
                centroid_offset=voxel_centroid_offset[v2p_map.long()].cpu().numpy(),
                corners_offset=voxel_corners_offset[v2p_map.long()].cpu().numpy(),
                centroid_offset_labels=centroid_offset_labels.cpu().numpy(),
                corners_offset_labels=corners_offset_labels.cpu().numpy(),
            )

            return ret

        spp_semantic_scores_sm = custom_scatter_mean(
            voxel_semantic_scores_sm,
            voxel_spps[:, None].expand(-1, voxel_semantic_scores_sm.shape[-1]),
            pool=self.use_spp_pool,
        )

        spp_object_conditions = torch.any(
            spp_semantic_scores_sm[:, self.label_shift :] >= self.filter_bg_thresh, dim=-1
        )
        object_conditions = spp_object_conditions[voxel_spps]

        if object_conditions.sum() == 0:
            ret.update(dict(pred_instances=[]))
            return ret

        # NOTE Local Aggregator

        object_idxs = torch.nonzero(object_conditions).view(-1)
        voxel_batch_idxs_ = voxel_batch_idxs[object_idxs]
        voxel_output_feats_ = voxel_output_feats[object_idxs]
        voxel_coords_float_ = voxel_coords_float[object_idxs]
        voxel_box_preds_ = voxel_box_preds[object_idxs]
        voxel_batch_offsets_ = get_batch_offsets(voxel_batch_idxs_, batch_size)

        voxel_spps_ = voxel_spps[object_idxs]
        voxel_spps_ = torch.unique(voxel_spps_, return_inverse=True)[1]

        dc_coords_float, dc_output_feats, dc_box_preds, dc_batch_offsets = self.spp_pool(
            voxel_coords_float, voxel_output_feats, voxel_box_preds, voxel_spps, voxel_batch_offsets
        )

        dc_mask_features = self.mask_tower(torch.unsqueeze(dc_output_feats, dim=2).permute(2, 1, 0)).permute(2, 1, 0)
        # -------------------------------

        if self.dataset_name != "s3dis":
            query_locs1, query_feats1, query_boxes1, query_inds1 = self.point_aggregator1(
                voxel_coords_float_,
                voxel_output_feats_,
                voxel_box_preds_,
                batch_offsets=voxel_batch_offsets_,
                batch_size=batch_size,
                sampled_before=False,
            )
        else:  # NOTE subsample for s3dis as in testing, density is higher than training
            n_samples = ((voxel_coords[:, 0] == 0) | (voxel_coords[:, 0] == 1)).sum()
            s_cond = object_idxs <= n_samples
            voxel_batch_offsets_s = torch.tensor([0, s_cond.sum()], device=voxel_coords.device, dtype=torch.int)
            query_locs1, query_feats1, query_boxes1, query_inds1 = self.point_aggregator1(
                voxel_coords_float_[s_cond],
                voxel_output_feats_[s_cond],
                voxel_box_preds_[s_cond],
                batch_offsets=voxel_batch_offsets_s,
                batch_size=batch_size,
                sampled_before=False,
            )

        # NOTE iterative sampling
        cls_logits_final = []
        mask_logits_final = []
        conf_logits_final = []
        box_preds_final = []

        union_mask = torch.zeros_like(query_inds1[0])

        if self.iterative_sampling:
            n_sample_arr = [192, 128, 64]
        else:
            n_sample_arr = [256]

        query_locs2 = query_locs1.clone()
        query_feats2 = query_feats1.clone()
        query_boxes2 = query_boxes1.clone()

        for i in range(len(n_sample_arr)):
            self.point_aggregator2.n_sample = min(n_sample_arr[i], query_locs2.shape[1])
            query_locs, query_feats, query_boxes, query_inds = self.point_aggregator2(
                query_locs2, query_feats2, query_boxes2, batch_size, sampled_before=False
            )

            forward_head_outputs = self.forward_head(
                query_feats,
                query_locs,
                dc_mask_features,
                dc_coords_float,
                dc_box_preds,
                dc_batch_offsets,
                inference=True,
            )

            if forward_head_outputs is None:
                break

            cls_logits, mask_logits, conf_logits, box_preds = forward_head_outputs

            mask_logit = mask_logits[:, voxel_spps]

            cls_logits_final.append(cls_logits)
            mask_logits_final.append(mask_logit)
            conf_logits_final.append(conf_logits)
            box_preds_final.append(box_preds)

            if i == len(n_sample_arr) - 1:
                break

            mask_logits_0 = (mask_logits[:, voxel_spps_][:, query_inds1[0].long()].detach() > 0).float()
            union_mask = union_mask + mask_logits_0.sum(dim=0)

            nonvisited_query_inds1 = torch.nonzero(union_mask == 0).view(-1)
            if len(nonvisited_query_inds1) == 0:
                break

            query_locs2 = torch.gather(
                query_locs1, dim=1, index=nonvisited_query_inds1[None, :, None].expand(-1, -1, query_locs1.shape[-1])
            )
            query_feats2 = torch.gather(
                query_feats1, dim=2, index=nonvisited_query_inds1[None, None, :].expand(-1, query_feats1.shape[1], -1)
            )
            query_boxes2 = torch.gather(
                query_boxes1, dim=1, index=nonvisited_query_inds1[None, :, None].expand(-1, -1, query_boxes1.shape[-1])
            )

        if len(cls_logits_final) == 0:
            ret.update(dict(pred_instances=[]))
            return ret

        cls_logits_final = torch.cat(cls_logits_final)
        mask_logits_final = torch.cat(mask_logits_final)
        conf_logits_final = torch.cat(conf_logits_final)
        box_preds_final = torch.cat(box_preds_final)

        pred_instances = self.get_instance(
            scan_ids[0],
            cls_logits_final,
            mask_logits_final,
            conf_logits_final,
            box_preds_final,
            v2p_map,
            voxel_semantic_scores_pred,
            spps,
            logit_thresh=self.test_cfg.logit_thresh,
            score_thresh=self.test_cfg.score_thresh,
            npoint_thresh=self.test_cfg.npoint_thresh,
        )

        ret.update(dict(pred_instances=pred_instances))
        return ret

    def forward_backbone(
        self, input_feats, coords_float, voxel_coords, spatial_shape, batch_size, p2v_map, x4_split=False
    ):

        context = torch.no_grad if self.freeze_backbone else torch.enable_grad
        with context():
            if self.with_coords:
                input_feats = torch.cat((input_feats, coords_float), 1)

            voxel_input_feats = voxelization(input_feats, p2v_map)
            input = spconv.SparseConvTensor(voxel_input_feats, voxel_coords.int(), spatial_shape, batch_size)

            if x4_split:
                output_feats, output_batch_idxs = self.forward_4_parts(input)
            else:

                output = self.input_conv(input)
                output = self.unet(output)
                output = self.output_layer(output)

                output_feats = output.features
                output_batch_idxs = output.indices[:, 0]

            return output_feats, output_batch_idxs

    def merge_4_parts(self, x):
        """Helper function for s3dis: take output of 4 parts and merge them."""
        inds = torch.arange(x.size(0), device=x.device)
        p1 = inds[::4]
        p2 = inds[1::4]
        p3 = inds[2::4]
        p4 = inds[3::4]
        ps = [p1, p2, p3, p4]

        x_split = torch.split(x, [p.size(0) for p in ps])
        x_new = torch.zeros_like(x)
        for i, p in enumerate(ps):
            x_new[p] = x_split[i]
        return x_new

    def forward_4_parts(self, x):
        """Helper function for s3dis: devide and forward 4 parts of a scene."""
        outs = []
        for i in range(4):
            inds = x.indices[:, 0] == i
            feats = x.features[inds]
            coords = x.indices[inds]
            coords[:, 0] = 0
            x_new = spconv.SparseConvTensor(
                indices=coords, features=feats, spatial_shape=x.spatial_shape, batch_size=1
            )

            output = self.input_conv(x_new)
            output = self.unet(output)
            output = self.output_layer(output)

            outs.append(output.features)
        outs = torch.cat(outs, dim=0)
        idxs = torch.zeros((outs.shape[0]), dtype=torch.int, device=outs.device)
        return outs, idxs

    def forward_pointwise_head(self, output_feats):

        context = torch.no_grad if self.freeze_backbone else torch.enable_grad
        with context():
            semantic_scores = self.semantic_linear(output_feats)
            centroid_offset = self.offset_linear(output_feats)
            corners_offset = self.offset_vertices_linear(output_feats)
            box_conf = self.box_conf_linear(output_feats).squeeze(-1)
            return semantic_scores, centroid_offset, corners_offset, box_conf

    def spp_pool(self, voxel_coords_float, voxel_output_feats, voxel_box_preds, voxel_spps, voxel_batch_offsets):
        spp_batch_offsets = [0]
        for b in range(len(voxel_batch_offsets) - 1):
            b_s, b_e = voxel_batch_offsets[b], voxel_batch_offsets[b + 1]
            n_spps_ = len(torch.unique(voxel_spps[b_s:b_e]))

            spp_batch_offsets.append(spp_batch_offsets[-1] + n_spps_)
        spp_batch_offsets = torch.tensor(spp_batch_offsets, dtype=torch.long, device=voxel_batch_offsets.device)

        spp_coords_float = custom_scatter_mean(voxel_coords_float, voxel_spps, pool=self.use_spp_pool)
        spp_box_preds = custom_scatter_mean(voxel_box_preds, voxel_spps, pool=self.use_spp_pool)
        spp_output_feats = custom_scatter_mean(voxel_output_feats, voxel_spps, pool=self.use_spp_pool)

        return spp_coords_float, spp_output_feats, spp_box_preds, spp_batch_offsets

    def forward_head(
        self, queries_features, queries_locs, mask_features_, locs_float_, box_preds_, batch_offsets_, inference=False
    ):

        queries_features = self.inst_shared_mlp(queries_features)

        cls_logits = self.inst_sem_head(queries_features).transpose(1, 2)  # batch x n_queries x n_classes
        conf_logits = self.inst_conf_head(queries_features).transpose(1, 2).squeeze(-1)  # batch x n_queries
        box_offsets_preds = self.inst_box_head(queries_features).transpose(1, 2)  # batch x n_queries x 6
        queries_box_preds = box_offsets_preds + queries_locs.repeat(1, 1, 2)

        if inference:
            cls_preds = torch.argmax(cls_logits[0], dim=-1)
            fg_conds = cls_preds < self.instance_classes

            if fg_conds.sum() == 0:
                return None

            queries_features = queries_features[0, :, fg_conds].unsqueeze(0)
            cls_logits = cls_logits[0, fg_conds].unsqueeze(0)
            conf_logits = conf_logits[0, fg_conds].unsqueeze(0)
            queries_box_preds = queries_box_preds[0, fg_conds].unsqueeze(0)
            queries_locs = queries_locs[0, fg_conds].unsqueeze(0)

        batch_size = queries_features.shape[0]
        n_queries = queries_features.shape[2]

        controllers = (
            self.inst_mask_head(queries_features.permute(0, 2, 1)[..., None].flatten(0, 1))
            .squeeze(dim=2)
            .reshape(batch_size, n_queries, -1)
        )

        mask_logits = []
        for b in range(batch_size):
            start, end = batch_offsets_[b], batch_offsets_[b + 1]
            num_points = end - start
            if num_points == 0:
                mask_logits.append(None)
                continue

            controller = controllers[b]  # n_queries x channel
            weights, biases = self.parse_dynamic_params(controller, self.mask_dim_out)

            mask_feature_b = mask_features_[start:end]
            locs_float_b = locs_float_[start:end]
            box_preds_b = box_preds_[start:end]
            queries_locs_b = queries_locs[b]
            queries_box_preds_b = queries_box_preds[b]

            # NOTE as # of points in s3dis is very big, we split dyco into multiple chunks to avoid OOM
            if not self.training and self.dataset_name == "s3dis":
                num_chunks = 16
                chunk_size = math.ceil(n_queries / num_chunks)
            else:
                num_chunks = 1
                chunk_size = n_queries

            mask_logit = []
            for i in range(num_chunks):
                chunk_start = i * chunk_size
                chunk_end = (i + 1) * chunk_size
                weights_split = [w[chunk_start:chunk_end] for w in weights]
                biases_split = [b[chunk_start:chunk_end] for b in biases]

                mask_logit_ = self.mask_heads_forward(
                    mask_feature_b,
                    weights_split,
                    biases_split,
                    weights_split[0].shape[0],
                    locs_float_b,
                    box_preds_b,
                    queries_locs_b[chunk_start:chunk_end],
                    queries_box_preds_b[chunk_start:chunk_end],
                )
                mask_logit.append(mask_logit_)
            mask_logit = torch.cat(mask_logit, dim=0)

            mask_logits.append(mask_logit)

        if inference:
            return cls_logits[0], mask_logits[0], conf_logits[0], queries_box_preds[0]
        return cls_logits, mask_logits, conf_logits, queries_box_preds

    def parse_dynamic_params(self, params, out_channels):
        assert params.dim() == 2
        assert len(self.weight_nums) == len(self.bias_nums)
        assert params.size(1) == sum(self.weight_nums) + sum(self.bias_nums)

        num_instances = params.size(0)
        num_layers = len(self.weight_nums)
        params_splits = list(torch.split_with_sizes(params, self.weight_nums + self.bias_nums, dim=1))

        weight_splits = params_splits[:num_layers]
        bias_splits = params_splits[num_layers:]

        weight_splits[0] = weight_splits[0].reshape(num_instances, out_channels + 6, out_channels)
        bias_splits[0] = bias_splits[0].reshape(num_instances, out_channels)
        weight_splits[1] = weight_splits[1].reshape(num_instances, out_channels, out_channels // 2)
        bias_splits[1] = bias_splits[1].reshape(num_instances, out_channels // 2)
        weight_splits[2] = weight_splits[2].reshape(num_instances, out_channels // 2, 1)
        bias_splits[2] = bias_splits[2].reshape(num_instances, 1)

        return weight_splits, bias_splits  # LIST OF [n_queries, C_in, C_out]

    def mask_heads_forward(
        self, mask_features, weights, biases, num_insts, coords_, boxes_, queries_coords, queries_boxes
    ):
        assert mask_features.dim() == 3
        n_layers = len(weights)
        x = mask_features.permute(2, 1, 0).repeat(num_insts, 1, 1)  # num_inst * c * N_mask

        relative_coords = queries_coords.reshape(-1, 1, 3) - coords_.reshape(1, -1, 3)  # N_inst * N_mask * 3
        relative_coords = relative_coords.permute(0, 2, 1)

        queries_boxes_dim = queries_boxes[:, 3:] - queries_boxes[:, :3]
        boxes_dim = boxes_[:, 3:] - boxes_[:, :3]

        relative_boxes = torch.abs(
            queries_boxes_dim.reshape(-1, 1, 3) - boxes_dim.reshape(1, -1, 3)
        )  # N_inst * N_mask * 3
        relative_boxes = relative_boxes.permute(0, 2, 1)

        x = torch.cat([relative_coords, relative_boxes, x], dim=1)  # num_inst * (3+c) * N_mask

        for i, (w, b) in enumerate(zip(weights, biases)):
            if i < n_layers - 1:
                x = torch.einsum("qab,qan->qbn", w, x) + b.unsqueeze(-1)
                x = F.relu(x)
            else:
                x = torch.einsum(
                    "qab,qan->qbn", w, x
                )  # NOTE Empirically, we do not add biases in last dynamic_conv layer
                x = x.squeeze(1)

        return x

    def get_instance(
        self,
        scan_id,
        cls_logits,
        mask_logits,
        conf_logits,
        box_preds,
        v2p_map,
        semantic_scores_preds,
        spps,
        logit_thresh=0.0,
        score_thresh=0.1,
        npoint_thresh=100,
    ):

        # NOTE only batch 1 when test

        instances = []

        # NOTE for S3DIS: it requires instance mask of background classes (floor, ceil)
        for i in self.sem2ins_classes:
            mask_pred = semantic_scores_preds == i

            mask_pred = mask_pred[v2p_map.long()]

            if torch.count_nonzero(spps == self.ignore_label) != spps.shape[0]:
                mask_pred = superpoint_align(spps, mask_pred.unsqueeze(0)).squeeze(0)

            pred = {}
            pred["scan_id"] = scan_id
            pred["label_id"] = i + 1
            pred["conf"] = 1.0

            # rle encode mask to save memory
            pred["pred_mask"] = rle_encode_gpu(mask_pred)
            instances.append(pred)

        cls_logits = F.softmax(cls_logits, dim=-1)[:, :-1]
        conf_logits = torch.clamp(conf_logits, 0.0, 1.0)
        cls_scores = torch.sqrt(cls_logits * conf_logits[:, None])
        mask_preds = mask_logits >= logit_thresh

        cls_scores_flatten = cls_scores.reshape(-1)  # n_cls * n_queries
        labels = (
            torch.arange(self.instance_classes, device=cls_scores.device)
            .unsqueeze(0)
            .repeat(mask_preds.shape[0], 1)
            .flatten(0, 1)
        )

        # idx = torch.nonzero(cls_scores_flatten >= score_thresh).view(-1)
        _, idx = torch.topk(cls_scores_flatten, k=min(300, len(cls_scores_flatten)), largest=True)
        mask_idx = torch.div(idx, self.instance_classes, rounding_mode="floor")

        cls_final = labels[idx]
        scores_final = cls_scores_flatten[idx]
        masks_final = mask_preds[mask_idx]
        boxes_final = box_preds[mask_idx]

        proposals_npoints = torch.sum(masks_final, dim=1)
        npoints_cond = proposals_npoints >= npoint_thresh
        cls_final = cls_final[npoints_cond]
        masks_final = masks_final[npoints_cond]
        scores_final = scores_final[npoints_cond]
        boxes_final = boxes_final[npoints_cond]

        # NOTE NMS
        masks_final, cls_final, scores_final, boxes_final = nms(
            masks_final, cls_final, scores_final, boxes_final, test_cfg=self.test_cfg
        )

        if len(cls_final) == 0:
            return instances

        # NOTE devoxelization
        masks_final = masks_final[:, v2p_map.long()]

        # NOTE due to the quantization error, we apply superpoint refinement to the final mask after devoxelization
        spp_masks_final = custom_scatter_mean(
            masks_final,
            spps[None, :].expand(len(masks_final), -1),
            dim=-1,
            pool=self.use_spp_pool,
            output_type=torch.float32,
        )
        masks_final = (spp_masks_final >= 0.5)[:, spps]

        valid_masks_final_after_spp = torch.sum(masks_final, dim=1) >= npoint_thresh
        masks_final = masks_final[valid_masks_final_after_spp]
        scores_final = scores_final[valid_masks_final_after_spp]
        cls_final = cls_final[valid_masks_final_after_spp]
        boxes_final = boxes_final[valid_masks_final_after_spp]

        # NOTE save results
        scores_final = scores_final.cpu().numpy()
        cls_final = cls_final.cpu().numpy()
        boxes_final = boxes_final.cpu().numpy()

        # NOTE rle encode mask to save memory
        rle_masks = rle_encode_gpu_batch(masks_final)

        for i in range(cls_final.shape[0]):
            pred = {}
            pred["scan_id"] = scan_id

            if self.dataset_name == "scannetv2":
                pred["label_id"] = cls_final[i] + 1
            elif self.dataset_name == "scannet200":
                pred["label_id"] = cls_final[i] + 1
            elif self.dataset_name == "s3dis":
                pred["label_id"] = cls_final[i] + 3
            elif self.dataset_name == "stpls3d":
                pred["label_id"] = cls_final[i] + 1
            else:
                raise Exception("Invalid dataset name")

            pred["conf"] = scores_final[i]

            pred["pred_mask"] = rle_masks[i]

            instances.append(pred)

        return instances
