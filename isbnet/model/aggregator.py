import torch
import torch.nn as nn
import torch.nn.functional as F

from isbnet.ops import ballquery_batchflat, furthestsampling_batchflat
from isbnet.pointnet2.pointnet2_utils import ball_query, furthest_point_sample
from .module_utils import Conv1d, SharedMLP


class LocalAggregator(nn.Module):
    def __init__(
        self,
        mlp_dim: int = 32,
        n_sample: int = 1024,
        radius: float = 0.4,
        n_neighbor: int = 64,
        n_neighbor_post: int = 64,
        bn: bool = True,
        use_xyz: bool = True,
    ) -> None:
        super().__init__()

        self.n_sample = n_sample
        self.radius = radius
        self.n_neighbor = n_neighbor
        self.n_neighbor_post = n_neighbor_post
        self.use_xyz = use_xyz

        self.radius_post = 2 * radius

        mlp_spec1 = [mlp_dim, mlp_dim, mlp_dim * 2]
        mlp_spec1[0] += 3 + 3

        self.mlp_module1 = SharedMLP(mlp_spec1, bn=bn)

        mlp_spec2 = [mlp_dim * 2, mlp_dim * 2]
        if use_xyz and len(mlp_spec2) > 0:
            mlp_spec2[0] += 3 + 3
        self.mlp_module2 = SharedMLP(mlp_spec2, bn=bn, activation=None)

        mlp_module3 = [
            Conv1d(in_size=mlp_dim * 2, out_size=mlp_dim * 2 * 4, bn=True),
            Conv1d(
                in_size=mlp_dim * 2 * 4,
                out_size=mlp_dim * 2,
                bn=True,
                activation=None,
            ),
        ]

        self.mlp_module3 = nn.Sequential(*mlp_module3)

        self.skip_act = nn.ReLU()

    def forward(self, locs, feats, boxes, batch_offsets=None, batch_size=1, sampled_before=False):
        if len(locs.shape) == 2:
            return self.forward_batchflat(locs, feats, boxes, batch_offsets, batch_size, sampled_before=sampled_before)
        elif len(locs.shape) == 3:
            return self.forward_batch(locs, feats, boxes, batch_size, sampled_before=sampled_before)
        else:
            raise RuntimeError("Invalid shape to LocalAggregator")

    def forward_batchflat(self, locs, feats, boxes, batch_offsets, batch_size, sampled_before=False):
        dim_boxes = boxes[:, 3:] - boxes[:, :3]

        fps_offsets = torch.arange(
            0, self.n_sample * (batch_size + 1), self.n_sample, dtype=torch.int, device=locs.device
        )
        fps_inds = furthestsampling_batchflat(locs, batch_offsets, fps_offsets)

        fps_locs_float = locs[fps_inds.long(), :]  # m, 3
        fps_dim_boxes = dim_boxes[fps_inds.long(), :]  # m, 3
        fps_boxes = boxes[fps_inds.long(), :]  # m, 6

        neighbor_inds = ballquery_batchflat(
            self.radius, self.n_neighbor, locs, fps_locs_float, batch_offsets, fps_offsets
        )  # m, nsample
        neighbor_inds = neighbor_inds.reshape(-1).long()

        grouped_xyz = torch.gather(locs, 0, neighbor_inds[:, None].expand(-1, locs.shape[-1])).reshape(
            batch_size * self.n_sample, self.n_neighbor, locs.shape[-1]
        )  # m, nsample, 3
        grouped_xyz = (grouped_xyz - fps_locs_float[:, None, :]) / self.radius

        grouped_dim_box = torch.gather(dim_boxes, 0, neighbor_inds[:, None].expand(-1, dim_boxes.shape[-1])).reshape(
            batch_size * self.n_sample, self.n_neighbor, dim_boxes.shape[-1]
        )  # m, nsample, 3
        grouped_dim_box = torch.abs(grouped_dim_box - fps_dim_boxes[:, None, :])

        grouped_features = torch.gather(feats, 0, neighbor_inds[:, None].expand(-1, feats.shape[-1])).reshape(
            batch_size * self.n_sample, self.n_neighbor, feats.shape[-1]
        )  # m, nsample, 3
        grouped_features = torch.cat([grouped_xyz, grouped_dim_box, grouped_features], dim=-1)  # m, nsample, C

        grouped_xyz = grouped_xyz.reshape(batch_size, self.n_sample, self.n_neighbor, -1)
        grouped_features = grouped_features.reshape(batch_size, self.n_sample, self.n_neighbor, -1)

        grouped_features = grouped_features.permute(0, 3, 1, 2).contiguous()  # B, C, nqueries, npoints

        # NOTE MLP and reduce
        new_features = self.mlp_module1(grouped_features)  # (B, mlp[-1], npoint, nsample)
        new_features = F.max_pool2d(new_features, kernel_size=[1, new_features.size(3)])  # (B, mlp[-1], npoint, 1)
        new_features = new_features.squeeze(-1)  # (B, mlp[-1], npoint)

        # NOTE group 2nd
        # query_locs B, m, 3
        fps_locs_float = fps_locs_float.reshape(batch_size, self.n_sample, -1).contiguous()
        fps_dim_boxes = fps_dim_boxes.reshape(batch_size, self.n_sample, -1).contiguous()
        fps_boxes = fps_boxes.reshape(batch_size, self.n_sample, -1).contiguous()
        fps_inds = fps_inds.reshape(batch_size, self.n_sample)
        # fps_offsets = fps_offsets.reshape(batch_size, -1)

        identity = new_features

        neighbor_inds2 = ball_query(self.radius_post, self.n_neighbor_post, fps_locs_float, fps_locs_float)  # B, m, m1
        neighbor_inds2 = neighbor_inds2.reshape(batch_size, -1).long()  # b, m*m1

        grouped_xyz2 = torch.gather(
            fps_locs_float, 1, neighbor_inds2[:, :, None].expand(-1, -1, fps_locs_float.shape[-1])
        ).reshape(
            batch_size, self.n_sample, self.n_neighbor_post, fps_locs_float.shape[-1]
        )  # m, nsample, 3
        grouped_xyz2 = (grouped_xyz2 - fps_locs_float[:, :, None, :]) / self.radius_post
        grouped_xyz2 = grouped_xyz2.permute(0, 3, 1, 2)

        grouped_dim_box2 = torch.gather(
            fps_dim_boxes, 1, neighbor_inds2[:, :, None].expand(-1, -1, fps_dim_boxes.shape[-1])
        ).reshape(
            batch_size, self.n_sample, self.n_neighbor_post, fps_dim_boxes.shape[-1]
        )  # m, nsample, 3
        grouped_dim_box2 = torch.abs(grouped_dim_box2 - fps_dim_boxes[:, :, None, :])
        grouped_dim_box2 = grouped_dim_box2.permute(0, 3, 1, 2)

        grouped_features2 = torch.gather(
            new_features, 2, neighbor_inds2[:, None, :].expand(-1, new_features.shape[1], -1)
        ).reshape(
            batch_size, new_features.shape[1], self.n_sample, self.n_neighbor_post
        )  # m, nsample, 3
        grouped_features2 = torch.cat([grouped_xyz2, grouped_dim_box2, grouped_features2], dim=1)  # m, nsample, C

        # NOTE MLP and reduce
        new_features2 = self.mlp_module2(grouped_features2)  # (B, mlp[-1], npoint, nsample)
        new_features2 = F.max_pool2d(new_features2, kernel_size=[1, new_features2.size(3)])  # (B, mlp[-1], npoint, 1)
        new_features2 = new_features2.squeeze(-1)  # (B, mlp[-1], npoint)

        # NOTE big MLP
        new_features3 = self.mlp_module3(new_features2)

        # NOTE Skip connection
        feats = self.skip_act(new_features3 + identity)

        return fps_locs_float, feats, fps_boxes, fps_inds

    def forward_batch(self, locs, feats, boxes, batch_size, sampled_before=False):
        # locs: B, N, 3
        # boxes: B, N, 6
        dim_boxes = boxes[..., 3:] - boxes[..., :3]
        feats = feats.permute(0, 2, 1)

        if sampled_before:
            fps_inds = torch.arange(0, self.n_sample, dtype=torch.long, device=feats.device)[None, ...].expand(
                batch_size, self.n_sample
            )

            fps_locs_float = locs[:, : self.n_sample, :].contiguous()
            fps_dim_boxes = dim_boxes[:, : self.n_sample, :].contiguous()
            fps_boxes = boxes[:, : self.n_sample, :].contiguous()

        else:
            fps_inds = furthest_point_sample(locs, self.n_sample).long()

            fps_locs_float = torch.gather(locs, 1, fps_inds[..., None].expand(-1, -1, locs.shape[-1]))
            fps_dim_boxes = torch.gather(dim_boxes, 1, fps_inds[..., None].expand(-1, -1, dim_boxes.shape[-1]))
            fps_boxes = torch.gather(boxes, 1, fps_inds[..., None].expand(-1, -1, boxes.shape[-1]))

        neighbor_inds = ball_query(self.radius, self.n_neighbor, locs, fps_locs_float)  # B, m, m1
        neighbor_inds = neighbor_inds.reshape(batch_size, -1).long()  # b, m*m1

        grouped_xyz = torch.gather(locs, 1, neighbor_inds[:, :, None].expand(-1, -1, locs.shape[-1])).reshape(
            batch_size, self.n_sample, self.n_neighbor, locs.shape[-1]
        )  # m, nsample, 3
        grouped_xyz = (grouped_xyz - fps_locs_float[:, :, None, :]) / self.radius

        grouped_dim_box = torch.gather(
            dim_boxes, 1, neighbor_inds[:, :, None].expand(-1, -1, dim_boxes.shape[-1])
        ).reshape(
            batch_size, self.n_sample, self.n_neighbor, dim_boxes.shape[-1]
        )  # m, nsample, 3
        grouped_dim_box = torch.abs(grouped_dim_box - fps_dim_boxes[:, :, None, :])

        grouped_features = torch.gather(feats, 1, neighbor_inds[:, :, None].expand(-1, -1, feats.shape[-1])).reshape(
            batch_size, self.n_sample, self.n_neighbor, feats.shape[-1]
        )  # m, nsample, 3
        grouped_features = torch.cat([grouped_xyz, grouped_dim_box, grouped_features], dim=-1)  # m, nsample, C
        grouped_features = grouped_features.permute(0, 3, 1, 2).contiguous()  # B, C, nqueries, npoints

        # NOTE MLP and reduce
        new_features = self.mlp_module1(grouped_features)  # (B, mlp[-1], npoint, nsample)
        new_features = F.max_pool2d(new_features, kernel_size=[1, new_features.size(3)])  # (B, mlp[-1], npoint, 1)
        new_features = new_features.squeeze(-1)  # (B, mlp[-1], npoint)

        identity = new_features

        neighbor_inds2 = ball_query(self.radius_post, self.n_neighbor_post, fps_locs_float, fps_locs_float)  # B, m, m1
        neighbor_inds2 = neighbor_inds2.reshape(batch_size, -1).long()  # b, m*m1

        grouped_xyz2 = torch.gather(
            fps_locs_float, 1, neighbor_inds2[:, :, None].expand(-1, -1, fps_locs_float.shape[-1])
        ).reshape(
            batch_size, self.n_sample, self.n_neighbor_post, fps_locs_float.shape[-1]
        )  # m, nsample, 3
        grouped_xyz2 = (grouped_xyz2 - fps_locs_float[:, :, None, :]) / self.radius_post
        grouped_xyz2 = grouped_xyz2.permute(0, 3, 1, 2)

        grouped_dim_box2 = torch.gather(
            fps_dim_boxes, 1, neighbor_inds2[:, :, None].expand(-1, -1, fps_dim_boxes.shape[-1])
        ).reshape(
            batch_size, self.n_sample, self.n_neighbor_post, fps_dim_boxes.shape[-1]
        )  # m, nsample, 3
        grouped_dim_box2 = torch.abs(grouped_dim_box2 - fps_dim_boxes[:, :, None, :])
        grouped_dim_box2 = grouped_dim_box2.permute(0, 3, 1, 2)

        grouped_features2 = torch.gather(
            new_features, 2, neighbor_inds2[:, None, :].expand(-1, new_features.shape[1], -1)
        ).reshape(
            batch_size, new_features.shape[1], self.n_sample, self.n_neighbor_post
        )  # m, nsample, 3
        grouped_features2 = torch.cat([grouped_xyz2, grouped_dim_box2, grouped_features2], dim=1)  # m, nsample, C

        # NOTE MLP and reduce
        new_features2 = self.mlp_module2(grouped_features2)  # (B, mlp[-1], npoint, nsample)
        new_features2 = F.max_pool2d(new_features2, kernel_size=[1, new_features2.size(3)])  # (B, mlp[-1], npoint, 1)
        new_features2 = new_features2.squeeze(-1)  # (B, mlp[-1], npoint)

        # NOTE big MLP
        new_features3 = self.mlp_module3(new_features2)

        # NOTE Skip connection
        feats = self.skip_act(new_features3 + identity)

        return fps_locs_float, feats, fps_boxes, fps_inds
