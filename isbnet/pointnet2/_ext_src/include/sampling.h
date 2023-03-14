// Copyright (c) Facebook, Inc. and its affiliates.


#pragma once
#include <torch/extension.h>

at::Tensor gather_points(at::Tensor points, at::Tensor idx);
at::Tensor gather_points_grad(at::Tensor grad_out, at::Tensor idx, const int n);
at::Tensor furthest_point_sampling(at::Tensor points, const int nsamples);
at::Tensor furthest_point_sampling_weights(at::Tensor points, at::Tensor weights, const int nsamples);
at::Tensor furthest_point_sampling_hybrid(at::Tensor points, at::Tensor points_offset, const int nsamples, const float ratio);
at::Tensor furthest_point_sampling_with_dist(at::Tensor points, const int nsamples);
