// Copyright (c) Facebook, Inc. and its affiliates.


#include "ball_query.h"
#include "group_points.h"
#include "interpolate.h"
#include "sampling.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gather_points", &gather_points);
  m.def("gather_points_grad", &gather_points_grad);
  m.def("furthest_point_sampling", &furthest_point_sampling);
  m.def("furthest_point_sampling_weights", &furthest_point_sampling_weights);
  m.def("furthest_point_sampling_hybrid", &furthest_point_sampling_hybrid);
  m.def("furthest_point_sampling_with_dist", &furthest_point_sampling_with_dist);

  m.def("three_nn", &three_nn);
  m.def("three_interpolate", &three_interpolate);
  m.def("three_interpolate_grad", &three_interpolate_grad);

  m.def("ball_query", &ball_query);

  m.def("group_points", &group_points);
  m.def("group_points_grad", &group_points_grad);
}
