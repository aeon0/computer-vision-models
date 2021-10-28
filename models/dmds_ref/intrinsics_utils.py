# coding=utf-8
# Copyright 2021 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utils for handling camera intrinsics in TensorFlow."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import tensorflow as tf


def invert_intrinsics_matrix(intrinsics_mat):
  """Inverts an intrinsics matrix.

  Inverting matrices in not supported on TPU. The intrinsics matrix has however
  a closed form expression for its inverse, and this function invokes it.

  Args:
    intrinsics_mat: A tensor of shape [.... 3, 3], representing an intrinsics
      matrix `(in the last two dimensions).

  Returns:
    A tensor of the same shape containing the inverse of intrinsics_mat
  """
  with tf.name_scope('invert_intrinsics_matrix'):
    intrinsics_mat = tf.convert_to_tensor(intrinsics_mat)
    intrinsics_mat_cols = tf.unstack(intrinsics_mat, axis=-1)
    if len(intrinsics_mat_cols) != 3:
      raise ValueError('The last dimension of intrinsics_mat should be 3, not '
                       '%d.' % len(intrinsics_mat_cols))

    fx, _, _ = tf.unstack(intrinsics_mat_cols[0], axis=-1)
    _, fy, _ = tf.unstack(intrinsics_mat_cols[1], axis=-1)
    x0, y0, _ = tf.unstack(intrinsics_mat_cols[2], axis=-1)

    zeros = tf.zeros_like(fx)
    ones = tf.ones_like(fx)

    row1 = tf.stack([1.0 / fx, zeros, zeros], axis=-1)
    row2 = tf.stack([zeros, 1.0 / fy, zeros], axis=-1)
    row3 = tf.stack([-x0 / fx, -y0 / fy, ones], axis=-1)

    return tf.stack([row1, row2, row3], axis=-1)
