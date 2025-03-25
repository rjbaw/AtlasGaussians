#!/usr/bin/env python
# coding=utf-8
import torch
import torch.nn.functional as F
from pytorch3d.ops.knn import knn_gather, knn_points


def _chamfer_distance_single_direction(x, y, x_attr, y_attr):
    '''
    Args:
        x: [N, P1, 3]
        y: [N, P2, 3]
        x_attr: [N, P1, D], extra attributes of x
        y_attr: [N, P2, D], extra attributes of y
    Returns:
        cham_x: chamfer distance
        cham_attr_x: distance of attributes
    '''
    x_nn = knn_points(x, y, K=1)
    cham_x = x_nn.dists[..., 0]  # [N, P1]
    cham_x = cham_x.mean()

    x_attr_near = knn_gather(y_attr, x_nn.idx)[..., 0, :]  # [N, P1, D]
    cham_attr_x = F.mse_loss(x_attr, x_attr_near)

    return cham_x, cham_attr_x


def chamfer_distance(x, y, x_attr, y_attr):
    '''
    Args:
        x: [N, P1, 3]
        y: [N, P2, 3]
        x_attr: [N, P1, D], extra attributes of x
        y_attr: [N, P2, D], extra attributes of y
    Returns:
        cham: chamfer distance
        cham_attr: distance of attributes
    '''
    cham_x, cham_attr_x = _chamfer_distance_single_direction(x, y, x_attr, y_attr)
    cham_y, cham_attr_y = _chamfer_distance_single_direction(y, x, y_attr, x_attr)

    cham = cham_x + cham_y
    cham_attr = cham_attr_x + cham_attr_y

    return cham, cham_attr



