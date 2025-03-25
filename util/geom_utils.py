#!/usr/bin/env python
# coding=utf-8
import os
import math
import numpy as np

import torch


def build_grid2D(vmin=0., vmax=1., res=4, device=None):
    ''' Sample a square of points in a grid [vmin, vmax] x [vmin, vmax]
    Args:
        res: resolution
    Returns:
        queries_grid: [res, res, 2]
    '''
    queries_u = torch.linspace(vmin, vmax, res, dtype=torch.float32, device=device)
    queries_v = torch.linspace(vmin, vmax, res, dtype=torch.float32, device=device)
    u_grid, v_grid = torch.meshgrid(queries_u, queries_v, indexing='xy')
    queries_grid = torch.stack((u_grid, v_grid), dim=-1)  # [res, res, 2]
    return queries_grid


def get_points(n_views):
    """
    :param n_points: number of points
    :type n_points: int
    :return: list of points
    :rtype: numpy.ndarray
    """

    rnd = 1.
    points = []
    offset = 2. / n_views
    increment = math.pi * (3. - math.sqrt(5.))

    for i in range(n_views):
        y = ((i * offset) - 1) + (offset / 2)
        r = math.sqrt(1 - pow(y, 2))

        phi = ((i + rnd) % n_views) * increment

        x = math.cos(phi) * r
        z = math.sin(phi) * r

        points.append([x, y, z])
    return np.array(points)


def get_views(n_views, semi_sphere=False):
    """
    Generate a set of views to generate depth maps from.

    :param n_views: number of views per axis
    :type n_views: int
    :return: rotation matrices
    :rtype: [numpy.ndarray]
    """

    Rs = []
    points = get_points(n_views)
    if semi_sphere:
        points[:, 2] = -np.abs(points[:, 2]) - 0.1

    for i in range(points.shape[0]):
        longitude = - math.atan2(points[i, 0], points[i, 1])
        latitude = math.atan2(points[i, 2], math.sqrt(points[i, 0] ** 2 + points[i, 1] ** 2))

        R_x = np.array(
            [[1, 0, 0],
             [0, math.cos(latitude), -math.sin(latitude)],
             [0, math.sin(latitude), math.cos(latitude)]])
        R_y = np.array(
            [[math.cos(longitude), 0, math.sin(longitude)],
             [0, 1, 0],
             [-math.sin(longitude), 0, math.cos(longitude)]])
        R = R_x @ R_y
        Rs.append(R)

    return Rs


def get_W2C_uniform(n_views, radius=2.0, device='cpu'):
    ''' get n_views 4x4 world2cams from a unit sphere uniformly
    '''
    Rs = get_views(n_views)
    T_bx4x4 = torch.zeros((n_views, 4, 4), dtype=torch.float32, device=device)
    T_bx4x4[:, 3, 3] = 1
    T_bx4x4[:, 2, 3] = radius
    Rs = np.stack(Rs)
    T_bx4x4[:, :3, :3] = torch.tensor(Rs, dtype=torch.float32, device=device)
    return T_bx4x4


def fusion(depthmaps, T_bx4x4, fusion_intrisics, n_points, n_views, depth_thres):
    """
    Fuse the rendered depth maps.

    :param depthmaps: depth maps, [V, H, W]
    :param T_bx4x4: world2cam corresponding to views, [V, 4, 4]
    """
    coordspx2 = np.stack(np.nonzero(np.ones((depthmaps.shape[1], depthmaps.shape[2]))), -1).astype(np.float32)
    coordspx2 = coordspx2[:, ::-1]

    # sample points inside mask
    sample_per_view = n_points // n_views
    sample_bxn = torch.zeros((n_views, sample_per_view), device='cuda', dtype=torch.long)
    for i in range(len(T_bx4x4)):
        mask = depthmaps[i] > depth_thres
        valid_idx = torch.nonzero(mask.reshape(-1)).squeeze(-1)
        idx = list(range(valid_idx.shape[0]))
        np.random.shuffle(idx)
        idx = idx[:sample_per_view]
        sample_bxn[i] = torch.tensor(valid_idx[idx])

    depthmaps = torch.gather(depthmaps.reshape(n_views, -1), 1, sample_bxn)

    inv_Ks_bx3x3 = torch.tensor(np.linalg.inv(fusion_intrisics), dtype=torch.float32, device='cuda').unsqueeze(
        0).repeat(n_views, 1, 1)
    inv_T_bx4x4 = torch.inverse(T_bx4x4)

    tf_coords_bxpx2 = torch.tensor(coordspx2.copy(), dtype=torch.float32, device='cuda').unsqueeze(0).repeat(
        n_views, 1, 1)
    tf_coords_bxpx2 = torch.gather(tf_coords_bxpx2, 1, sample_bxn.unsqueeze(-1).repeat(1, 1, 2))

    tf_coords_bxpx3 = torch.cat([tf_coords_bxpx2, torch.ones_like(tf_coords_bxpx2[..., :1])], -1)
    tf_coords_bxpx3 *= depthmaps.reshape(n_views, -1, 1)
    tf_cam_bxpx3 = torch.bmm(inv_Ks_bx3x3, tf_coords_bxpx3.transpose(1, 2)).transpose(1, 2)
    tf_cam_bxpx4 = torch.cat([tf_cam_bxpx3, torch.ones_like(tf_cam_bxpx3[..., :1])], -1)
    tf_world_bxpx3 = torch.bmm(inv_T_bx4x4, tf_cam_bxpx4.transpose(1, 2)).transpose(1, 2)[..., :3]

    return tf_world_bxpx3.reshape(-1, 3)


