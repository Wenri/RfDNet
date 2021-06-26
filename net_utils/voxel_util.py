import itertools

import numpy as np
import torch
import neuralnet_pytorch as nnt

from net_utils.box_util import get_3d_box_cuda, roty_cuda
from net_utils.libs import flip_axis_to_camera_cuda, flip_axis_to_depth_cuda


def pointcloud2voxel_fast(pc: torch.Tensor, voxel_size: int, grid_size=1., filter_outlier=True):
    b, n, _ = pc.shape
    half_size = grid_size / 2.
    valid = (pc >= -half_size) & (pc <= half_size)
    valid = torch.all(valid, 2)
    pc_grid = (pc + half_size) * (voxel_size - 1.)
    indices_floor = torch.floor(pc_grid)
    indices = indices_floor.long()
    batch_indices = torch.arange(b).to(pc.device)
    batch_indices = nnt.utils.shape_padright(batch_indices)
    batch_indices = nnt.utils.tile(batch_indices, (1, n))
    batch_indices = nnt.utils.shape_padright(batch_indices)
    indices = torch.cat((batch_indices, indices), 2)
    indices = torch.reshape(indices, (-1, 4))

    r = pc_grid - indices_floor
    rr = (1. - r, r)
    if filter_outlier:
        valid = valid.flatten()
        indices = indices[valid]

    out_shape = (b,) + (voxel_size,) * 3
    voxels = torch.zeros(*out_shape, device=pc.device).flatten()

    # interpolate_scatter3d
    for pos in itertools.product(range(2), repeat=3):
        updates_raw = rr[pos[0]][..., 0] * rr[pos[1]][..., 1] * rr[pos[2]][..., 2]
        updates = updates_raw.flatten()

        if filter_outlier:
            updates = updates[valid]

        indices_shift = torch.tensor([(0,) + pos]).to(pc.device)
        indices_loc = indices + indices_shift

        voxels.scatter_add_(-1, nnt.utils.ravel_index(indices_loc.t(), out_shape), updates)

    voxels = torch.clamp(voxels, 0., 1.).view(*out_shape)
    return voxels


def gather_bbox(dataset_config, gather_ids, center, heading_class, heading_residual, size_class, size_residual):
    gather_ids_vec3 = gather_ids.unsqueeze(-1).expand(-1, -1, 3)

    # gather proposal centers
    pred_centers = torch.gather(center, 1, gather_ids_vec3)
    pred_centers_upright_camera = flip_axis_to_camera_cuda(pred_centers)

    # gather proposal orientations
    heading_angles = dataset_config.class2angle_cuda(heading_class, heading_residual)
    heading_angles = torch.gather(heading_angles, 1, gather_ids)

    # gather proposal box size
    box_size = dataset_config.class2size_cuda(size_class, size_residual)
    box_size = torch.gather(box_size, 1, gather_ids_vec3)

    corners_3d_upright_camera = get_3d_box_cuda(box_size, -heading_angles, pred_centers_upright_camera)
    box3d = flip_axis_to_depth_cuda(corners_3d_upright_camera)

    return box3d, box_size, pred_centers, heading_angles


def voxels_from_proposals(cfg, end_points, data, BATCH_PROPOSAL_IDs, voxel_size=32):
    device = end_points['center'].device
    dataset_config = cfg.eval_config['dataset_config']
    batch_size = BATCH_PROPOSAL_IDs.size(0)
    N_proposals = BATCH_PROPOSAL_IDs.size(1)

    pred_heading_class = torch.argmax(end_points['heading_scores'], -1)  # B,num_proposal
    heading_residuals = end_points['heading_residuals_normalized'] * (
            np.pi / dataset_config.num_heading_bin)  # Bxnum_proposalxnum_heading_bin
    pred_heading_residual = torch.gather(heading_residuals, 2, pred_heading_class.unsqueeze(-1))  # B,num_proposal,1
    pred_heading_residual.squeeze_(2)

    pred_size_class = torch.argmax(end_points['size_scores'], -1)
    size_residuals = end_points['size_residuals_normalized'] * torch.from_numpy(
        dataset_config.mean_size_arr.astype(np.float32)).cuda().unsqueeze(0).unsqueeze(0)
    pred_size_residual = torch.gather(size_residuals, 2,
                                      pred_size_class.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, 3))
    pred_size_residual.squeeze_(2)

    gather_ids_p = BATCH_PROPOSAL_IDs[..., 0].long().to(device)
    gather_param_p = (end_points['center'], pred_heading_class, pred_heading_residual,
                      pred_size_class, pred_size_residual)

    box3d, box_size, pred_centers, heading_angles = gather_bbox(dataset_config, gather_ids_p, *gather_param_p)

    transform_shapenet = torch.tensor([[0, 0, -1], [-1, 0, 0], [0, 1, 0]], dtype=torch.float32,
                                      device=heading_angles.device)

    cos_orientation, sin_orientation = torch.cos(heading_angles), torch.sin(heading_angles)
    zero_orientation, one_orientation = torch.zeros_like(heading_angles), torch.ones_like(heading_angles)
    axis_rectified = torch.stack([torch.stack([cos_orientation, -sin_orientation, zero_orientation], dim=-1),
                                  torch.stack([sin_orientation, cos_orientation, zero_orientation], dim=-1),
                                  torch.stack([zero_orientation, zero_orientation, one_orientation], dim=-1)], dim=-1)
    # world to obj
    point_clouds = data['point_clouds'][..., 0:3].unsqueeze(1).expand(-1, N_proposals, -1, -1)
    point_clouds = torch.matmul(point_clouds - pred_centers.unsqueeze(2), axis_rectified.transpose(2, 3))

    pcd_cuda = torch.matmul(point_clouds / box_size.unsqueeze(2), transform_shapenet)
    all_voxels = pointcloud2voxel_fast(pcd_cuda.view(batch_size * N_proposals, -1, 3), voxel_size)

    return all_voxels
