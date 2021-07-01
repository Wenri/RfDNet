import numpy as np
import torch

from torch.nn import functional as F
from net_utils.box_util import get_3d_box_cuda
from net_utils.libs import flip_axis_to_camera_cuda, flip_axis_to_depth_cuda


@torch.jit.script
def pointcloud2voxel_fast(pc: torch.Tensor, voxel_size: int = 32, grid_size: float = 1.):
    b, n, _ = pc.shape
    half_size = grid_size / 2.
    pc_grid = (pc + half_size) * (voxel_size - 1.)
    indices_floor = torch.floor(pc_grid)
    indices = indices_floor.long()
    valid = torch.logical_and(indices >= 0, indices < voxel_size - 1)
    valid = torch.all(valid, 2)
    batch_indices = torch.arange(b, device=pc.device)
    batch_indices = batch_indices.unsqueeze(1).expand(-1, n).unsqueeze(2)
    indices = torch.cat((batch_indices, indices), 2)
    indices = torch.reshape(indices, (-1, 4))

    r = pc_grid - indices_floor
    rr = (1. - r, r)
    valid = torch.flatten(valid)
    indices = indices[valid]

    out_shape = (b, voxel_size, voxel_size, voxel_size)
    voxels = torch.flatten(torch.zeros(*out_shape, dtype=pc.dtype, device=pc.device))
    out_shape_tensor = torch.tensor(out_shape, dtype=indices.dtype, device=pc.device)

    # interpolate_scatter3d
    for i in range(2):
        for j in range(2):
            for k in range(2):
                updates_raw = rr[i][..., 0] * rr[j][..., 1] * rr[k][..., 2]
                updates = torch.flatten(updates_raw)[valid]

                indices_shift = torch.tensor([[0, i, j, k]], dtype=indices.dtype, device=pc.device)
                indices_loc = (indices + indices_shift).T
                indices_loc = [indices_loc[i].long() * torch.prod(out_shape_tensor[i + 1:])
                               for i in range(len(out_shape))]
                indices_loc = indices_loc[0] + indices_loc[1] + indices_loc[2] + indices_loc[3]
                voxels.scatter_add_(-1, indices_loc, updates)

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


def voxels_from_proposals(cfg, end_points, data, BATCH_PROPOSAL_IDs):
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

    transform_shapenet = torch.tensor([[0., 0., -1.], [-1., 0., 0.], [0., 1., 0.]], device=device)

    cos_orientation, sin_orientation = torch.cos(heading_angles), torch.sin(heading_angles)
    zero_orientation, one_orientation = torch.zeros_like(heading_angles), torch.ones_like(heading_angles)
    axis_rectified = torch.stack([torch.stack([cos_orientation, -sin_orientation, zero_orientation], dim=-1),
                                  torch.stack([sin_orientation, cos_orientation, zero_orientation], dim=-1),
                                  torch.stack([zero_orientation, zero_orientation, one_orientation], dim=-1)], dim=-1)
    # world to obj
    point_clouds = data['point_clouds'][..., 0:3].unsqueeze(1).expand(-1, N_proposals, -1, -1)
    point_clouds = torch.matmul(point_clouds - pred_centers.unsqueeze(2), axis_rectified.transpose(2, 3))

    pcd_cuda = torch.matmul(point_clouds / box_size.unsqueeze(2), transform_shapenet)
    all_voxels = pointcloud2voxel_fast(pcd_cuda.view(batch_size * N_proposals, -1, 3))

    return all_voxels


def pc2voxel_test():
    p = torch.rand(1, 1, 3) - 0.5
    x = pointcloud2voxel_fast(p)
    print(torch.where(x))
    x = x.transpose(1, 3).unsqueeze(1)
    p = p.unsqueeze(1).unsqueeze(1) * 2
    ret = F.grid_sample(x, p, padding_mode='border')
    print(float(ret.flatten()))


if __name__ == '__main__':
    pc2voxel_test()
