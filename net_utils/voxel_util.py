import open3d as o3d
import numpy as np
import torch

from net_utils.box_util import get_3d_box_cuda
from net_utils.libs import flip_axis_to_camera_cuda, flip_axis_to_depth_cuda, extract_pc_in_box3d


def voxels_from_point_cloud(pcd, centre, voxel_size=0.05):
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd))
    min_bound = centre - 1
    max_bound = centre + 1
    min_bound.resize([3, 1])
    max_bound.resize([3, 1])
    voxel_grid = o3d.geometry.VoxelGrid. create_from_point_cloud_within_bounds(pcd, voxel_size, min_bound, max_bound)
    return voxel_grid


def voxels_from_proposals(dataset_config, end_points, data, BATCH_PROPOSAL_IDs):
    device = end_points['center'].device

    # gather proposal centers
    gather_ids = BATCH_PROPOSAL_IDs[..., 0].unsqueeze(-1).repeat(1, 1, 3).long().to(device)
    pred_centers = torch.gather(end_points['center'], 1, gather_ids)
    pred_centers_upright_camera = flip_axis_to_camera_cuda(pred_centers)

    # gather proposal orientations
    pred_heading_class = torch.argmax(end_points['heading_scores'], -1)  # B,num_proposal
    heading_residuals = end_points['heading_residuals_normalized'] * (
            np.pi / dataset_config.num_heading_bin)  # Bxnum_proposalxnum_heading_bin
    pred_heading_residual = torch.gather(heading_residuals, 2,
                                         pred_heading_class.unsqueeze(-1))  # B,num_proposal,1
    pred_heading_residual.squeeze_(2)
    heading_angles = dataset_config.class2angle_cuda(pred_heading_class, pred_heading_residual)
    heading_angles = torch.gather(heading_angles, 1, BATCH_PROPOSAL_IDs[..., 0])

    # gather proposal box size
    pred_size_class = torch.argmax(end_points['size_scores'], -1)
    size_residuals = end_points['size_residuals_normalized'] * torch.from_numpy(
        dataset_config.mean_size_arr.astype(np.float32)).cuda().unsqueeze(0).unsqueeze(0)
    pred_size_residual = torch.gather(size_residuals, 2,
                                      pred_size_class.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1, 3))
    pred_size_residual.squeeze_(2)
    box_size = dataset_config.class2size_cuda(pred_size_class, pred_size_residual)
    box_size = torch.gather(box_size, 1, gather_ids)

    corners_3d_upright_camera = get_3d_box_cuda(box_size, -heading_angles, pred_centers_upright_camera)
    box3d = flip_axis_to_depth_cuda(corners_3d_upright_camera)

    for pcs, boxes3d, centers in zip(data['point_clouds'].cpu().numpy()[..., 0:3], box3d.detach().cpu().numpy(),
                                     pred_centers.detach().cpu().numpy()):
        for box3d, centre in zip(boxes3d, centers):
            pcd, ids = extract_pc_in_box3d(pcs, box3d)
            voxels = voxels_from_point_cloud(pcd.astype(np.float64), centre.astype(np.float64))
            voxels = np.stack([a.grid_index for a in voxels.get_voxels()])
            states_max = np.max(voxels, axis=0)
            states_min = np.min(voxels, axis=0)
            print(states_min, states_max)

    return voxels
