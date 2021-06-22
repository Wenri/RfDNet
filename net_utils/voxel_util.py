import open3d as o3d
import numpy as np
import torch

from net_utils.box_util import get_3d_box_cuda
from net_utils.libs import flip_axis_to_camera_cuda, flip_axis_to_depth_cuda, extract_pc_in_box3d


def voxels_from_point_cloud(pcd, centroid, orientation, sizes, voxel_size=0.05):
    transform_m = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]])

    axis_rectified = np.array(
        [[np.cos(orientation), np.sin(orientation), 0], [-np.sin(orientation), np.cos(orientation), 0], [0, 0, 1]])
    pcd = np.linalg.solve(axis_rectified.dot(np.diag(sizes).dot(transform_m)).T, (pcd - centroid).T)

    min_bound = centroid - 1
    max_bound = centroid + 1
    min_bound.resize([3, 1])
    max_bound.resize([3, 1])

    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd.T))
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud_within_bounds(pcd, voxel_size, min_bound, max_bound)
    return voxel_grid


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


def voxels_from_proposals(dataset_config, end_points, data, BATCH_PROPOSAL_IDs):
    device = end_points['center'].device

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
    gather_param_p = [end_points['center'], pred_heading_class, pred_heading_residual,
                      pred_size_class, pred_size_residual]

    gather_ids_g = BATCH_PROPOSAL_IDs[..., 1].long().to(device)
    gather_param_g = [data[k] for k in ('center_label', 'heading_class_label', 'heading_residual_label',
                                        'size_class_label', 'size_residual_label')]
    box3d_p, box_size_p, centers_p, heading_angles_p = gather_bbox(dataset_config, gather_ids_p, *gather_param_p)
    box3d_g, box_size_g, centers_g, heading_angles_g = gather_bbox(dataset_config, gather_ids_g, *gather_param_g)

    # axis_rectified = np.array(
    #     [[np.cos(orientation), np.sin(orientation), 0], [-np.sin(orientation), np.cos(orientation), 0], [0, 0, 1]])
    cos_orientation_g, sin_orientation_g = torch.cos(heading_angles_g), torch.sin(heading_angles_g)
    zero_orientation, one_orientation = torch.zeros_like(heading_angles_g), torch.ones_like(heading_angles_g)
    axis_rectified_g = torch.stack([torch.stack([cos_orientation_g, -sin_orientation_g, zero_orientation], dim=-1),
                                    torch.stack([sin_orientation_g, cos_orientation_g, zero_orientation], dim=-1),
                                    torch.stack([zero_orientation, zero_orientation, one_orientation], dim=-1)], dim=-1)

    # for pcs, boxes3d, centroid, orientation, sizes in zip(data['point_clouds'][..., 0:3].cpu().numpy(),
    #                                                       box3d.detach().cpu().numpy(),
    #                                                       pred_centers.detach().cpu().numpy(),
    #                                                       heading_angles.detach().cpu().numpy(),
    #                                                       box_size.detach().cpu().numpy()):
    #
    #     for box3d, bb_center, bb_orientation, bb_size in zip(boxes3d, centroid, orientation, sizes):
    #         pcd, ids = extract_pc_in_box3d(pcs, box3d)
    #         voxels = voxels_from_point_cloud(pcd.astype(np.float64), bb_center, bb_orientation, bb_size)
    #         # voxels.get_voxels() could be empty
    #         # voxels = np.stack([a.grid_index for a in voxels.get_voxels()])
    #         # states_max = np.max(voxels, axis=0)
    #         # states_min = np.min(voxels, axis=0)
    #         # print(states_min, states_max)

    return box_size_g
