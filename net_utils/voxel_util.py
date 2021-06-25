import open3d as o3d
import numpy as np
import torch

from net_utils.box_util import get_3d_box_cuda
from net_utils.libs import flip_axis_to_camera_cuda, flip_axis_to_depth_cuda, extract_pc_in_box3d


def voxels_from_point_cloud(pcd, centroid, orientation, sizes, voxel_size=1 / 32):
    transform_m = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]])
    transform_m = np.matmul(transform_m.T, np.diag(sizes))

    axis_rectified = np.array(
        [[np.cos(orientation), np.sin(orientation), 0], [-np.sin(orientation), np.cos(orientation), 0], [0, 0, 1]])
    transform_m = np.matmul(transform_m, axis_rectified)

    pcd = np.linalg.solve(transform_m.T, (pcd - centroid).T)

    min_bound = np.ones_like(centroid) * -0.5
    max_bound = np.ones_like(centroid) * 0.5

    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd.T))
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud_within_bounds(pcd, voxel_size, min_bound, max_bound)
    voxel_list = voxel_grid.get_voxels()
    voxels = np.zeros((32, 32, 32), dtype=np.float32)

    if voxel_list:
        voxel_index = np.stack([a.grid_index for a in voxel_list])
        voxel_index = np.clip(voxel_index, 0, 31)
        voxels[voxel_index[:, 0], voxel_index[:, 1], voxel_index[:, 2]] = 1.

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


def unpack_data(point_clouds, box3d, box_size, pred_centers, heading_angles):
    for pcs, boxes3d, centroid, orientation, sizes in zip(point_clouds.cpu().numpy(),
                                                          box3d.detach().cpu().numpy(),
                                                          pred_centers.detach().cpu().numpy(),
                                                          heading_angles.detach().cpu().numpy(),
                                                          box_size.detach().cpu().numpy()):
        for box3d, bb_center, bb_orientation, bb_size in zip(boxes3d, centroid, orientation, sizes):
            yield pcs, box3d, bb_center, bb_orientation, bb_size


def generate_voxel(pcs, box3d, bb_center, bb_orientation, bb_size):
    pcd, ids = extract_pc_in_box3d(pcs, box3d)
    voxels = voxels_from_point_cloud(pcd.astype(np.float64), bb_center, bb_orientation, bb_size)
    return torch.from_numpy(voxels)


def voxels_from_proposals(cfg, end_points, data, BATCH_PROPOSAL_IDs):
    device = end_points['center'].device
    dataset_config = cfg.eval_config['dataset_config']

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

    bbox_param = gather_bbox(dataset_config, gather_ids_p, *gather_param_p)

    all_voxels = [generate_voxel(*a) for a in unpack_data(data['point_clouds'][..., 0:3], *bbox_param)]
    return torch.stack(all_voxels).to(device)
