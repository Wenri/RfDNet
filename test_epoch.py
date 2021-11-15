# Testing functions.
# author: ynie
# date: April, 2020
from time import time
from types import SimpleNamespace

import numpy as np

from models.iscnet.modules import ISCNet
from net_utils.ap_helper import APCalculator
from net_utils.utils import LossRecorder
from net_utils.voxel_util import get_bbox, voxels_from_scannet


def export_mesh(cfg, est_data, data, dump_threshold=0.5):
    BATCH_PROPOSAL_IDs = est_data[3]
    eval_dict = est_data[4]
    meshes = est_data[5]
    parsed_predictions = est_data[7]
    pred_mask = eval_dict['pred_mask']
    obj_prob = parsed_predictions['obj_prob']
    bsize, N_proposals = pred_mask.shape
    for i in range(bsize):
        c = SimpleNamespace(**{
            k: v[i] for k, v in get_bbox(cfg.eval_config['dataset_config'], **data).items()})
        for j in range(N_proposals):
            if not (pred_mask[i, j] == 1 and obj_prob[i, j] > dump_threshold):
                continue
            # get index
            idx = BATCH_PROPOSAL_IDs[i, :, 0].tolist().index(j)
            oid = BATCH_PROPOSAL_IDs[i, idx, 1].item()

            if ISCNet.cat_set is not None and c.shapenet_catids[oid] not in ISCNet.cat_set:
                continue

            # get mesh points
            mesh_data = meshes[idx]

            # get scan points
            ins_id = c.object_instance_labels[oid]
            ins_pc = c.point_clouds[c.point_instance_labels == ins_id, :3].cuda()

            voxels, ins_pc, overscan = voxels_from_scannet(ins_pc, c.box_centers[oid].cuda(),
                                                           c.box_sizes[oid].cuda(),
                                                           c.axis_rectified[oid].cuda())

            scan_idx = c.scan_idx.item()
            mesh_data.export(f'/workspace/Export/{scan_idx}_{oid}_{idx}_mesh.ply')
    return


def test_func(cfg, tester, test_loader):
    """
    test function.
    :param cfg: configuration file
    :param tester: specific tester for networks
    :param test_loader: dataloader for testing
    :return:
    """
    mode = cfg.config['mode']
    batch_size = cfg.config[mode]['batch_size']
    loss_recorder = LossRecorder(batch_size)
    AP_IOU_THRESHOLDS = cfg.config[mode]['ap_iou_thresholds']
    evaluate_mesh_mAP = True if cfg.config[mode]['phase'] == 'completion' and cfg.config['generation'][
        'generate_mesh'] and cfg.config[mode]['evaluate_mesh_mAP'] else False
    ap_calculator_list = [APCalculator(iou_thresh, cfg.dataset_config.class2type, evaluate_mesh_mAP) for iou_thresh in
                          AP_IOU_THRESHOLDS]
    cfg.log_string('-' * 100)
    total_cds = {}
    for iter, data in enumerate(test_loader):
        loss, est_data = tester.test_step(data)
        export_mesh(cfg, est_data, data)
        eval_dict = est_data[4]
        for ap_calculator in ap_calculator_list:
            ap_calculator.step(eval_dict['batch_pred_map_cls'], eval_dict['batch_gt_map_cls'])
        # visualize intermediate results.
        # if cfg.config['generation']['dump_results']:
        #     tester.visualize_step(mode, iter, data, est_data, eval_dict)
        total_cds.update(est_data[8])
        loss_recorder.update_loss(loss)

        if ((iter + 1) % cfg.config['log']['print_step']) == 0:
            cfg.log_string('Process: Phase: %s. Epoch %d: %d/%d. Current loss: %s.' % (
                mode, 0, iter + 1, len(test_loader), str({key: np.mean(item) for key, item in loss.items()})))

    total_n = 0
    total_cd = 0
    for x in total_cds.values():
        total_n += len(x)
        total_cd += sum(x.values())

    cd_value = total_cd / total_n
    print(f'{cd_value=}')
    return loss_recorder.loss_recorder, ap_calculator_list


def test(cfg, tester, test_loader):
    '''
    train epochs for network
    :param cfg: configuration file
    :param tester: specific tester for networks
    :param test_loader: dataloader for testing
    :return:
    '''
    cfg.log_string('-' * 100)
    # set mode
    mode = cfg.config['mode']
    tester.net.train(mode == 'train')
    start = time()
    test_loss_recoder, ap_calculator_list = test_func(cfg, tester, test_loader)
    cfg.log_string('Test time elapsed: (%f).' % (time() - start))
    for key, test_loss in test_loss_recoder.items():
        cfg.log_string('Test loss (%s): %f' % (key, test_loss.avg))

    # Evaluate average precision
    AP_IOU_THRESHOLDS = cfg.config[mode]['ap_iou_thresholds']
    for i, ap_calculator in enumerate(ap_calculator_list):
        cfg.log_string(('-' * 10 + 'iou_thresh: %f' + '-' * 10) % (AP_IOU_THRESHOLDS[i]))
        metrics_dict = ap_calculator.compute_metrics()
        for key in metrics_dict:
            cfg.log_string('eval %s: %f' % (key, metrics_dict[key]))
