# Base data of networks
# author: ynie
# date: Feb, 2020
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset

from utils.read_and_write import read_json

roty = np.array([[0., 0., -1.], [0., 1., 0.], [1., 0., 0.]])


class ScanNet(Dataset):
    def __init__(self, cfg, mode):
        """
        initiate SUNRGBD dataset for data loading
        :param cfg: config file
        :param mode: train/val/test mode
        """
        self.config = cfg.config
        self.dataset_config = cfg.dataset_config
        self.mode = mode
        # mode = 'train'
        split_file = os.path.join(cfg.config['data']['split'], 'scannetv2_' + mode + '.json')
        self.split = read_json(split_file)

    def __len__(self):
        return len(self.split)


class ABNormalDataset(ScanNet):
    """ 3D Occupancy ABNormal dataset class.
    """

    def __init__(self, cfg, mode):
        super(ABNormalDataset, self).__init__(cfg, mode)
        self.data_dir = Path(cfg.config['data']['abnormal_path'])
        self.npz_maps = defaultdict(dict)
        mapname, mapdict = self.get_maptabel()
        for d in self.data_dir.glob('*/gen/scan_*/*_output.npz'):
            ins_id, shapenet_id = d.name.split('_')[:2]
            scene_str = mapname[d.parent.name.split('_')[1]]
            overscan = mapdict[scene_str][(int(ins_id), shapenet_id[:8])]
            self.npz_maps[scene_str][int(ins_id)] = (shapenet_id, d, float(overscan))

        self.rand = np.random.default_rng()
        self.OCCN = 100000

    def get_maptabel(self):
        maptab = self.data_dir / 'maptable.txt'
        mapdict = defaultdict(dict)
        mapname = dict()
        with maptab.open('r') as f:
            for line in f:
                line = line.strip()
                if line:
                    scan_str, scene_str, _, overscan_str = line.split(':')
                    scan_id, ins_id, shapenet_id = scan_str.split('_')
                    scan_id = scan_id[4:]
                    scene_str = '_'.join(scene_str.split('_')[:2])
                    scene_str_chk = mapname.get(scan_id)
                    overscan_str = overscan_str.split('_')[1].strip()
                    if scene_str_chk is None:
                        mapname[scan_id] = scene_str
                    else:
                        assert scene_str_chk == scene_str
                    mapdict[scene_str][(int(ins_id), shapenet_id)] = overscan_str
        return mapname, mapdict

    def subsample(self, points, N):
        total = points.shape[0]
        indices = self.rand.permutation(total)
        if indices.shape[0] < N:
            indices = np.concatenate([indices, self.rand.integers(total, size=N - total)])
        indices = indices[:N]
        return points[indices]

    def get_scannet_abnormal(self, scene_name, idx, shapenet_catid, shapenet_id):
        scene_dict = self.npz_maps[scene_name]
        path = scene_dict.get(int(idx))
        if path is None:
            return None
        assert path[0] == shapenet_id
        p, occ = self.loadnpz(path[1])
        return p, occ, path[2]

    def loadnpz(self, npz_file):
        npz_file = np.load(npz_file)
        pts = npz_file['pts']
        pts_mask = npz_file['pts_mask']
        inpts = pts[np.all(pts_mask, axis=-1)]
        outpts = pts[~pts_mask[:, 0] & pts_mask[:, 1]]

        n_in = self.OCCN // 2
        inpts = self.subsample(inpts, n_in)
        outpts = self.subsample(outpts, self.OCCN - n_in)
        indices = self.rand.permutation(self.OCCN)
        pts = np.concatenate((inpts, outpts), axis=0)
        pts_mask = np.zeros(self.OCCN, dtype=np.bool_)
        pts_mask[:n_in] = True

        return pts[indices] @ roty, pts_mask[indices]
