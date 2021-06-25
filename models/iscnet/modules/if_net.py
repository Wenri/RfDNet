import torch
import torch.nn as nn
from models.registers import MODULES
import torch.distributions as dist
from models.iscnet.modules.encoder_latent import Encoder_Latent
from models.iscnet.modules.occ_decoder import DecoderCBatchNorm
from torch.nn import functional as F
from external.common import make_3d_grid


@MODULES.register_module
class IFNet(nn.Module):
    def __init__(self, cfg, optim_spec=None, hidden_dim=256):
        super().__init__()
        '''Optimizer parameters used in training'''
        self.optim_spec = optim_spec

        '''Parameter Configs'''
        self.z_dim = cfg.config['data']['z_dim']
        dim = 3
        self.use_cls_for_completion = cfg.config['data']['use_cls_for_completion']
        if not cfg.config['data']['skip_propagate']:
            self.c_dim = self.use_cls_for_completion * cfg.dataset_config.num_class + 128
        else:
            self.c_dim = self.use_cls_for_completion * cfg.dataset_config.num_class + cfg.config['data']['c_dim']
        self.threshold = cfg.config['data']['threshold']

        '''Module Configs'''
        if self.z_dim != 0:
            self.encoder_latent = Encoder_Latent(dim=dim, z_dim=self.z_dim, c_dim=self.c_dim)
        else:
            self.encoder_latent = None

        '''Mount mesh generator'''
        if 'generation' in cfg.config and cfg.config['generation']['generate_mesh']:
            from models.iscnet.modules.generator import Generator3D
            self.generator = Generator3D(self,
                                         threshold=cfg.config['data']['threshold'],
                                         resolution0=cfg.config['generation']['resolution_0'],
                                         upsampling_steps=cfg.config['generation']['upsampling_steps'],
                                         sample=cfg.config['generation']['use_sampling'],
                                         refinement_step=cfg.config['generation']['refinement_step'],
                                         simplify_nfaces=cfg.config['generation']['simplify_nfaces'],
                                         preprocessor=None)

        # 128**3 res input
        self.conv_in = nn.Conv3d(1, 16, 3, padding=1, padding_mode='replicate')
        self.conv_0 = nn.Conv3d(16, 32, 3, padding=1, padding_mode='replicate')
        self.conv_0_1 = nn.Conv3d(32, 32, 3, padding=1, padding_mode='replicate')
        self.conv_1 = nn.Conv3d(32, 64, 3, padding=1, padding_mode='replicate')
        self.conv_1_1 = nn.Conv3d(64, 64, 3, padding=1, padding_mode='replicate')
        self.conv_2 = nn.Conv3d(64, 128, 3, padding=1, padding_mode='replicate')
        self.conv_2_1 = nn.Conv3d(128, 128, 3, padding=1, padding_mode='replicate')
        self.conv_3 = nn.Conv3d(128, 128, 3, padding=1, padding_mode='replicate')
        self.conv_3_1 = nn.Conv3d(128, 128, 3, padding=1, padding_mode='replicate')

        feature_size = (1 + 16 + 32 + 64 + 128 + 128) * 7
        hidden_size = feature_size

        self.fc_p = nn.Conv1d(dim, hidden_size, 1)

        if not self.z_dim == 0:
            self.fc_z = nn.Linear(self.z_dim, hidden_size)

        if not self.c_dim == 0:
            self.fc_c = nn.Linear(self.c_dim, hidden_size)

        self.fc_0 = nn.Conv1d(feature_size, hidden_dim, 1)
        self.fc_1 = nn.Conv1d(hidden_dim, hidden_dim, 1)
        self.fc_2 = nn.Conv1d(hidden_dim, hidden_dim, 1)
        self.fc_out = nn.Conv1d(hidden_dim, 1, 1)
        self.actvn = nn.ReLU()

        self.maxpool = nn.MaxPool3d(2)

        self.conv_in_bn = nn.BatchNorm3d(16)
        self.conv0_1_bn = nn.BatchNorm3d(32)
        self.conv1_1_bn = nn.BatchNorm3d(64)
        self.conv2_1_bn = nn.BatchNorm3d(128)
        self.conv3_1_bn = nn.BatchNorm3d(128)

        displacment = 0.0722
        displacments = []
        displacments.append([0, 0, 0])
        for x in range(3):
            for y in [-1, 1]:
                input = [0, 0, 0]
                input[x] = y * displacment
                displacments.append(input)

        self.displacments = torch.Tensor(displacments)

    def forward(self, p, z, c, x=None):
        net = self.fc_p(p.transpose(1, 2))

        if self.z_dim != 0:
            net_z = self.fc_z(z).unsqueeze(2)
            net = net + net_z

        if self.c_dim != 0:
            net_c = self.fc_c(c).unsqueeze(2)
            net = net + net_c

        features = net

        if x is not None:
            x = x.unsqueeze(1)

            p_features = features
            p = p.unsqueeze(1).unsqueeze(1)
            p = torch.cat([p + d for d in self.displacments.to(p.device)], dim=2)  # (B,1,7,num_samples,3)
            feature_0 = F.grid_sample(x, p, padding_mode='border')  # out : (B,C (of x), 1,1,sample_num)

            net = self.actvn(self.conv_in(x))
            net = self.conv_in_bn(net)
            feature_1 = F.grid_sample(net, p, padding_mode='border')  # out : (B,C (of x), 1,1,sample_num)
            net = self.maxpool(net)

            net = self.actvn(self.conv_0(net))
            net = self.actvn(self.conv_0_1(net))
            net = self.conv0_1_bn(net)
            feature_2 = F.grid_sample(net, p, padding_mode='border')  # out : (B,C (of x), 1,1,sample_num)
            net = self.maxpool(net)

            net = self.actvn(self.conv_1(net))
            net = self.actvn(self.conv_1_1(net))
            net = self.conv1_1_bn(net)
            feature_3 = F.grid_sample(net, p, padding_mode='border')  # out : (B,C (of x), 1,1,sample_num)
            net = self.maxpool(net)

            net = self.actvn(self.conv_2(net))
            net = self.actvn(self.conv_2_1(net))
            net = self.conv2_1_bn(net)
            feature_4 = F.grid_sample(net, p, padding_mode='border')
            net = self.maxpool(net)

            net = self.actvn(self.conv_3(net))
            net = self.actvn(self.conv_3_1(net))
            net = self.conv3_1_bn(net)
            feature_5 = F.grid_sample(net, p, padding_mode='border')

            # here every channel corresponds to one feature.

            features = torch.cat((feature_0, feature_1, feature_2, feature_3, feature_4, feature_5),
                                 dim=1)  # (B, features, 1,7,sample_num)
            shape = features.shape
            features = features.view(shape[0], shape[1] * shape[3], shape[4])  # (B, featues_per_sample, samples_num)
            features = features + p_features  # (B, featue_size, samples_num)

        net = self.actvn(self.fc_0(features))
        net = self.actvn(self.fc_1(net))
        net = self.actvn(self.fc_2(net))
        net = self.fc_out(net)
        out = net.squeeze(1)

        return out

    def compute_loss(self, input_features_for_completion, input_points_for_completion, input_points_occ_for_completion,
                     cls_codes_for_completion, voxel_grids, export_shape=False):
        '''
        Compute loss for OccNet
        :param input_features_for_completion (N_B x D): Number of bounding boxes x Dimension of proposal feature.
        :param input_points_for_completion (N_B, N_P, 3): Number of bounding boxes x Number of Points x 3.
        :param input_points_occ_for_completion (N_B, N_P): Corresponding occupancy values.
        :param cls_codes_for_completion (N_B, N_C): One-hot category codes.
        :param export_shape (bool): whether to export a shape voxel example.
        :return:
        '''
        device = input_features_for_completion.device
        batch_size = input_features_for_completion.size(0)
        if self.use_cls_for_completion:
            cls_codes_for_completion = cls_codes_for_completion.to(device).float()
            input_features_for_completion = torch.cat([input_features_for_completion, cls_codes_for_completion], dim=-1)

        kwargs = {}
        '''Infer latent code z.'''
        if self.z_dim > 0:
            q_z = self.infer_z(input_points_for_completion, input_points_occ_for_completion,
                               input_features_for_completion, device, **kwargs)
            z = q_z.rsample()
            # KL-divergence
            p0_z = self.get_prior_z(self.z_dim, device)
            kl = dist.kl_divergence(q_z, p0_z).sum(dim=-1)
            loss = kl.mean()
        else:
            z = torch.empty(size=(batch_size, 0), device=device)
            loss = 0.

        '''Decode to occupancy voxels.'''
        logits = self(input_points_for_completion, z, input_features_for_completion, voxel_grids)
        loss_i = F.binary_cross_entropy_with_logits(
            logits, input_points_occ_for_completion, reduction='none')
        loss = loss + loss_i.sum(-1).mean()

        '''Export Shape Voxels.'''
        if export_shape:
            shape = (16, 16, 16)
            p = make_3d_grid([-0.5 + 1 / 32] * 3, [0.5 - 1 / 32] * 3, shape).to(device)
            p = p.expand(batch_size, *p.size())
            z = self.get_z_from_prior((batch_size,), device, sample=False)
            kwargs = {}
            p_r = self.decode(p, z, input_features_for_completion, **kwargs)

            occ_hat = p_r.probs.view(batch_size, *shape)
            voxels_out = (occ_hat >= self.threshold)
        else:
            voxels_out = None

        return loss, voxels_out

    def get_z_from_prior(self, size=torch.Size([]), device='cuda', sample=False):
        ''' Returns z from prior distribution.

        Args:
            size (Size): size of z
            sample (bool): whether to sample
        '''
        p0_z = self.get_prior_z(self.z_dim, device)
        if sample:
            z = p0_z.sample(size)
        else:
            z = p0_z.mean
            z = z.expand(*size, *z.size())

        return z

    def decode(self, input_points_for_completion, z, features, voxel_grid):
        """ Returns occupancy probabilities for the sampled points.
        :param input_points_for_completion: points
        :param z: latent code z
        :param features: latent conditioned features
        :return:
        """
        logits = self(input_points_for_completion, z, features, voxel_grid)
        p_r = dist.Bernoulli(logits=logits)
        return p_r

    def infer_z(self, p, occ, c, device, **kwargs):
        '''
        Infers latent code z.
        :param p : points tensor
        :param occ: occupancy values for occ
        :param c: latent conditioned code c
        :param kwargs:
        :return:
        '''
        if self.encoder_latent is not None:
            mean_z, logstd_z = self.encoder_latent(p, occ, c, **kwargs)
        else:
            batch_size = p.size(0)
            mean_z = torch.empty(batch_size, 0).to(device)
            logstd_z = torch.empty(batch_size, 0).to(device)

        q_z = dist.Normal(mean_z, torch.exp(logstd_z))
        return q_z

    def get_prior_z(self, z_dim, device):
        ''' Returns prior distribution for latent code z.

        Args:
            zdim: dimension of latent code z.
            device (device): pytorch device
        '''
        p0_z = dist.Normal(
            torch.zeros(z_dim, device=device),
            torch.ones(z_dim, device=device)
        )

        return p0_z
