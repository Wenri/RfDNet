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
    def __init__(self, hidden_dim=256):
        super().__init__()
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

        self.displacments = torch.Tensor(displacments).cuda()

    def forward(self, p, x):
        x = x.unsqueeze(1)

        p_features = p.transpose(1, -1)
        p = p.unsqueeze(1).unsqueeze(1)
        p = torch.cat([p + d for d in self.displacments], dim=2)  # (B,1,7,num_samples,3)
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
        features = torch.reshape(features,
                                 (shape[0], shape[1] * shape[3], shape[4]))  # (B, featues_per_sample, samples_num)
        # features = torch.cat((features, p_features), dim=1)  # (B, featue_size, samples_num)

        net = self.actvn(self.fc_0(features))
        net = self.actvn(self.fc_1(net))
        net = self.actvn(self.fc_2(net))
        net = self.fc_out(net)
        out = net.squeeze(1)

        return out
