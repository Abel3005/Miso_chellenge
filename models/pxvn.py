import torch
import torch.nn as nn
import torch.nn.functional as F
from cnn.pspnet import PSPNet
import pytorch_utils as pt_utils


psp_models = {
    'resnet18': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet18'),
    'resnet34': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet34'),
    'resnet50': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet50'),
}


class PXVN(nn.Module):
    def __init__(
        self, n_classes, n_kps=8
    ):
        super().__init__()

        # ######################## prepare stages#########################
        self.n_cls = n_classes
        self.n_kps = n_kps
        cnn = psp_models['resnet34'.lower()]()

        self.cnn_pre_stages = nn.Sequential(
            cnn.feats.conv1,  # stride = 2, [bs, c, 240, 320]
            cnn.feats.bn1, cnn.feats.relu,
            cnn.feats.maxpool  # stride = 2, [bs, 64, 120, 160]
        )
        # ####################### downsample stages#######################
        self.cnn_ds_stages = nn.ModuleList([
            cnn.feats.layer1,    # stride = 1, [bs, 64, 120, 160]
            cnn.feats.layer2,    # stride = 2, [bs, 128, 60, 80]
            # stride = 1, [bs, 128, 60, 80]
            nn.Sequential(cnn.feats.layer3, cnn.feats.layer4),
            nn.Sequential(cnn.psp, cnn.drop_1)   # [bs, 1024, 60, 80]
        ])
        self.ds_sr = [4, 8, 8, 8]
        self.ds_rgb_oc = [64, 128, 512, 1024]
        #-------------------------------------------#
        # ###################### upsample stages #############################
        self.cnn_up_stages = nn.ModuleList([
            nn.Sequential(cnn.up_1, cnn.drop_2),  # [bs, 256, 120, 160]
            nn.Sequential(cnn.up_2, cnn.drop_2),  # [bs, 64, 240, 320]
            nn.Sequential(cnn.final),  # [bs, 64, 240, 320]
            nn.Sequential(cnn.up_3, cnn.final)  # [bs, 64, 480, 640]
        ])
        self.up_rgb_oc = [256, 64, 64]
        #-----------------------------------------#
        # ####################### prediction headers #############################
        # We use 3D keypoint prediction header for pose estimation following PVN3D
        # You can use different prediction headers for different downstream tasks.

        self.rgb_seg_layer = (
            pt_utils.Seq(self.up_rgb_oc[-1])
            .conv1d(128, bn=True, activation=nn.ReLU())
            .conv1d(128, bn=True, activation=nn.ReLU())
            .conv1d(128, bn=True, activation=nn.ReLU())
            .conv1d(n_classes, activation=None)
        )

        self.ctr_ofst_layer = (
            pt_utils.Seq(self.up_rgb_oc[-1])
            .conv1d(128, bn=True, activation=nn.ReLU())
            .conv1d(128, bn=True, activation=nn.ReLU())
            .conv1d(128, bn=True, activation=nn.ReLU())
            .conv1d(3, activation=None)
        )

        self.kp_ofst_layer = (
            pt_utils.Seq(self.up_rgb_oc[-1])
            .conv1d(128, bn=True, activation=nn.ReLU())
            .conv1d(128, bn=True, activation=nn.ReLU())
            .conv1d(128, bn=True, activation=nn.ReLU())
            .conv1d(n_kps*3, activation=None)
        )

    @staticmethod
    def random_sample(feature, pool_idx):
        """
        :param feature: [B, N, d] input features matrix
        :param pool_idx: [B, N', max_num] N' < N, N' is the selected position after pooling
        :return: pool_features = [B, N', d] pooled features matrix
        """
        if len(feature.size()) > 3:
            feature = feature.squeeze(dim=3)  # batch*channel*npoints
        num_neigh = pool_idx.shape[-1]
        d = feature.shape[1]
        batch_size = pool_idx.shape[0]
        pool_idx = pool_idx.reshape(batch_size, -1)  # batch*(npoints,nsamples)
        pool_features = torch.gather(
            feature, 2, pool_idx.unsqueeze(1).repeat(1, feature.shape[1], 1)
        ).contiguous()
        pool_features = pool_features.reshape(batch_size, d, -1, num_neigh)
        pool_features = pool_features.max(dim=3, keepdim=True)[0]  # batch*channel*npoints*1
        return pool_features

    def forward(
        self, inputs, end_points=None, scale=1,
    ):
        """
        Params:
        inputs: dict of :
            rgb         : FloatTensor [bs, 3, h, w]
        Returns:
            end_points:
        """
        # ###################### prepare stages #############################
        if not end_points:
            end_points = {}
        # ResNet pre + layer1 + layer2
        rgb_emb = self.cnn_pre_stages(inputs['rgb'])  # stride = 2, [bs, c, 240, 320]
        #--------------------------------------------------------#

        # ###################### encoding stages #############################
        ds_emb = []
        for i_ds in range(4):
            # encode rgb downsampled feature
            rgb_emb = self.cnn_ds_stages[i_ds](rgb_emb)
        # ###################### decoding stages #############################
        n_up_layers = len(self.cnn_up_stages)
        for i_up in range(n_up_layers-1):
            # decode rgb upsampled feature
            rgb_emb = self.cnn_up_stages[i_up](rgb_emb)
        # final upsample layers:
        rgb_emb = self.cnn_up_stages[n_up_layers-1](rgb_emb)
        # ###################### prediction stages #############################
        rgb_segs = self.rgb_seg_layer(rgb_emb)
        pred_kp_ofs = self.kp_ofst_layer(rgb_emb)
        pred_ctr_ofs = self.ctr_ofst_layer(rgb_emb)

        pred_kp_ofs = pred_kp_ofs.view(
            bs, self.n_kps, 3, -1
        ).permute(0, 1, 3, 2).contiguous()
        pred_ctr_ofs = pred_ctr_ofs.view(
            bs, 1, 3, -1
        ).permute(0, 1, 3, 2).contiguous()

        # return rgbd_seg, pred_kp_of, pred_ctr_of
        end_points['pred_rgb_segs'] = rgb_segs
        end_points['pred_kp_ofs'] = pred_kp_ofs
        end_points['pred_ctr_ofs'] = pred_ctr_ofs
        return end_points


# Copy from PVN3D: https://github.com/ethnhe/PVN3D


def main():
    from common import ConfigRandLA
    rndla_cfg = ConfigRandLA
    n_cls = 22
    model = PXVN(n_cls)
    print(model)

    print(
        "model parameters:", sum(param.numel() for param in model.parameters())
    )

if __name__ == "__main__":
    main()
