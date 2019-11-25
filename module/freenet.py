import torch.nn as nn
import torch.nn.functional as F
from simplecv.interface import CVModule
from simplecv.module import SEBlock
from simplecv import registry
import torch
import math


def conv3x3_gn_relu(in_channel, out_channel, num_group):
    return nn.Sequential(
        nn.Conv2d(in_channel, out_channel, 3, 1, 1),
        nn.GroupNorm(num_group, out_channel),
        nn.ReLU(inplace=True),
    )


def downsample2x(in_channel, out_channel):
    return nn.Sequential(
        nn.Conv2d(in_channel, out_channel, 3, 2, 1),
        nn.ReLU(inplace=True)
    )


def repeat_block(block_channel, r, n):
    layers = [
        nn.Sequential(
            SEBlock(block_channel, r),
            conv3x3_gn_relu(block_channel, block_channel, r)
        )
        for _ in range(n)]
    return nn.Sequential(*layers)


@registry.MODEL.register('FreeNet')
class FreeNet(CVModule):
    def __init__(self, config):
        super(FreeNet, self).__init__(config)
        r = int(16 * self.config.reduction_ratio)
        block1_channels = int(self.config.block_channels[0] * self.config.reduction_ratio / r) * r
        block2_channels = int(self.config.block_channels[1] * self.config.reduction_ratio / r) * r
        block3_channels = int(self.config.block_channels[2] * self.config.reduction_ratio / r) * r
        block4_channels = int(self.config.block_channels[3] * self.config.reduction_ratio / r) * r

        self.feature_ops = nn.ModuleList([
            conv3x3_gn_relu(self.config.in_channels, block1_channels, r),

            repeat_block(block1_channels, r, self.config.num_blocks[0]),
            nn.Identity(),
            downsample2x(block1_channels, block2_channels),

            repeat_block(block2_channels, r, self.config.num_blocks[1]),
            nn.Identity(),
            downsample2x(block2_channels, block3_channels),

            repeat_block(block3_channels, r, self.config.num_blocks[2]),
            nn.Identity(),
            downsample2x(block3_channels, block4_channels),

            repeat_block(block4_channels, r, self.config.num_blocks[3]),
            nn.Identity(),
        ])
        inner_dim = int(self.config.inner_dim * self.config.reduction_ratio)
        self.reduce_1x1convs = nn.ModuleList([
            nn.Conv2d(block1_channels, inner_dim, 1),
            nn.Conv2d(block2_channels, inner_dim, 1),
            nn.Conv2d(block3_channels, inner_dim, 1),
            nn.Conv2d(block4_channels, inner_dim, 1),
        ])
        self.fuse_3x3convs = nn.ModuleList([
            nn.Conv2d(inner_dim, inner_dim, 3, 1, 1),
            nn.Conv2d(inner_dim, inner_dim, 3, 1, 1),
            nn.Conv2d(inner_dim, inner_dim, 3, 1, 1),
            nn.Conv2d(inner_dim, inner_dim, 3, 1, 1),
        ])
        self.cls_pred_conv = nn.Conv2d(inner_dim, self.config.num_classes, 1)

    def top_down(self, top, lateral):
        top2x = F.interpolate(top, scale_factor=2.0, mode='nearest')
        return lateral + top2x

    def forward(self, x, y=None, w=None, **kwargs):
        feat_list = []
        for op in self.feature_ops:
            x = op(x)
            if isinstance(op, nn.Identity):
                feat_list.append(x)

        inner_feat_list = [self.reduce_1x1convs[i](feat) for i, feat in enumerate(feat_list)]
        inner_feat_list.reverse()

        out_feat_list = [self.fuse_3x3convs[0](inner_feat_list[0])]
        for i in range(len(inner_feat_list) - 1):
            inner = self.top_down(out_feat_list[i], inner_feat_list[i + 1])
            out = self.fuse_3x3convs[i](inner)
            out_feat_list.append(out)

        final_feat = out_feat_list[-1]

        logit = self.cls_pred_conv(final_feat)
        if self.training:
            loss_dict = {
                'cls_loss': self.loss(logit, y, w)
            }
            return loss_dict

        return torch.softmax(logit, dim=1)

    def loss(self, x, y, weight):
        losses = F.cross_entropy(x, y.long() - 1, weight=None,
                                 ignore_index=-1, reduction='none')

        v = losses.mul_(weight).sum() / weight.sum()
        return v

    def set_defalut_config(self):
        self.config.update(dict(
            in_channels=204,
            num_classes=16,
            block_channels=(96, 128, 192, 256),
            num_blocks=(1, 1, 1, 1),
            inner_dim=128,
            reduction_ratio=1.0,
        ))
