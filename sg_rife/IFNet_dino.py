import torch
import torch.nn as nn
import torch.nn.functional as F
from .dino_config import DinoConfig
from .warplayer import warp
from .refine_dino import Unet, Contextnet
from .dino_adapter import FAPM_Encoder, FAPM_Refiner


def get_downsampled_flow(flow, target_h, target_w):
    """
    Downsamples 4-channel flow (t->0, t->1).
    """
    B, C, H, W = flow.shape
    flow_low = F.interpolate(flow, size=(target_h, target_w), mode="area")

    scale_w = target_w / W
    scale_h = target_h / H

    flow_low_scaled = torch.cat(
        [
            flow_low[:, 0:1] * scale_w,
            flow_low[:, 1:2] * scale_h,
            flow_low[:, 2:3] * scale_w,
            flow_low[:, 3:4] * scale_h,
        ],
        dim=1,
    )

    return flow_low_scaled


def deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
    return nn.Sequential(
        torch.nn.ConvTranspose2d(
            in_channels=in_planes,
            out_channels=out_planes,
            kernel_size=4,
            stride=2,
            padding=1,
        ),
        nn.PReLU(out_planes),
    )


def conv(
    in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1
):
    return nn.Sequential(
        nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=True,
        ),
        nn.PReLU(out_planes),
    )


class IFBlock(nn.Module):
    def __init__(self, in_planes, c=64):
        super(IFBlock, self).__init__()
        self.conv0 = nn.Sequential(
            conv(in_planes, c // 2, 3, 2, 1),
            conv(c // 2, c, 3, 2, 1),
        )
        self.convblock = nn.Sequential(
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
        )
        self.lastconv = nn.ConvTranspose2d(c, 5, 4, 2, 1)

    def forward(self, x, flow, scale):
        if scale != 1:
            x = F.interpolate(
                x,
                scale_factor=1.0 / scale,
                mode="bilinear",
                align_corners=False,
            )
        if flow is not None:
            flow = (
                F.interpolate(
                    flow,
                    scale_factor=1.0 / scale,
                    mode="bilinear",
                    align_corners=False,
                )
                * 1.0
                / scale
            )
            x = torch.cat((x, flow), 1)
        x = self.conv0(x)
        x = self.convblock(x) + x
        tmp = self.lastconv(x)
        tmp = F.interpolate(
            tmp, scale_factor=scale * 2, mode="bilinear", align_corners=False
        )
        flow = tmp[:, :4] * scale * 2
        mask = tmp[:, 4:5]
        return flow, mask


class IFNet(nn.Module):
    """
    IFNet integrating DINO semantics.
    """

    def __init__(self, dino_in_channels, dino_patch_size):
        super(IFNet, self).__init__()
        self.block0 = IFBlock(6, c=240)
        self.block1 = IFBlock(13 + 4, c=150)
        self.block2 = IFBlock(13 + 4, c=90)
        self.block_tea = IFBlock(16 + 4, c=90)
        self.contextnet = Contextnet()

        self.cfg = DinoConfig()
        self.dino_embed_dim = dino_in_channels
        self.dino_patch_size = dino_patch_size
        self.dino_compressor = FAPM_Encoder(
            in_dim=dino_in_channels,
            rank=self.cfg.compressor_rank,
            num_layers=2,
        )
        self.dino_refiner = FAPM_Refiner(
            rank=self.cfg.compressor_rank, out_ch_list=[128, 256]
        )

        self.unet = Unet()

    def forward(self, x, dino_feats, scale=[4, 2, 1], timestep=0.5):
        img0 = x[:, :3]
        img1 = x[:, 3:6]
        gt = x[:, 6:]  # In inference time, gt is None
        flow_list = []
        merged = []
        mask_list = []
        warped_img0 = img0
        warped_img1 = img1
        flow = None
        loss_distill = 0
        stu = [self.block0, self.block1, self.block2]
        for i in range(3):
            if flow is None:
                flow, mask = stu[i](
                    torch.cat((img0, img1), 1), None, scale=scale[i]
                )
            else:
                cat = torch.cat(
                    (img0, img1, warped_img0, warped_img1, mask), 1
                )
                flow_d, mask_d = stu[i](cat, flow, scale=scale[i])
                flow = flow + flow_d
                mask = mask + mask_d
            mask_list.append(torch.sigmoid(mask))
            flow_list.append(flow)
            warped_img0 = warp(img0, flow[:, :2])
            warped_img1 = warp(img1, flow[:, 2:4])
            merged_student = (warped_img0, warped_img1)
            merged.append(merged_student)
        if gt.shape[1] == 3:
            cat = torch.cat(
                (img0, img1, warped_img0, warped_img1, mask, gt), 1
            )
            flow_d, mask_d = self.block_tea(cat, flow, scale=1)
            flow_teacher = flow + flow_d
            warped_img0_teacher = warp(img0, flow_teacher[:, :2])
            warped_img1_teacher = warp(img1, flow_teacher[:, 2:4])
            mask_teacher = torch.sigmoid(mask + mask_d)
            merged_teacher = (
                warped_img0_teacher * mask_teacher
                + warped_img1_teacher * (1 - mask_teacher)
            )
        else:
            flow_teacher = None
            merged_teacher = None
        for i in range(3):
            merged[i] = merged[i][0] * mask_list[i] + merged[i][1] * (
                1 - mask_list[i]
            )
            if gt.shape[1] == 3:
                student_error = (merged[i] - gt).abs().mean(1, True)
                teacher_error = (merged_teacher - gt).abs().mean(1, True)
                loss_mask = (
                    (student_error > (teacher_error + 0.01)).float().detach()
                )
                flow_diff = flow_teacher.detach() - flow_list[i]
                flow_dist = (flow_diff**2).mean(1, True) ** 0.5
                loss_distill += (flow_dist * loss_mask).mean()

        c0 = self.contextnet(img0, flow[:, :2])
        c1 = self.contextnet(img1, flow[:, 2:4])

        final_flow = flow_list[2]  # [B, 4, H, W]

        d0_list, d1_list = dino_feats
        B, C_d, H_d, W_d = dino_feats[0][0].shape

        # List of [2B, C, H_d, W_d]
        combined_dino_inputs = []
        for feat0, feat1 in zip(d0_list, d1_list):
            combined_dino_inputs.append(torch.cat([feat0, feat1], dim=0))

        # List of [2B, C_comp, H_d, W_d]
        compressed_combined = self.dino_compressor(combined_dino_inputs)

        # [B, 4, H_d, W_d]
        flow_down = get_downsampled_flow(final_flow, H_d, W_d)

        # [2B, 2, H_d, W_d]
        flow_combined = torch.cat([flow_down[:, :2], flow_down[:, 2:4]], dim=0)

        warped_combined = []
        for feat in compressed_combined:
            # feat: [2B, C, H_d, W_d], flow_combined: [2B, 2, H_d, W_d]
            warped_combined.append(warp(feat, flow_combined))

        refined_combined = self.dino_refiner(warped_combined)

        dino_finals0 = []
        dino_finals1 = []
        B = x.shape[0]

        for r in refined_combined:
            # r [2B, C, H, W] split back to [B, C, H, W]
            r0, r1 = torch.split(r, B, dim=0)
            dino_finals0.append(r0)
            dino_finals1.append(r1)

        tmp, offsets = self.unet(
            img0,
            img1,
            warped_img0,
            warped_img1,
            mask,
            flow,
            c0,
            c1,
            dino_finals0,
            dino_finals1,
        )
        res = tmp[:, :3] * 2 - 1
        merged[2] = torch.clamp(merged[2] + res, 0, 1)
        return (
            flow_list,
            mask_list[2],
            merged,
            flow_teacher,
            merged_teacher,
            loss_distill,
            offsets,
        )
