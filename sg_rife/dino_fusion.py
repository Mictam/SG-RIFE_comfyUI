import torch
import torch.nn as nn
import torchvision.ops as ops


def kaiming_init(
    # Copyright (c) Meta Platforms, Inc. and affiliates.
    #
    # This software may be used and distributed in accordance with
    # the terms of the DINOv3 License Agreement.
    module: nn.Module,
    a: float = 0,
    mode: str = "fan_out",
    nonlinearity: str = "relu",
    bias: float = 0,
    distribution: str = "normal",
) -> None:
    assert distribution in ["uniform", "normal"]
    if hasattr(module, "weight") and module.weight is not None:
        if distribution == "uniform":
            nn.init.kaiming_uniform_(
                module.weight, a=a, mode=mode, nonlinearity=nonlinearity
            )
        else:
            nn.init.kaiming_normal_(
                module.weight, a=a, mode=mode, nonlinearity=nonlinearity
            )
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def constant_init(module, val, bias=0):
    # Copyright (c) Meta Platforms, Inc. and affiliates.
    #
    # This software may be used and distributed in accordance with
    # the terms of the DINOv3 License Agreement.
    if hasattr(module, "weight") and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


class DinoFusion(nn.Module):
    """
    Reference: PCDAlignment & ModulatedDeformConvPack
    Mechanism:
    1. Align: Predict offsets from (RIFE, DINO) mismatch.
    2. Warp: Use Deformable Conv to align DINO to RIFE.
    3. Gate: Use Sigmoid mask to suppress invalid/occluded regions.
    4. Inject: Residual addition.
    """

    def __init__(self, dim, groups=4, bias=True):
        super(DinoFusion, self).__init__()

        self.to_q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)  # RIFE
        self.to_k = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)  # DINO
        self.to_v = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)  # DINO
        self.groups = groups

        # Offset & Mask Predictor
        # Channels = groups * (2 offsets + 1 mask) * (kernel_size^2)
        #          = groups * 27
        self.conv_offset = nn.Conv2d(
            dim * 2, 27 * self.groups, kernel_size=3, padding=1, bias=True
        )

        # Deformable Convolution Kernel
        self.dcn_weight = nn.Parameter(
            torch.empty(dim, dim // self.groups, 3, 3)
        )

        self.gamma = nn.Parameter(torch.zeros(1))

        self._init_weights()

    def _init_weights(self):
        constant_init(self.conv_offset, 0, 0)
        kaiming_init(self.to_q)
        kaiming_init(self.to_k)
        kaiming_init(self.to_v)
        nn.init.kaiming_normal_(
            self.dcn_weight, mode="fan_out", nonlinearity="relu"
        )
        constant_init(self.gamma, 0, 0)

    def forward(self, rife_feat, dino_feats):
        """
        Args:
            rife_feat: [B, C, H, W]
            dino_feats:  2*[B, C, H, W]
        """
        # Q = Pixel Context, K = Semantic Reference
        d0, d1 = dino_feats
        Q_stack = torch.cat([rife_feat, rife_feat], dim=0)  # [2B, C, H, W]
        D_stack = torch.cat([d0, d1], dim=0)  # [2B, C, H, W]

        Q = self.to_q(Q_stack)
        K = self.to_k(D_stack)
        V = self.to_v(D_stack)

        combined = torch.cat([Q, K], dim=1)
        out = self.conv_offset(combined)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)

        aligned_stack = ops.deform_conv2d(
            input=V,
            offset=offset,
            weight=self.dcn_weight,
            mask=mask,
            padding=1,
            stride=1,
        )

        aligned_0, aligned_1 = torch.chunk(aligned_stack, 2, dim=0)
        aligned_combined = aligned_0 + aligned_1

        offset_0, offset_1 = torch.chunk(offset, 2, dim=0)

        return rife_feat + self.gamma * aligned_combined, [offset_0, offset_1]
