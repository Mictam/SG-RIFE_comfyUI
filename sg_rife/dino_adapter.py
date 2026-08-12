import torch
import torch.nn as nn


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


def init_fapm_weights(module):
    if isinstance(module, nn.Conv2d):
        kaiming_init(module)


class SqueezeExcitation(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        reduced = max(1, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, reduced, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(reduced, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.fc(self.pool(x))


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_ch,
            in_ch,
            kernel_size=kernel_size,
            padding=padding,
            groups=in_ch,
            bias=True,
        )
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=True)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.act(x)
        return x


class FAPM_Encoder(nn.Module):
    """
    Stage 1: Multi-Layer Pre-Warp Compression.
    Fuses info from multiple DINO layers using Shared/Specific decomposition.
    """

    def __init__(self, in_dim=384, num_layers=2, rank=256):
        super().__init__()

        # --- Stage 1: Dual-branch feature extraction ---
        self.shared_basis = nn.Conv2d(in_dim, rank, 1, bias=True)
        self.specific_bases = nn.ModuleList(
            [
                nn.Conv2d(in_dim, rank, kernel_size=1, bias=True)
                for _ in range(num_layers)
            ]
        )

        # --- FiLM parameter generators ---
        self.film_generators = nn.ModuleList(
            [
                nn.Conv2d(rank, rank * 2, kernel_size=1, bias=True)
                for _ in range(num_layers)
            ]
        )
        self.apply(init_fapm_weights)

    def forward(self, x_list):
        """
        Input: List of [2B, 384, H, W]
        Output: List of [2B, 256, H, W]
        """
        compressed_list = []

        for i, x in enumerate(x_list):
            # --- Stage 1: Get context features and main features ---
            z_shared = self.shared_basis(x)
            z_specific = self.specific_bases[i](x)

            # --- FiLM modulation process ---
            gamma_beta = self.film_generators[i](z_shared)
            gamma, beta = torch.chunk(gamma_beta, 2, dim=1)
            z_modulated = gamma * z_specific + beta

            compressed_list.append(z_modulated)

        return compressed_list


class FAPM_Refiner(nn.Module):
    """
    Stage 2: Post-Warp Refinement
    """

    def __init__(self, rank, out_ch_list):
        super().__init__()
        self.refinement_blocks = nn.ModuleList()
        self.shortcut_projections = nn.ModuleList()

        for oc in out_ch_list:
            # --- Refinement module backbone ---
            self.refinement_blocks.append(
                nn.Sequential(
                    nn.Conv2d(rank, oc, kernel_size=1, bias=True),
                    nn.GELU(),
                    DepthwiseSeparableConv(oc, oc),
                    nn.Conv2d(oc, oc, kernel_size=1, bias=True),
                    SqueezeExcitation(oc),
                )
            )

            # --- Shortcut branch ---
            # If refinement block input/output channel counts differ,
            #  need 1x1 conv to match dimensions
            if rank != oc:
                self.shortcut_projections.append(
                    nn.Conv2d(rank, oc, kernel_size=1, bias=True)
                )
            else:
                # If dimensions are the same, no operation needed
                self.shortcut_projections.append(nn.Identity())

        self.apply(init_fapm_weights)

    def forward(self, x_list):
        """
        Input: List of [2B, 256, H, W]
        Output: List of [2B, oc, H, W]
        """
        refined_list = []
        for i, x in enumerate(x_list):
            refined = self.refinement_blocks[i](x)

            shortcut = self.shortcut_projections[i](x)

            final_output = refined + shortcut

            refined_list.append(final_output)

        return refined_list
