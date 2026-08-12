import torch
import torch.nn.functional as F


def warp(image: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    batch, _, height, width = flow.shape
    horizontal = torch.linspace(-1.0, 1.0, width, device=flow.device, dtype=flow.dtype)
    vertical = torch.linspace(-1.0, 1.0, height, device=flow.device, dtype=flow.dtype)
    grid_y, grid_x = torch.meshgrid(vertical, horizontal, indexing="ij")
    base_grid = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0).expand(batch, -1, -1, -1)
    normalized_flow = torch.stack(
        (
            flow[:, 0] * (2.0 / (image.shape[3] - 1.0)),
            flow[:, 1] * (2.0 / (image.shape[2] - 1.0)),
        ),
        dim=-1,
    )
    return F.grid_sample(
        image,
        base_grid + normalized_flow,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )

