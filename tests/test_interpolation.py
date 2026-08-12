from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sg_rife.interpolation import interpolate_images


class LinearRuntime:
    device = torch.device("cpu")
    output_device = torch.device("cpu")
    dtype = torch.float32

    def __init__(self):
        self.seen_shapes = []

    def ensure_loaded(self, image_shape):
        self.image_shape = image_shape

    def extract_features(self, image):
        self.seen_shapes.append(tuple(image.shape))
        return None

    def interpolate(self, image0, image1, features0, features1, timestep, scale, tta):
        return image0 * (1.0 - timestep) + image1 * timestep


@pytest.mark.parametrize("multiplier", range(2, 9))
def test_interpolation_multipliers(multiplier):
    images = torch.stack((torch.zeros((3, 5, 3)), torch.ones((3, 5, 3))))

    result = interpolate_images(
        images, LinearRuntime(), multiplier=multiplier, scale=1.0, tta=True, include_last=True
    )

    assert result.shape == (multiplier + 1, 3, 5, 3)
    assert torch.allclose(result[:, 0, 0, 0], torch.linspace(0, 1, multiplier + 1))


def test_padding_and_crop():
    runtime = LinearRuntime()
    images = torch.rand((2, 65, 66, 3))

    result = interpolate_images(images, runtime, multiplier=2, scale=1.0, tta=False, include_last=True)

    assert runtime.seen_shapes == [(1, 3, 128, 128), (1, 3, 128, 128)]
    assert result.shape == (3, 65, 66, 3)
    assert torch.equal(result[0], images[0])
    assert torch.equal(result[-1], images[-1])


def test_chunk_without_last_frame():
    images = torch.stack((torch.zeros((2, 2, 3)), torch.ones((2, 2, 3))))

    result = interpolate_images(images, LinearRuntime(), multiplier=2, scale=1.0, tta=False, include_last=False)

    assert result.shape == (2, 2, 2, 3)
    assert torch.allclose(result[:, 0, 0, 0], torch.tensor([0.0, 0.5]))


def test_rejects_non_rgb_input():
    with pytest.raises(ValueError, match="RGB"):
        interpolate_images(
            torch.zeros((2, 4, 4, 4)),
            LinearRuntime(),
            multiplier=2,
            scale=1.0,
            tta=False,
            include_last=True,
        )


def test_rejects_multiplier_above_eight():
    with pytest.raises(ValueError, match="between 2 and 8"):
        interpolate_images(
            torch.zeros((2, 4, 4, 3)),
            LinearRuntime(),
            multiplier=9,
            scale=1.0,
            tta=False,
            include_last=True,
        )
