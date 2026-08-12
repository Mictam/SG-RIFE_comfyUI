import math

import torch
import torch.nn.functional as F


def _interpolate_interval(runtime, image0, image1, features0, features1, depth, scale, tta):
    if depth == 0:
        return []

    midpoint = runtime.interpolate(
        image0, image1, features0, features1, timestep=0.5, scale=scale, tta=tta
    )
    if depth == 1:
        return [midpoint]

    midpoint_features = runtime.extract_features(midpoint)
    return [
        *_interpolate_interval(
            runtime, image0, midpoint, features0, midpoint_features, depth - 1, scale, tta
        ),
        midpoint,
        *_interpolate_interval(
            runtime, midpoint, image1, midpoint_features, features1, depth - 1, scale, tta
        ),
    ]


def _resample_interval(frames, source_multiplier, target_multiplier):
    result = []
    for index in range(1, target_multiplier):
        position, remainder = divmod(index * source_multiplier, target_multiplier)
        if remainder == 0:
            result.append(frames[position])
        else:
            result.append(torch.lerp(frames[position], frames[position + 1], remainder / target_multiplier))
    return result


def interpolate_images(images, runtime, multiplier=2, scale=1.0, tta=True, include_last=True):
    if images.ndim != 4 or images.shape[-1] != 3:
        raise ValueError("SG-RIFE expects an RGB IMAGE batch")
    if multiplier < 2 or multiplier > 8:
        raise ValueError("SG-RIFE multiplier must be between 2 and 8")
    if scale <= 0:
        raise ValueError("SG-RIFE scale must be positive")
    if images.shape[0] < 2:
        return images

    runtime.ensure_loaded(images.shape)

    height, width = images.shape[1:3]
    padded_height = math.ceil(height / 64) * 64
    padded_width = math.ceil(width / 64) * 64
    pad_height = padded_height - height
    pad_width = padded_width - width
    top = pad_height // 2
    left = pad_width // 2
    padding = (left, pad_width - left, top, pad_height - top)
    source_multiplier = 1 << (multiplier - 1).bit_length()
    depth = source_multiplier.bit_length() - 1

    def prepare(frame):
        frame = frame.movedim(-1, 0).unsqueeze(0).to(device=runtime.device, dtype=runtime.dtype)
        return F.pad(frame, padding, mode="replicate")

    def crop(frame):
        return frame[0, :, top : top + height, left : left + width].movedim(0, -1).float().to(runtime.output_device)

    result = []
    image0 = prepare(images[0])
    features0 = runtime.extract_features(image0)

    for index in range(images.shape[0] - 1):
        image1 = prepare(images[index + 1])
        features1 = runtime.extract_features(image1)
        mids = _interpolate_interval(
            runtime, image0, image1, features0, features1, depth, scale, tta
        )
        mids = _resample_interval([image0, *mids, image1], source_multiplier, multiplier)
        result.append(images[index].float().to(runtime.output_device))
        result.extend(crop(midpoint) for midpoint in mids)
        image0 = image1
        features0 = features1

    if include_last:
        result.append(images[-1].float().to(runtime.output_device))

    return torch.stack(result).clamp_(0.0, 1.0)
