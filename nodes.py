import importlib
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn

import comfy.model_management
import comfy.model_patcher
import comfy.utils

from .sg_rife.IFNet_dino import IFNet
from .sg_rife.interpolation import interpolate_images


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_DINOV3_REPO = Path(os.environ.get("SG_RIFE_DINOV3_REPO", REPO_ROOT / "models" / "dinov3_repo"))
DEFAULT_DINO_CHECKPOINT = Path(
    os.environ.get(
        "SG_RIFE_DINO_CHECKPOINT",
        REPO_ROOT / "models" / "dino" / "dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
    )
)
DEFAULT_SG_RIFE_CHECKPOINT = Path(
    os.environ.get("SG_RIFE_CHECKPOINT", REPO_ROOT / "models" / "sg_rife" / "flownet.pkl")
)


class SGRIFEModule(nn.Module):
    def __init__(self, dino, flownet):
        super().__init__()
        self.dino = dino
        self.flownet = flownet

    def extract_features(self, image):
        return self.dino.get_features(image)

    def interpolate(self, image0, image1, features0, features1, scale, tta):
        scale_list = [4.0 / scale, 2.0 / scale, 1.0 / scale]
        images = torch.cat((image0, image1), dim=1)
        _, _, merged, _, _, _, _ = self.flownet(images, (features0, features1), scale_list, timestep=0.5)
        result = merged[2]

        if tta:
            flipped_features = (
                [feature.flip(2, 3) for feature in features0],
                [feature.flip(2, 3) for feature in features1],
            )
            _, _, flipped, _, _, _, _ = self.flownet(
                images.flip(2, 3), flipped_features, scale_list, timestep=0.5
            )
            result = (result + flipped[2].flip(2, 3)) * 0.5

        return result


class DinoFeatureExtractor(nn.Module):
    def __init__(self, model, interaction_indices):
        super().__init__()
        self.model = model
        self.interaction_indices = interaction_indices
        self.patch_size = model.patch_size
        self.embed_dim = model.embed_dim
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)

    def get_features(self, image):
        return [
            feature.detach()
            for feature in self.model.get_intermediate_layers(
                (image - self.mean) / self.std,
                n=self.interaction_indices,
                reshape=True,
            )
        ]


@dataclass
class SGRIFERuntime:
    patcher: comfy.model_patcher.ModelPatcher
    dtype: torch.dtype

    @property
    def device(self):
        return self.patcher.load_device

    @property
    def output_device(self):
        return comfy.model_management.intermediate_device()

    def ensure_loaded(self, image_shape):
        _, height, width, _ = image_shape
        padded_height = math.ceil(height / 64) * 64
        padded_width = math.ceil(width / 64) * 64
        dtype_size = comfy.model_management.dtype_size(self.dtype)
        activation_memory = padded_height * padded_width * 3 * dtype_size * 192
        comfy.model_management.load_models_gpu([self.patcher], memory_required=activation_memory)

    def extract_features(self, image):
        return self.patcher.model.extract_features(image)

    def interpolate(self, image0, image1, features0, features1, timestep, scale, tta):
        if timestep != 0.5:
            raise ValueError("SG-RIFE only supports midpoint interpolation")
        return self.patcher.model.interpolate(image0, image1, features0, features1, scale, tta)


def _check_paths(dinov3_repo, dino_checkpoint, sg_rife_checkpoint):
    required = [dinov3_repo / "hubconf.py", dino_checkpoint, sg_rife_checkpoint]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing SG-RIFE files:\n" + "\n".join(missing))


def load_sg_rife(
    precision,
    dinov3_repo=DEFAULT_DINOV3_REPO,
    dino_checkpoint=DEFAULT_DINO_CHECKPOINT,
    sg_rife_checkpoint=DEFAULT_SG_RIFE_CHECKPOINT,
):
    dinov3_repo = Path(dinov3_repo)
    dino_checkpoint = Path(dino_checkpoint)
    sg_rife_checkpoint = Path(sg_rife_checkpoint)
    _check_paths(dinov3_repo, dino_checkpoint, sg_rife_checkpoint)

    repo_path = str(dinov3_repo)
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)
    dinov3_vits16 = importlib.import_module("dinov3.hub.backbones").dinov3_vits16

    dino_model = dinov3_vits16(pretrained=False)
    dino_model.load_state_dict(comfy.utils.load_torch_file(str(dino_checkpoint), safe_load=True))
    dino = DinoFeatureExtractor(dino_model, interaction_indices=[8, 11])
    flownet = IFNet(dino_in_channels=dino.embed_dim, dino_patch_size=dino.patch_size)

    state_dict = comfy.utils.load_torch_file(str(sg_rife_checkpoint), safe_load=True)
    state_dict = {key[7:] if key.startswith("module.") else key: value for key, value in state_dict.items()}
    flownet.load_state_dict(state_dict)

    dtype = torch.bfloat16 if precision == "bf16" else torch.float32
    model = SGRIFEModule(dino, flownet).eval().to(dtype=dtype)
    patcher = comfy.model_patcher.CoreModelPatcher(
        model,
        load_device=comfy.model_management.get_torch_device(),
        offload_device=comfy.model_management.unet_offload_device(),
    )
    return SGRIFERuntime(patcher, dtype)


class SGRIFELoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"precision": (["fp32", "bf16"], {"default": "fp32"})}}

    RETURN_TYPES = ("SG_RIFE_MODEL",)
    RETURN_NAMES = ("sg_rife",)
    FUNCTION = "load"
    CATEGORY = "model/loaders"

    def load(self, precision):
        return (load_sg_rife(precision),)


class SGRIFEInterpolate:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sg_rife": ("SG_RIFE_MODEL",),
                "images": ("IMAGE",),
                "multiplier": ("INT", {"default": 2, "min": 2, "max": 8, "step": 1}),
                "scale": ("FLOAT", {"default": 1.0, "min": 0.25, "max": 4.0, "step": 0.25}),
                "tta": ("BOOLEAN", {"default": True}),
                "include_last": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "interpolate"
    CATEGORY = "video/interpolation"

    def interpolate(self, sg_rife, images, multiplier, scale, tta, include_last):
        return (
            interpolate_images(
                images,
                sg_rife,
                multiplier=int(multiplier),
                scale=scale,
                tta=tta,
                include_last=include_last,
            ),
        )


NODE_CLASS_MAPPINGS = {
    "SGRIFELoader": SGRIFELoader,
    "SGRIFEInterpolate": SGRIFEInterpolate,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SGRIFELoader": "Load SG-RIFE",
    "SGRIFEInterpolate": "Interpolate Frames (SG-RIFE)",
}
