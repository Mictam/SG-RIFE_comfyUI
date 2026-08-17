# SG-RIFE for ComfyUI

**Semantic-Guided RIFE frame interpolation with a DINOv3 backbone for ComfyUI.**

[Based on Paper: *SG-RIFE: Semantic-Guided Real-Time Intermediate Flow Estimation with Diffusion-Competitive Perceptual Quality*](https://arxiv.org/abs/2512.18241)

Standalone ComfyUI integration for the SG-RIFE / FlowNet-DINO frame-interpolation model extracted from my local video processing pipeline. The repository contains the inference architecture and ComfyUI integration, packaged primarily for accessibility and experimentation rather than maximum throughput.

## Highlights

- Dedicated SG-RIFE loader and interpolation nodes
- FP32 reference mode and BF16 execution mode
- ComfyUI-native model loading, VRAM management, and offloading
- Test-time augmentation (TTA)
- 2× through 8× interpolation
- Correct scheduling for the checkpoint’s midpoint-only interpolation behaviour
- Automatic or manually selected output FPS in the example workflow

## Benchmark results


All quality metrics are calculated only on  intermediate frames; 
- Higher PSNR and SSIM are better.
- Lower LPIPS is better.
- Synth FPS measures interpolation only; it excludes model loading, warm-up, decoding, quality scoring, preview, and encoding.

Results are specific to the evaluated clips, resolution, hardware, and runtime configuration. They are provided as reference measurements rather than universal performance claims.

### 2× interpolation — 2560 × 1440

| Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ | FPS ↑ |
| --- | ---: | ---: | ---: | ---: |
| SG-RIFE (FlowNet-DINO) | **38.9102 dB** | **0.983131** | **0.013067** | 0.581 |
| RIFE 4.26 | 38.3679 dB | 0.980663 | 0.014550 | **2.960** |

### 2× interpolation — 1280 × 720

| Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ | FPS ↑ |
| --- | ---: | ---: | ---: | ---: |
| SG-RIFE (FlowNet-DINO) | **42.4162 dB** | **0.989753** | **0.006685** | 5.071 |
| RIFE 4.26 | 39.9550 dB | 0.983548 | 0.007972 | **38.087** |

### 3× interpolation - 1280 x 720

| Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ | FPS ↑ |
| --- | ---: | ---: | ---: | ---: |
| SG-RIFE (FlowNet-DINO) | **36.5544 dB** | **0.967801** | 0.034863 | 3.786 |
| RIFE 4.26 | 36.2611 dB | 0.965302 | **0.026568** | **38.471** |

### 4× interpolation - 1280 x 720

| Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ | FPS ↑ |
| --- | ---: | ---: | ---: | ---: |
| SG-RIFE (FlowNet-DINO) | 34.3005 dB | **0.954079** | 0.033502 | 5.501 |
| RIFE 4.26 | **34.4418 dB** | 0.953859 | **0.032446** | **37.490** |

## Nodes

### Load SG-RIFE

Loads DINOv3 ViT-S/16 and the SG-RIFE FlowNet checkpoint. FP32 matches the reference pipeline; BF16 uses less memory but should be checked for output quality.

### Interpolate Frames (SG-RIFE)

Accepts a ComfyUI `IMAGE` batch in BHWC format and returns an interpolated `IMAGE` batch.

For `F` source frames and multiplier `N`, the complete output contains `(F - 1) * N + 1` frames when `include_last` is enabled.

The supplied checkpoint is midpoint-only. Multipliers 2×, 4×, and 8× use recursive midpoint interpolation directly. Other multipliers generate the next power-of-two timeline and temporally resample it. This avoids the duplicate frames produced by treating the ignored timestep argument as arbitrary-time support.

## Installation

Clone this repository into ComfyUI's `custom_nodes` directory:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/mictam/SG-RIFE_comfyUI.git
```

Install dependencies in the same Python environment used by ComfyUI:

```bash
python -m pip install -r SG-RIFE_comfyUI/requirements.txt
```

Restart ComfyUI after installation or after changing Python source files.

## Model setup

The repository does not include checkpoints or Meta's DINOv3 source tree. By default it expects:

```text
SG-RIFE_comfyUI/
└── models/
    ├── dinov3_repo/
    │   └── hubconf.py
    ├── dino/
    │   └── dinov3_vits16_pretrain_lvd1689m-08c60483.pth
    └── sg_rife/
        └── flownet.pkl
```

You can reuse existing files without copying them by setting:

```text
SG_RIFE_DINOV3_REPO
SG_RIFE_DINO_CHECKPOINT
SG_RIFE_CHECKPOINT
```

PowerShell example:

```powershell
$env:SG_RIFE_DINOV3_REPO = "C:\path\to\dinov3_repo"
$env:SG_RIFE_DINO_CHECKPOINT = "C:\path\to\dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
$env:SG_RIFE_CHECKPOINT = "C:\path\to\flownet.pkl"
```

Set the variables in the environment that launches ComfyUI. The loader reports every missing path instead of downloading anything automatically.

DINOv3 source and weights remain subject to Meta's DINOv3 License Agreement.

## Example workflow

Import `example_workflows/sg_rife_video_interpolation.json`.

The workflow includes:

1. video loading and component extraction;
2. SG-RIFE loading and interpolation;
3. a 2×–8× interpolation selector;
4. automatic FPS multiplication;
5. optional manual final FPS;
6. video creation and saving with original audio and bit depth.

`Use Manual Final FPS = false` preserves the source duration by using source FPS × multiplier. Set it to `true` to use `Manual Final Video FPS` instead.

## Development

Run the standalone tests with:

```bash
python -m pip install -e ".[test]"
pytest -q
```

The tests do not require model weights. They cover every multiplier, ordering, output count, padding/cropping, chunk behavior, validation, and model construction.

## License and attribution

The repository is distributed under Apache-2.0.
