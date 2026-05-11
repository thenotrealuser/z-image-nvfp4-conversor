<img width="1099" height="789" alt="image" src="https://github.com/user-attachments/assets/63d14f04-7f88-4c80-99f1-140a28cdcaa8" />

<img width="351" height="52" alt="image" src="https://github.com/user-attachments/assets/7262f67a-8d16-4307-bae7-9898f38b86b9" />

<img width="351" height="51" alt="image" src="https://github.com/user-attachments/assets/5951cfed-2fe2-4334-9de9-f4d7299ec9e9" />


REQUIRES PYTHON 3.10.11


Z-Image / HiDream NVFP4 Kitchen Converter

GPU-friendly NVFP4 converter for modern diffusion models using mixed-precision strategies focused on real-world ComfyUI compatibility.

Supports:

Z-Image Turbo
HiDream-I1
FP16/BF16 source models
Experimental FP8 → NVFP4 conversion

Designed and tested around:

NVIDIA Blackwell GPUs
RTX 50 series
ComfyUI
comfy-kitchen quant runtime
Features

✅ GUI interface
✅ CLI support
✅ Dry scan before conversion
✅ Mixed precision NVFP4 conversion
✅ Automatic architecture detection
✅ Per-model conversion profiles
✅ Safe MoE handling for HiDream
✅ FP8 detection
✅ Quantization metadata generation
✅ Real-time progress log
✅ CUDA acceleration
✅ ComfyUI-compatible output

Supported Models
Model	Status
Z-Image Turbo	Stable
HiDream-I1	Stable
HiDream-I1 FP8 source	Experimental
Generic Diffusers	Experimental
Why This Exists

Most NVFP4 tooling is:

Linux-focused
datacenter-oriented
TensorRT-heavy
not compatible with ComfyUI workflows
painful for desktop users

This project focuses on:

Windows
ComfyUI
RTX 50 GPUs
real safetensors workflows

without requiring enterprise pipelines.

Recommended Source Models
Best quality
FP16 / BF16 → NVFP4

Recommended for:

HiDream-I1
Z-Image
large DiT models
Experimental
FP8 → NVFP4

Can work, but may:

reduce quality
carry old quant metadata
create instability
break some loaders

Use only if FP16/BF16 is unavailable.

Installation
1. Extract zip

Example:

C:\NVFP4_Converter\
2. Install environment

Run:

install_venv.bat
3. Launch GUI
run_gui.bat
GUI Workflow
Step 1

Select input .safetensors

Step 2

Choose output file

Step 3

Select profile

Z-Image
Z-Image-Turbo
Z-Image-Turbo-Conservative
HiDream
HiDream-I1-Auto
Step 4

Run:

Dry scan

before conversion.

The scan detects:

tensor structure
FP8 metadata
MoE layers
quantizable layers
unsafe tensors
Step 5

Run:

Convert
HiDream Notes

HiDream-I1 uses:

Sparse Diffusion Transformer
Mixture-of-Experts (MoE)

Some tensors MUST remain BF16/FP16 for stability.

The converter automatically protects:

ff_i.gate.weight
ff_t.gate.weight

This avoids:

shape mismatch
routing corruption
ComfyUI crashes
empty expert dispatch failures
Z-Image Notes

Z-Image works extremely well with NVFP4 mixed precision.

Recommended path:

FP16 → NVFP4

FP8 sources may still work but are considered experimental.

Recommended ComfyUI Settings
Loader
weight_dtype: default
First test

Recommended sanity test:

1024x1024
20 steps
CFG 4~5
no LoRA
no tiled diffusion
Verifying NVFP4 Is Active

ComfyUI logs should show:

Found quantization metadata version 1
Detected mixed precision quantization
Using mixed precision operations
Example Results
HiDream-I1 Full FP16

Original:

~31.8 GB

Converted:

~13 GB

while remaining loadable inside ComfyUI on RTX 5060 Ti 16GB.

Known Limitations
Full NVFP4

Possible experimentally, but not recommended.

Some tensors are too sensitive for 100% FP4 quantization.

Mixed precision is the recommended approach.

FP8 sources

FP8 → NVFP4 can:

work perfectly
partially work
fail depending on metadata/runtime

Results vary between models.

Tested Environment
Component	Version
Windows	11
Python	3.10 / 3.11
PyTorch	cu130
GPU	RTX 5060 Ti 16GB
ComfyUI	0.21+
CLI Example
Dry scan
python -u nvfp4_tool\convert_cli.py --input model.safetensors --output model_nvfp4.safetensors --model-type HiDream-I1-Auto --scan-only
Convert
python -u nvfp4_tool\convert_cli.py --input model.safetensors --output model_nvfp4.safetensors --model-type HiDream-I1-Auto
Credits

Built around:

PyTorch
comfy-kitchen runtime
safetensors
ComfyUI ecosystem
Disclaimer

This project is experimental.

NVFP4 support in consumer workflows is still evolving rapidly across:

PyTorch
ComfyUI
kernels
runtime implementations

Always keep backups of original models.
