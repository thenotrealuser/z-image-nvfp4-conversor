import argparse
import gc
import json
import os
import sys
import time
from collections import OrderedDict, Counter
from pathlib import Path


def log(msg=""):
    print(msg, flush=True)


BLACKLISTS = {
    # HiDream-I1 sparse DiT / MoE profile.
    # Strategy: quantize big 2D Linear weights, keep glue/small/sensitive tensors BF16.
    # Architecture names are based on HiDreamImageTransformer2DModel:
    # x_embedder, t_embedder, p_embedder, pe_embedder, caption_projection,
    # double_stream_blocks, single_stream_blocks, final_layer,
    # block.adaLN_modulation, block.attn1, block.ff_i, block.ff_t.
    "HiDream-I1-Auto": [
        # HiDream MoE router gates must stay raw BF16/FP16.
        # If quantized, ComfyUI sees packed NVFP4 shape [16, 1280]
        # where HiDreamImageTransformer2DModel expects [4, 2560].
        "ff_i.gate.weight", "ff_t.gate.weight",
        "bias", "norm", "adaLN_modulation", "x_embedder", "t_embedder",
        "p_embedder", "pe_embedder", "final_layer", "caption_projection",
    ],
    # More aggressive: also quantizes caption_projection.
    "HiDream-I1-Aggressive": [
        # Even aggressive mode must not quantize MoE router gates.
        "ff_i.gate.weight", "ff_t.gate.weight",
        "bias", "norm", "adaLN_modulation", "x_embedder", "t_embedder",
        "p_embedder", "pe_embedder", "final_layer",
    ],
    # Safer output: keeps attention and caption projection BF16, mostly quantizes FF/MoE weights.
    "HiDream-I1-Conservative": [
        "ff_i.gate.weight", "ff_t.gate.weight",
        "bias", "norm", "adaLN_modulation", "attn", "attention", "x_embedder",
        "t_embedder", "p_embedder", "pe_embedder", "final_layer", "caption_projection",
    ],
    # Emergency scanner profile: converts almost nothing; useful for seeing the structure first.
    "HiDream-I1-ScanOnly-Safe": [
        "",  # blacklist everything if conversion is accidentally pressed
    ],
    # Keep old profiles available because the underlying converter is generic.
    "Z-Image-Turbo": [
        "cap_embedder", "x_embedder", "noise_refiner", "context_refiner", "t_embedder", "final_layer",
    ],
    "Z-Image-Turbo-Conservative": [
        "attention", "adaLN_modulation", "norm", "cap_embedder", "x_embedder", "noise_refiner",
        "context_refiner", "t_embedder", "final_layer",
    ],
}

FP8_LAYERS = {
    "Qwen-Image-2512": ["txt_mlp", "txt_mod"],
}

# These are usually auxiliary tensors from an already-quantized FP8/NVFP4 source.
# Keeping them in the output can overwrite the new comfy-kitchen NVFP4 scale tensors,
# producing ComfyUI loader errors such as:
# "self.dim() cannot be 0 to view BFloat16 as Float8_e4m3fn".
SOURCE_QUANT_AUX_SUFFIXES = (
    ".weight_scale",
    ".comfy_quant",
)


def import_deps():
    try:
        import torch
        import safetensors
        import safetensors.torch
        from safetensors import safe_open
        import comfy_kitchen as ck
        from comfy_kitchen.tensor import TensorCoreNVFP4Layout
        return torch, safetensors, safe_open, ck, TensorCoreNVFP4Layout
    except Exception as e:
        log("ERROR: Failed to import required modules.")
        log(f"{type(e).__name__}: {e}")
        log("Fix: run install_venv.bat or install_deps_only.bat")
        sys.exit(2)


def base_meta_key(k: str) -> str:
    base = k.replace(".weight", "")
    if "model.diffusion_model." in base:
        return base.split("model.diffusion_model.", 1)[-1]
    return base


def base_file_key(k: str) -> str:
    return k[:-len(".weight")] if k.endswith(".weight") else k.replace(".weight", "")


def is_quantizable_weight(k, v):
    return k.endswith(".weight") and getattr(v, "ndim", 0) == 2


def is_source_quant_aux_key(k: str) -> bool:
    return k.endswith(SOURCE_QUANT_AUX_SUFFIXES)


def dtype_name(dtype):
    return str(dtype).replace("torch.", "")


def is_fp8_dtype_name(name):
    return "float8" in name.lower() or "fp8" in name.lower()


def is_fp8_tensor(v):
    return is_fp8_dtype_name(dtype_name(v.dtype))


def gpu_line():
    try:
        import subprocess
        out = subprocess.check_output([
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
            "--format=csv,noheader,nounits",
        ], text=True, stderr=subprocess.DEVNULL, timeout=3).strip().splitlines()[0]
        return out
    except Exception:
        return "n/a"


def load_source_scale(f, keys_set, weight_key, torch, device):
    scale_key = base_file_key(weight_key) + ".weight_scale"
    if scale_key not in keys_set:
        return None, scale_key
    try:
        scale = f.get_tensor(scale_key).to(device=device, dtype=torch.bfloat16)
        return scale, scale_key
    except Exception as e:
        log(f"WARN: failed to read source FP8 scale {scale_key}: {type(e).__name__}: {e}")
        return None, scale_key


def apply_source_scale(t, scale, torch):
    """Best-effort dequant for ComfyUI-style FP8 scaled weights."""
    if scale is None:
        return t
    try:
        if scale.numel() == 1:
            return t * scale.reshape(())
        # Try common broadcast shapes.
        try:
            return t * scale
        except Exception:
            pass
        if t.ndim == 2:
            if scale.numel() == t.shape[0]:
                return t * scale.reshape(-1, 1)
            if scale.numel() == t.shape[1]:
                return t * scale.reshape(1, -1)
        log(f"WARN: source FP8 scale shape {tuple(scale.shape)} not broadcastable to {tuple(t.shape)}; using raw cast.")
        return t
    except Exception as e:
        log(f"WARN: failed applying source FP8 scale: {type(e).__name__}: {e}; using raw cast.")
        return t


def source_weight_to_bf16(f, keys_set, k, v, torch, device, dequant_fp8=True):
    # Convert FP8 source weights to BF16 before re-quantizing/keeping.
    # For already-quantized ComfyUI FP8 models this avoids quantizing raw FP8 codes as if they were real BF16 weights.
    t = v.to(device=device, dtype=torch.bfloat16)
    used_scale_key = None
    if dequant_fp8 and is_fp8_tensor(v):
        scale, scale_key = load_source_scale(f, keys_set, k, torch, device)
        used_scale_key = scale_key if scale is not None else None
        t = apply_source_scale(t, scale, torch)
    return t, used_scale_key


def scan_file(path, model_type):
    torch, safetensors, safe_open, ck, TensorCoreNVFP4Layout = import_deps()
    path = Path(path)
    blacklist = BLACKLISTS.get(model_type, BLACKLISTS["HiDream-I1-Auto"])
    dtype_counts = Counter()
    quantizable = 0
    blacklisted_quantizable = 0
    total_tensors = 0
    total_bytes = path.stat().st_size if path.exists() else 0
    examples = []
    aux_keys = []
    prefix_counts = Counter()
    interesting_samples = {
        "double_stream_blocks": [],
        "single_stream_blocks": [],
        "attn": [],
        "ff_or_moe": [],
        "embedder": [],
        "caption_projection": [],
        "final_layer": [],
    }
    with safe_open(str(path), framework="pt", device="cpu") as f:
        keys = list(f.keys())
        meta = f.metadata() or {}
        for k in keys:
            total_tensors += 1
            parts = k.split(".")
            prefix_counts[".".join(parts[:3]) if len(parts) >= 3 else k] += 1
            for label, needles in {
                "double_stream_blocks": ["double_stream_blocks"],
                "single_stream_blocks": ["single_stream_blocks"],
                "attn": ["attn", "attention"],
                "ff_or_moe": ["ff_i", "ff_t", "moe", "experts", "gate"],
                "embedder": ["embedder"],
                "caption_projection": ["caption_projection"],
                "final_layer": ["final_layer"],
            }.items():
                if len(interesting_samples[label]) < 8 and any(n in k for n in needles):
                    interesting_samples[label].append(k)
            if is_source_quant_aux_key(k):
                aux_keys.append(k)
            v = f.get_tensor(k)
            dn = dtype_name(v.dtype)
            dtype_counts[dn] += 1
            if is_quantizable_weight(k, v):
                if any(name in k for name in blacklist):
                    blacklisted_quantizable += 1
                else:
                    quantizable += 1
                    if len(examples) < 10:
                        examples.append(k)
            del v
    return {
        "path": str(path),
        "size_gb": round(total_bytes / (1024**3), 2),
        "model_type": model_type,
        "total_tensors": total_tensors,
        "quantizable_2d_weights": quantizable,
        "blacklisted_2d_weights": blacklisted_quantizable,
        "source_quant_aux_tensors": len(aux_keys),
        "source_quant_aux_examples": aux_keys[:10],
        "dtype_counts": dict(dtype_counts),
        "metadata_keys": list(meta.keys()),
        "top_key_prefixes": dict(prefix_counts.most_common(30)),
        "interesting_samples": interesting_samples,
        "examples": examples,
        "fp8_detected": any(is_fp8_dtype_name(k) for k in dtype_counts),
    }


def do_scan(args):
    info = scan_file(args.input, args.model_type)
    log("=== Dry scan ===")
    log(json.dumps(info, indent=2, ensure_ascii=False))
    if info["fp8_detected"] and not args.allow_fp8:
        log("\nWARNING: FP8 source detected. Conversion is blocked unless --allow-fp8 is used.")
        log("Recommended source is BF16/FP16 original. FP8 -> NVFP4 can degrade quality or carry stale quantization sidecars.")
        return 8
    if info["source_quant_aux_tensors"]:
        log("\nNOTE: source quantization aux tensors detected. v2 will strip old aux tensors and rebuild fresh NVFP4 aux tensors.")
    return 0


def validate_output_tensors(new_sd, quant_map):
    bad = []
    for k, v in new_sd.items():
        if k.endswith(".weight_scale") and getattr(v, "ndim", 0) == 0:
            # ComfyUI's loader reinterprets some scale tensors as float8; 0D tensors explode there.
            bad.append((k, dtype_name(v.dtype), tuple(v.shape)))
    return bad


def convert(args):
    torch, safetensors, safe_open, ck, TensorCoreNVFP4Layout = import_deps()
    input_path = Path(args.input)
    output_path = Path(args.output)
    if output_path.suffix.lower() != ".safetensors":
        output_path = output_path.with_suffix(".safetensors")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.device == "cuda" and not torch.cuda.is_available():
        log("ERROR: CUDA selected but torch.cuda.is_available() is False.")
        return 3

    blacklist = BLACKLISTS.get(args.model_type, BLACKLISTS["HiDream-I1-Auto"])
    fp8_layers = FP8_LAYERS.get(args.model_type, [])

    log("=== HiDream NVFP4 Kitchen Lab v2 ===")
    log("HiDream lab: scans key structure, strips stale FP8 quant aux tensors, and can convert 2D weights to NVFP4.")
    log(f"Input: {input_path}")
    log(f"Output: {output_path}")
    log(f"Model type/profile: {args.model_type}")
    log(f"Device: {args.device}")
    log(f"Allow FP8 source: {args.allow_fp8}")
    log(f"Input size: {round(input_path.stat().st_size / (1024**3), 2)} GB")
    log(f"GPU: {gpu_line()}")
    log("")

    scan = scan_file(input_path, args.model_type)
    log("Scan summary:")
    log(json.dumps({k: scan[k] for k in ["total_tensors", "quantizable_2d_weights", "blacklisted_2d_weights", "source_quant_aux_tensors", "dtype_counts", "fp8_detected"]}, indent=2))
    if scan["fp8_detected"] and not args.allow_fp8:
        log("ERROR: FP8 source detected, but --allow-fp8 was not passed.")
        log("Use BF16/FP16 source for best quality, or pass --allow-fp8 to force conversion.")
        return 8

    new_sd = {}
    quant_map = {
        "format_version": "1.0",
        "layers": {},
        "source_model_type": args.model_type,
        "source_file": input_path.name,
        "source_dtype_counts": scan["dtype_counts"],
        "source_quant_aux_stripped": True,
        "fp8_source_dequantized_before_requant": bool(scan["fp8_detected"]),
        "warning": "Generated by an unofficial HiDream lab wrapper using comfy-kitchen TensorCoreNVFP4Layout. v2 keeps HiDream MoE router gates in BF16/FP16.",
    }

    source_metadata = {}
    start = time.time()
    converted = 0
    kept = 0
    failed = 0
    stripped_aux = 0
    fp8_dequantized = 0

    with safe_open(str(input_path), framework="pt", device="cpu") as f:
        source_metadata = f.metadata() or {}
        keys = list(f.keys())
        keys_set = set(keys)
        total = len(keys)
        for i, k in enumerate(keys, 1):
            try:
                pct = i * 100.0 / max(1, total)
                if i == 1 or i % args.progress_every == 0 or i == total:
                    log(f"PROGRESS|{i}|{total}|{pct:.2f}|converted={converted}|kept={kept}|failed={failed}|stripped_aux={stripped_aux}|fp8_deq={fp8_dequantized}|key={k}|gpu={gpu_line()}")

                # Critical FP8-source fix: do not keep stale source quantization sidecar tensors.
                # They can overwrite the fresh NVFP4 sidecars with scalar BF16 values and break ComfyUI loading.
                if is_source_quant_aux_key(k):
                    stripped_aux += 1
                    continue

                v = f.get_tensor(k)

                if any(name in k for name in blacklist):
                    if torch.is_floating_point(v):
                        if k.endswith(".weight") and is_fp8_tensor(v):
                            t, used_scale = source_weight_to_bf16(f, keys_set, k, v, torch, args.device, dequant_fp8=True)
                            if used_scale:
                                fp8_dequantized += 1
                            new_sd[k] = t.cpu()
                            del t
                        else:
                            new_sd[k] = v.to(dtype=torch.bfloat16).cpu()
                    else:
                        new_sd[k] = v.cpu()
                    kept += 1
                    del v
                    continue

                if is_quantizable_weight(k, v):
                    base_file = base_file_key(k)
                    base_meta = base_meta_key(k)

                    v_tensor, used_scale = source_weight_to_bf16(f, keys_set, k, v, torch, args.device, dequant_fp8=True)
                    if used_scale:
                        fp8_dequantized += 1
                    del v

                    # Optional FP8 fallback path for named layers. Mostly retained for profile parity.
                    if fp8_layers and any(name in k for name in fp8_layers):
                        weight_scale = (v_tensor.abs().max() / 448.0).clamp(min=1e-12).float().reshape(1)
                        weight_quantized = ck.quantize_per_tensor_fp8(v_tensor, weight_scale)
                        new_sd[k] = weight_quantized.cpu()
                        new_sd[f"{base_file}.weight_scale"] = weight_scale.to(torch.bfloat16).cpu()
                        quant_map["layers"][base_meta] = {"format": "float8_e4m3fn"}
                        converted += 1
                        del v_tensor
                        if args.device == "cuda":
                            torch.cuda.empty_cache()
                        continue

                    try:
                        qdata, params = TensorCoreNVFP4Layout.quantize(v_tensor)
                        tensors = TensorCoreNVFP4Layout.state_dict_tensors(qdata, params)
                        for suffix, tensor in tensors.items():
                            out_key = f"{base_file}.weight{suffix}"
                            new_sd[out_key] = tensor.cpu()
                        quant_map["layers"][base_meta] = {"format": "nvfp4"}
                        converted += 1
                    except Exception as e:
                        log(f"WARN: NVFP4 failed for {k}: {type(e).__name__}: {e}. Keeping BF16.")
                        new_sd[k] = v_tensor.to(dtype=torch.bfloat16).cpu()
                        failed += 1
                    finally:
                        del v_tensor
                        if args.device == "cuda":
                            torch.cuda.empty_cache()
                    continue

                # Non-2D tensors: keep BF16 if floating, otherwise keep as-is.
                if torch.is_floating_point(v):
                    new_sd[k] = v.to(dtype=torch.bfloat16).cpu()
                else:
                    new_sd[k] = v.cpu()
                kept += 1
                del v

            except KeyboardInterrupt:
                raise
            except Exception as e:
                log(f"ERROR processing {k}: {type(e).__name__}: {e}")
                failed += 1
                try:
                    del v
                except Exception:
                    pass
                gc.collect()
                if args.device == "cuda":
                    torch.cuda.empty_cache()
                if not args.continue_on_error:
                    return 5

    bad_scales = validate_output_tensors(new_sd, quant_map)
    if bad_scales:
        log("ERROR: Output contains 0D .weight_scale tensors, which ComfyUI can fail to reinterpret as FP8 scale bytes.")
        for item in bad_scales[:20]:
            log(f"  bad scale: {item}")
        log("Refusing to save broken output. Try a BF16/FP16 source or send this log.")
        return 9

    final_metadata = OrderedDict()
    final_metadata["_quantization_metadata"] = json.dumps(quant_map)
    final_metadata["converted_by"] = "HiDream NVFP4 Kitchen Lab v2"
    final_metadata["converter_basis"] = "comfy-kitchen TensorCoreNVFP4Layout"
    final_metadata["model_type"] = args.model_type
    final_metadata["source_file"] = input_path.name
    # Preserve useful existing metadata where possible without overwriting our quantization map.
    for mk, mv in source_metadata.items():
        if mk not in final_metadata and isinstance(mv, str):
            final_metadata[mk] = mv

    log("")
    log("Saving safetensors...")
    safetensors.torch.save_file(new_sd, str(output_path), metadata=final_metadata)
    size_gb = output_path.stat().st_size / (1024**3)
    elapsed = time.time() - start
    log("DONE")
    log(f"Output size: {size_gb:.2f} GB")
    log(f"Converted layers: {converted}")
    log(f"Kept tensors: {kept}")
    log(f"Stripped source quant aux tensors: {stripped_aux}")
    log(f"FP8 weights dequantized before NVFP4/BF16: {fp8_dequantized}")
    log(f"Failed layers kept BF16: {failed}")
    log(f"Elapsed: {elapsed:.1f}s")
    log(f"Output: {output_path}")
    return 0


def main():
    p = argparse.ArgumentParser(description="HiDream/ComfyUI NVFP4 converter lab using comfy-kitchen TensorCoreNVFP4Layout")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--model-type", default="Z-Image-Turbo", choices=list(BLACKLISTS.keys()))
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--allow-fp8", action="store_true", help="Allow FP8 source files. BF16/FP16 is recommended.")
    p.add_argument("--scan-only", action="store_true")
    p.add_argument("--continue-on-error", action="store_true")
    p.add_argument("--progress-every", type=int, default=5)
    args = p.parse_args()

    if not Path(args.input).exists():
        log(f"ERROR: input not found: {args.input}")
        return 1

    if args.scan_only:
        return do_scan(args)
    return convert(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        log("Interrupted by user.")
        raise SystemExit(130)
