# Changelog

## v2

- Corrige HiDream-I1: mantém `ff_i.gate.weight` e `ff_t.gate.weight` em BF16/FP16.
- Evita erro no ComfyUI: shape NVFP4 empacotado `[16, 1280]` tentando carregar em gate esperado `[4, 2560]`.
- Atualiza metadata `converted_by` para v2.
- Mantém o fluxo FP16/BF16 -> NVFP4 usando comfy-kitchen TensorCoreNVFP4Layout.

## v1

- Primeiro lab para converter HiDream/Z-Image para NVFP4.
