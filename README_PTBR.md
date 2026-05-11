# HiDream NVFP4 Kitchen Lab v2

Ferramenta experimental para investigar e tentar converter `.safetensors` do **HiDream-I1** para NVFP4 compatível com ComfyUI usando `comfy-kitchen` e `TensorCoreNVFP4Layout`.

## Por que é diferente do Z-Image?

HiDream-I1 é um Sparse DiT com MoE. A estrutura principal costuma envolver:

- `x_embedder`, `t_embedder`, `p_embedder`, `pe_embedder`
- `caption_projection`
- `double_stream_blocks`
- `single_stream_blocks`
- `block.attn1`
- `block.ff_i` / `block.ff_t` / experts MoE
- `adaLN_modulation`
- `final_layer`

Por isso este pacote tem perfis próprios e um **Dry scan** mais detalhado antes da conversão.

## Fonte recomendada

Use **FP16/BF16** sempre que possível.

O FP8 pode até ser testado com `Allow/force FP8 source`, mas é conversão em cima de conversão. A ferramenta tenta remover auxiliares antigos como `.weight_scale` e `.comfy_quant`, mas a qualidade pode cair e o loader do ComfyUI pode reclamar se sobrar metadata torta.

## Perfis

- `HiDream-I1-Auto`: recomendado para primeiro teste. Mantém embedder/final/caption/norm/adALN em BF16 e quantiza pesos 2D grandes.
- `HiDream-I1-Aggressive`: quantiza mais coisas, inclusive `caption_projection`.
- `HiDream-I1-Conservative`: mantém attention e caption em BF16, foca mais em FF/MoE.
- `HiDream-I1-ScanOnly-Safe`: perfil para inspecionar. Se apertar Convert sem querer, praticamente não converte nada.

## Como usar

1. Rode `install_venv.bat`.
2. Rode `run_gui.bat`.
3. Escolha o `.safetensors` do HiDream.
4. Rode primeiro **Dry scan**.
5. Veja `dtype_counts`, `top_key_prefixes` e `interesting_samples`.
6. Só depois rode **Convert** com `HiDream-I1-Auto`.

## Onde colocar

Coloque o resultado em:

```txt
ComfyUI/models/diffusion_models/
```

Use `UNETLoader` no ComfyUI.

## Aviso honesto

Isto é laboratório, não milagre engarrafado. O caminho mais limpo é **FP16/BF16 → NVFP4**. O caminho **FP8 → NVFP4** só deve ser usado como tentativa.

O conversor segura o output em RAM antes de salvar, então para HiDream Full FP16 de 34 GB é recomendável ter bastante RAM livre, idealmente 32 GB ou mais, e espaço em disco sobrando.


## v2 - correção HiDream MoE gate

Esta versão mantém `ff_i.gate.weight` e `ff_t.gate.weight` em BF16/FP16.
No HiDream, esses pesos são roteadores MoE e não podem ser empacotados em NVFP4 como uma Linear comum, senão o ComfyUI dá erro de shape tipo `[16, 1280]` vs `[4, 2560]`.

Recomendado para HiDream-I1 Full:

```bat
.venv\Scripts\python.exe -u nvfp4_tool\convert_cli.py --input C:\Users\kimim\Downloads\hidream_i1_full_fp16.safetensors --output C:\Users\kimim\Downloads\hidream_i1_full_fp16_nvfp4_v2.safetensors --model-type HiDream-I1-Auto
```
