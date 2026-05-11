# Z-Image NVFP4 Kitchen GUI

Ferramenta limpa para converter modelos `.safetensors` para o formato NVFP4 usado pelo ComfyUI, usando `comfy-kitchen` e `TensorCoreNVFP4Layout`.

## Importante

Este pacote NÃO usa NVIDIA ModelOpt. A tentativa antiga era o caminho errado para Z-Image single-file do ComfyUI.

O caminho usado aqui é inspirado no nó público `ComfyUI_Kitchen_nvfp4_Converter`, que converte `.safetensors` diretamente e grava metadados `_quantization_metadata`.

## Recomendação de fonte

Melhor fonte: BF16/FP16 original.

FP8 pode ser aceito pela ferramenta com a opção `Allow/force FP8 source`, mas é conversão em cima de quantização anterior. Pode perder qualidade ou gerar resultado pior.

## Instalação recomendada

Use venv. Python embedded com Torch/CUDA costuma dar dor de cabeça.

1. Rode:

```bat
install_venv.bat
```

2. Depois:

```bat
run_gui.bat
```

3. Na GUI:

- Clique em `Check environment`.
- Clique em `Dry scan` para ver quantos tensores serão convertidos.
- Escolha `Z-Image-Turbo`.
- Marque `Allow/force FP8 source` se seu arquivo for FP8.
- Clique em `Convert`.

## Onde salvar

Salve o resultado em:

```txt
ComfyUI/models/unet/
```

ou no diretório equivalente do seu Easy Install.

## Perfis

- `Z-Image-Turbo`: perfil igual ao conversor Kitchen público, mantendo embed/refiner/final em BF16.
- `Z-Image-Turbo-Conservative`: mantém atenção/norm/refiners em BF16. Arquivo maior, qualidade potencialmente mais segura.
- `Z-Image-Base`: perfil mais cauteloso para Base.
- Outros perfis foram incluídos por conveniência, mas o foco deste pacote é Z-Image.

## Arquivos

- `install_venv.bat`: cria venv e instala PyTorch cu130 nightly + dependências.
- `run_gui.bat`: abre GUI.
- `install_deps_only.bat`: instala só dependências extras no venv existente, sem mexer no Torch.
- `force_torch_cu130.bat`: reinstala PyTorch cu130 nightly.
- `nvfp4_tool/gui.py`: GUI.
- `nvfp4_tool/convert_cli.py`: conversor via terminal.
- `nvfp4_tool/env_check.py`: checagem do ambiente.

## Terminal manual

```bat
.venv\Scripts\python.exe -u nvfp4_tool\convert_cli.py ^
  --input C:\Users\kimim\Downloads\zImageTurbo_turbo.safetensors ^
  --output C:\Users\kimim\Downloads\zImageTurbo_turbo_nvfp4.safetensors ^
  --model-type Z-Image-Turbo ^
  --device cuda ^
  --allow-fp8
```

## Aviso de VRAM/RAM

A conversão carrega tensores e cria o novo checkpoint em memória. Para modelos grandes, feche ComfyUI, navegador pesado, jogos, NVIDIA Broadcast se possível, e deixe bastante RAM livre.
