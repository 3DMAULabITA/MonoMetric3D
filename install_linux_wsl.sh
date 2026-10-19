#!/usr/bin/env bash
# MonoMetric 3D - Linux / WSL2 (consigliato per MoGe-3)
set -e
conda create -n monometric python=3.11 -y
eval "$(conda shell.bash hook)"; conda activate monometric
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -e ".[all]"
pip install "git+https://github.com/microsoft/MoGe.git"      # MoGe-2 + MoGe-3 (FlexGEMM/Triton)
python -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0))"
python -c "from monometric import backends; print('backend installati:', backends.available())"
