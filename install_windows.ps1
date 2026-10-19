# MonoMetric 3D - installazione Windows + RTX 5090 (Blackwell, sm_120)
# Eseguire da "Anaconda PowerShell Prompt" nella cartella del progetto.
$ErrorActionPreference = "Stop"

conda create -n monometric python=3.11 -y
conda activate monometric

# PyTorch con CUDA 12.8: supporto nativo sm_120 (niente workaround fp32 / ONNX su CPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# MonoMetric (+ GUI ed export LAS/E57)
pip install -e ".[all]"

# MoGe-2 (backend principale). MoGe 3.x dichiara come dipendenza FlexGEMM (Triton), che su Windows
# non si installa in modo affidabile: installiamo MoGe senza dipendenze e poi solo quelle necessarie a MoGe-2.
pip install click numpy scipy matplotlib huggingface-hub requests tqdm `
  "git+https://github.com/EasternJournalist/utils3d-moge.git@62f09d58509485564e24d5d9f6aac9ee9ebc0c37" `
  "git+https://github.com/EasternJournalist/pipeline.git@1c511390d90226c00c101f34b84df26a0f8789b4"
pip install --no-deps "git+https://github.com/microsoft/MoGe.git"

# --- Opzionali -------------------------------------------------------------
# MoGe-3 (raffinamento sparso 3D, richiede Triton): consigliato in WSL2/Linux con
#   pip install "git+https://github.com/microsoft/MoGe.git"
# Su Windows nativo (sperimentale):
#   pip install triton-windows
#   pip install "git+https://github.com/JeffreyXiang/FlexGEMM.git@b2fadb29d41846c7981ade6801ffc689fae119cf"
# Depth Pro:      pip install "git+https://github.com/apple/ml-depth-pro.git"  (+ scaricare il checkpoint)
# UniDepthV2:     pip install --no-deps "git+https://github.com/lpiccinelli-eth/UniDepth.git"   (CC BY-NC)
# Depth Anything 3: pip install "git+https://github.com/ByteDance-Seed/Depth-Anything-3.git"

monometric doctor
