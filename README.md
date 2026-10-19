# MonoMetric 3D

**Single-image 3D reconstruction with ChArUco-based metric anchoring**  
*Research beta v0.1.0b1 · Python 3.10+*

MonoMetric 3D generates coloured 3D point clouds from individual photographs. It combines **MoGe-2** monocular geometry estimation, **ChArUco** references of known dimensions, camera parameter estimation and metric depth correction. Developed for research in **architectural surveying and heritage documentation**.

## Features

- Automatic ChArUco detection and camera-parameter estimation from EXIF and targets.
- **Zero-shot** reconstruction (model-predicted scale) and **anchored** reconstruction (depth corrected using measured targets).
- Five depth-correction families and **Leave-One-Board-Out (LOBO)** internal validation.
- Optional multi-image camera calibration, focal-length sensitivity analysis and batch processing.
- Local **Gradio** interface, CLI, coloured point-cloud export (**PLY**, optional **LAS/E57**) and reports.

## Installation

Clone or download this repository and open a terminal in its root folder. The included installers use **Conda** and install the required model dependencies:

- **Windows:** `install_windows.bat` (Anaconda Prompt) or `./install_windows.ps1` (PowerShell).
- **Linux / WSL2:** `bash install_linux_wsl.sh` (NVIDIA/CUDA environment).

Activate the environment and launch the local interface:

```bash
conda activate monometric
monometric app
```

Open `http://127.0.0.1:7860`. Model weights may be downloaded on first use.

## Command-line example

```bash
monometric detect photo.jpg --square-mm 60 -o board.json
monometric run photo.jpg --board board.json --models moge2-vitl -o results/
```

**Important:** `60` is an example, not a default: enter the **measured** ChArUco square size in millimetres. For better camera-parameter control, use `monometric calibrate` to generate a reusable `camera.json` for images acquired with the same camera settings. Run `monometric --help` for other commands.

## Scientific context

Developed alongside *Monocular Depth Estimation for Low-Cost 3D Reconstruction* (Tavolare, Cera and Antuono, 2026). The case study compares zero-shot and anchored reconstructions with an independently scaled photogrammetric reference.

**Authors:** Riccardo Tavolare (1), Valeria Cera (2), Giuseppe Antuono (3)

(1) Dept. DICATECh, Polytechnic University of Bari, Bari, Italy - riccardo.tavolare@poliba.it

(2) Dept. of Architecture, University of Naples Federico II, Naples, Italy - valeria.cera@unina.it 

(3) Dept. of Civil, Building and Environmental Engineering, University of Naples Federico II, Naples, Italy - giuseppe.antuono@unina.it

  
**Citation / DOI:** to be added after publication or release archiving.  
**License:** see the repository's license file, if provided; no reuse license is implied by this README.
