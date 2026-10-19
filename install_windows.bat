@echo off
REM MonoMetric 3D - installazione Windows + RTX 5090 (Blackwell, sm_120)
REM Eseguire da "Anaconda Prompt" nella cartella del progetto:  install_windows.bat

call conda create -n monometric python=3.11 -y || goto :err
call conda activate monometric || goto :err

REM git serve a pip per installare i pacchetti da GitHub
where git >nul 2>nul || call conda install -c conda-forge git -y

REM PyTorch con CUDA 12.8 (supporto nativo sm_120)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128 || goto :err

REM MonoMetric (+ GUI ed export LAS/E57)
pip install -e ".[all]" || goto :err

REM Dipendenze di MoGe-2 (senza FlexGEMM/Triton, non installabile in modo affidabile su Windows)
pip install click numpy scipy matplotlib huggingface-hub requests tqdm "git+https://github.com/EasternJournalist/utils3d-moge.git@62f09d58509485564e24d5d9f6aac9ee9ebc0c37" "git+https://github.com/EasternJournalist/pipeline.git@1c511390d90226c00c101f34b84df26a0f8789b4" || goto :err
pip install --no-deps "git+https://github.com/microsoft/MoGe.git" || goto :err

monometric doctor
echo.
echo Installazione completata.
goto :eof

:err
echo.
echo ERRORE durante l'installazione: controlla il messaggio qui sopra.
exit /b 1
