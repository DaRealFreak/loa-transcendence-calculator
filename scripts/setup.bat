@echo off

REM Navigate to the base directory
set BASE_DIR=%~dp0
cd "%BASE_DIR%\.."

REM Create a virtual environment
python -m venv .venv

REM Activate the virtual environment
call .\.venv\Scripts\activate.bat

REM Install specific packages
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
