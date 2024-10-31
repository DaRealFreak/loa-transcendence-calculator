#!/bin/bash

base_dir=$(dirname "$0")
cd "$base_dir/../" || exit

python -m venv .venv
sh ./.venv/Scripts/activate.sh
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
