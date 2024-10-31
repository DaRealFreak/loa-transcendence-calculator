#!/bin/bash

base_dir=$(dirname "$0")
cd "$base_dir/../" || exit

./.venv/Scripts/activate
./.venv/Scripts/python.exe game.py