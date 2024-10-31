#!/bin/bash

base_dir=$(dirname "$0")
"$base_dir"/../.venv/Scripts/activate
"$base_dir"/../.venv/Scripts/python.exe "$base_dir"/../game.py