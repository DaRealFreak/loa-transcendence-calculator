@echo off

REM Navigate to the base directory
set BASE_DIR=%~dp0
cd "%BASE_DIR%\.."

REM Activate the virtual environment
call .\.venv\Scripts\activate

REM Run the game.py script
.\.venv\Scripts\python.exe game.py
