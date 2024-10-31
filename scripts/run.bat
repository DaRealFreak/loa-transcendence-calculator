@echo off

set base_dir=%~dp0
call "%base_dir%\..\.venv\Scripts\activate.bat"
call "%base_dir%\..\.venv\Scripts\python.exe" "%base_dir%\..\game.py
