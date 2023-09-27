@echo off

if exist "env" (
    env\Scripts\python.exe api.py
) else (
    echo Not exist
    echo Please run prepare.bat file first!!!
)
pause
