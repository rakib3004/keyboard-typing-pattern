@echo off

pip install virtualenv
if exist "env" (
    echo Env folder found
) else (
    python -m venv env
)
env\Scripts\pip.exe install -r requirements.txt
pause
