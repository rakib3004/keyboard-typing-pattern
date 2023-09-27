@echo off

pip install virtualenv==20.17.1
if exist "env" (
    echo Env folder found
) else (
    python -m venv env
)
env\Scripts\pip.exe install -r requirements.txt
pause
