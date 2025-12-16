# PowerShell script to create gesture worker virtualenv and install requirements
python -m venv gesture_env
.\gesture_env\Scripts\Activate.ps1
pip install --upgrade pip setuptools wheel
pip install -r requirements_gesture.txt
Write-Host "Gesture virtualenv created in ./gesture_env. Run the worker with: .\gesture_env\Scripts\python.exe gesture_worker.py"