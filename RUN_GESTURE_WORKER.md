Gesture Worker (isolated environment)

Overview
---------
The gesture worker runs in a separate Python virtual environment (isolated from the main system environment) so it can install `mediapipe` with `numpy<2` without conflicting with the main OpenCV setup.

Setup
-----
1. Create the venv and install dependencies:
   - Open PowerShell in the project folder and run:
     ```powershell
     .\setup_gesture_env.ps1
     ```
   - Alternatively run manually:
     ```powershell
     python -m venv gesture_env
     .\gesture_env\Scripts\Activate.ps1
     pip install --upgrade pip setuptools wheel
     pip install -r requirements_gesture.txt
     ```

2. Start the worker:
   ```powershell
   .\gesture_env\Scripts\python.exe gesture_worker.py
   ```

How it works
------------
- The main security system saves short clips to `evidence/gesture_queue/incoming` (side-effect: `evidence/videos` also keeps the clip).
- The worker watches `evidence/gesture_queue/incoming`, processes clips with MediaPipe, writes result JSON to `evidence/gesture_queue/results`, and moves processed files to `evidence/gesture_queue/processed`.
- The main system polls the `results` folder and triggers alerts when the worker reports a suspicious gesture.

Notes
-----
- Keep the worker running as a background process or system service for real deployments.
- If you want tighter coupling (e.g., direct database writes), the worker can import `data_storage_module` and log events directly.
