Mediapipe integration (README)

Overview
--------
This project supports two modes of Mediapipe integration:

1. Legacy `solutions` API (older versions of the `mediapipe` package). Used when `mp.solutions.pose` / `mp.solutions.hands` are available. This is what the project originally targeted.

2. New `tasks` API (`mediapipe.tasks`) — provides `PoseLandmarker` and `HandLandmarker` classes. This is supported as an optional mode if you provide model files and enable `use_tasks` in the configuration.

Why both?
-----------
Different `mediapipe` distributions and versions expose different top-level APIs (notably `solutions` vs `tasks`). To be robust we:

- Prefer `solutions` if available (keeps earlier behavior).
- Optionally initialize `tasks` landmarkers when configured (higher fidelity models via TFLite), falling back to a lightweight contour-based estimator otherwise.

How to enable `tasks` mode
--------------------------
1. Install `mediapipe` into your `.venv`:
   pip install mediapipe

2. Add model paths to the config (file `config/security_config.json`) under the `gesture` section:

{
  "gesture": {
    "use_tasks": true,
    "pose_model_path": "models/pose_landmarker_full.task",
    "hand_model_path": "models/hand_landmarker.task",
    "auto_discover_tasks": true
  }
}

You can use the included helper to download models into `models/`:

    python -m tools.download_mediapipe_models --pose pose_landmarker --hand hand_landmarker

The helper accepts URLs or local `file://` paths for offline installs. It writes files into `models/` and the config watcher will pick them up. Restart the service or wait for the config watcher to pick up changes (hot-reload is supported).

Notes
-----
- The code will attempt to initialize Task-based landmarkers only if `use_tasks` is true and the model paths are provided.
- If initialization fails (missing model files or incompatible mediapipe build), the recognizer will log a warning and run in fallback mode.
- Fallback mode uses a fast contour heuristic and motion-based scoring so the system remains functional for testing, CI, and low-dependency setups.

If you'd like, I can also add a small script to download recommended task model assets and place them in `models/` for easy setup. Let me know and I'll add it.
