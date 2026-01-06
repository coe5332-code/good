import streamlit as st
import runpy
import sys
from pathlib import Path
import traceback

st.title("🎥 Training Videos")
st.markdown("This page integrates the BSK Training Video Generator.")

# project root (good/)
BASE_DIR = Path(__file__).resolve().parents[2]

# make project root importable
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

try:
    # Run the training video app. Prefer the new package layout
    # (`training_video_generation`) but fall back to the legacy
    # `training-video-generation/app.py` if present (some deployments use that name).
    pkg_dir = BASE_DIR / "training_video_generation"
    legacy_dir = BASE_DIR / "training-video-generation"
    # Also check for the new `train_video_main/train-video-main` layout
    new_dir = BASE_DIR / "train_video_main" / "train-video-main"

    if pkg_dir.exists():
        runpy.run_module("training_video_generation.app", run_name="__main__")
    elif new_dir.exists():
        runpy.run_path(str(new_dir / "app.py"), run_name="__main__")
    elif legacy_dir.exists():
        runpy.run_path(str(legacy_dir / "app.py"), run_name="__main__")
    else:
        # Last resort: attempt the module import (will raise and be shown)
        runpy.run_module("training_video_generation.app", run_name="__main__")
except Exception as e:
    st.error("Failed to load video generator")
    st.code(traceback.format_exc())
