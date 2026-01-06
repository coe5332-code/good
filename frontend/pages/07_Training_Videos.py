import streamlit as st
import runpy
import traceback
import sys
import os
from pathlib import Path

st.title("🎥 Training Videos")
st.markdown("This page integrates the BSK Training Video Generator into the frontend.")

# Locate the training-video script relative to this file
base = Path(__file__).resolve().parents[2]
tv_dir = base / "training-video-generation"
script_path = tv_dir / "app.py"

if script_path.exists():
    # Temporarily adjust sys.path and cwd so the training-video package imports resolve
    old_cwd = Path.cwd()
    old_sys_path = sys.path.copy()
    try:
        # Put the training-video directory first so its `utils` package is found
        sys.path.insert(0, str(tv_dir))

        # Also remove the frontend directory from sys.path to avoid collision with frontend/utils.py
        frontend_dir = base / "frontend"
        sys.path = [p for p in sys.path if str(frontend_dir) not in (p or "")]

        # Change working directory to the training-video folder to match expected relative paths
        os.chdir(tv_dir)

        # Run the training-video app
        runpy.run_path(str(script_path), run_name="__main__")
    except Exception as e:
        st.error(f"Failed to load video generator: {e}")
        st.code(traceback.format_exc())
    finally:
        # Restore cwd and sys.path
        os.chdir(old_cwd)
        sys.path[:] = old_sys_path
else:
    st.error(f"Video generator not found at {script_path}")
