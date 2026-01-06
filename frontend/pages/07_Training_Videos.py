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
    # run the training video app as a MODULE
    runpy.run_module(
        "training_video_generation.app",
        run_name="__main__"
    )
except Exception as e:
    st.error("Failed to load video generator")
    st.code(traceback.format_exc())
