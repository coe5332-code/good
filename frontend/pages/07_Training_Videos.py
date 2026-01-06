import streamlit as st
import runpy
import traceback
from pathlib import Path

st.title("🎥 Training Videos")
st.markdown("This page integrates the BSK Training Video Generator into the frontend.")

# Locate the training-video script relative to this file
base = Path(__file__).resolve().parents[2]
script_path = base / "training-video-generation" / "app.py"

if script_path.exists():
    try:
        # Execute the script so its Streamlit UI is rendered here
        runpy.run_path(str(script_path), run_name="__main__")
    except Exception as e:
        st.error(f"Failed to load video generator: {e}")
        st.code(traceback.format_exc())
else:
    st.error(f"Video generator not found at {script_path}")
