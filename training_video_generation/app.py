import streamlit as st
import asyncio
import logging
import os
import tempfile

# ✅ FIXED IMPORTS (ABSOLUTE PACKAGE IMPORTS)
from training_video_generation.utils.audio_utils import text_to_speech
from training_video_generation.utils.video_utils import (
    create_slide,
    combine_slides_and_audio,
)
from training_video_generation.services.unsplash_service import fetch_and_save_photo
from training_video_generation.services.gemini_service import generate_slides_from_raw
from training_video_generation.utils.avatar_utils import add_avatar_to_slide
from training_video_generation.utils.pdf_extractor import extract_raw_content
from training_video_generation.utils.pdf_utils import generate_service_pdf
from training_video_generation.utils.service_utils import (
    create_service_sections,
    validate_service_content,
)

logging.basicConfig(level=logging.INFO)

VOICES = {
    "en-IN-NeerjaNeural": "Neerja (Female, Indian English)",
    "en-IN-PrabhatNeural": "Prabhat (Male, Indian English)",
}


# -------------------------------------------------
# MAIN
# -------------------------------------------------
def main():
    st.set_page_config(
        page_title="BSK Training Video Generator",
        page_icon="🎥",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Load CSS
    css_path = os.path.join("assets", "style.css")
    if os.path.exists(css_path):
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    # ---------------- SIDEBAR ----------------
    with st.sidebar:
        st.markdown("### 🎥 BSK Training Generator")
        st.markdown("**Professional Training Videos**")
        st.markdown("*Bangla Sahayta Kendra*")
        st.markdown("---")

        page = st.selectbox(
            "Select Page:",
            ["🎬 Create New Video", "📂 View Existing Videos"],
            key="page_selector",
        )

        st.markdown("---")

        voice_keys = list(VOICES.keys())
        voice_labels = list(VOICES.values())
        voice_index = st.selectbox(
            "Select Narrator Voice:",
            range(len(voice_keys)),
            format_func=lambda i: voice_labels[i],
        )
        selected_voice = voice_keys[voice_index]

        st.markdown("---")
        st.markdown("### 📄 Optional Service PDF")
        uploaded_pdf = st.file_uploader(
            "Upload PDF (Overrides form)",
            type=["pdf"],
            help="If provided, form content will be ignored",
        )

        st.markdown("### 🧑‍🏫 AI Avatar")
        st.caption("Avatar will appear inside the generated training video.")

    # ---------------- ROUTING ----------------
    if page == "🎬 Create New Video":
        show_create_video_page(selected_voice, uploaded_pdf)
    else:
        show_existing_videos_page()


# -------------------------------------------------
# CREATE VIDEO PAGE
# -------------------------------------------------
def show_create_video_page(selected_voice, uploaded_pdf):
    st.title("🎥 BSK Training Video Generator")
    st.markdown("**Create training videos for BSK data entry operators**")
    st.markdown("---")

    with st.form("service_form"):
        st.subheader("📋 Service Training Information")

        col1, col2 = st.columns(2)

        with col1:
            service_name = st.text_input("Service Name *")
            service_description = st.text_area("Service Description *", height=100)

        with col2:
            how_to_apply = st.text_area(
                "Step-by-Step Application Process *", height=100
            )
            eligibility_criteria = st.text_area("Eligibility Criteria *", height=100)
            required_docs = st.text_area("Required Documents *", height=100)

        st.subheader("🎯 Training Specific Information")
        col3, col4 = st.columns(2)

        with col3:
            operator_tips = st.text_area("Operator Tips", height=100)
            service_link = st.text_input("Official Service Link")

        with col4:
            troubleshooting = st.text_area("Common Issues", height=100)
            fees_and_timeline = st.text_input("Fees & Processing Time")

        submitted = st.form_submit_button("🚀 Generate Training Video")

    if submitted:
        try:
            progress = st.progress(0)
            status = st.empty()

            video_clips = []
            audio_paths = []

            if uploaded_pdf:
                status.text("📄 Extracting content from PDF...")
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_pdf.read())
                    pdf_path = tmp.name

                pages = extract_raw_content(pdf_path)
                raw_text = "\n".join(line for page in pages for line in page["lines"])

            else:
                service_content = {
                    "service_name": service_name,
                    "service_description": service_description,
                    "how_to_apply": how_to_apply,
                    "eligibility_criteria": eligibility_criteria,
                    "required_docs": required_docs,
                    "operator_tips": operator_tips,
                    "troubleshooting": troubleshooting,
                    "service_link": service_link,
                    "fees_and_timeline": fees_and_timeline,
                }

                valid, msg = validate_service_content(service_content)
                if not valid:
                    st.error(msg)
                    return

                status.text("📄 Generating training PDF...")
                pdf_path = generate_service_pdf(service_content)

                pages = extract_raw_content(pdf_path)
                raw_text = "\n".join(line for page in pages for line in page["lines"])

            status.text("🧠 Structuring slides using AI...")
            slides = generate_slides_from_raw(raw_text)["slides"]

            for i, slide in enumerate(slides):
                status.text(f"🎬 Creating slide {i + 1}/{len(slides)}")

                narration = " ".join(slide["bullets"])
                audio = asyncio.run(text_to_speech(narration, voice=selected_voice))
                audio_paths.append(audio)

                image = fetch_and_save_photo(slide["image_keyword"])
                clip = create_slide(slide["title"], slide["bullets"], image, audio)
                clip = add_avatar_to_slide(clip, audio_duration=clip.duration)
                video_clips.append(clip)

                progress.progress(int((i + 1) / len(slides) * 80))

            status.text("🎞️ Rendering final video...")
            final_path = combine_slides_and_audio(video_clips, audio_paths)

            progress.progress(100)
            st.video(final_path)
            st.success("✅ Training video generated successfully!")

        except Exception as e:
            st.error(f"❌ Error generating video: {e}")


# -------------------------------------------------
# EXISTING VIDEOS PAGE
# -------------------------------------------------
def show_existing_videos_page():
    st.title("📂 Existing Training Videos")

    output_dir = "output_videos"
    if not os.path.exists(output_dir):
        st.info("No videos found.")
        return

    videos = [f for f in os.listdir(output_dir) if f.endswith(".mp4")]
    selected = st.selectbox("Select a video:", videos)

    with open(os.path.join(output_dir, selected), "rb") as f:
        st.video(f.read())


# -------------------------------------------------
# RUN
# -------------------------------------------------
if __name__ == "__main__":
    main()
