from moviepy.editor import (
    ImageClip,
    CompositeVideoClip,
    VideoFileClip,
)
import os


def add_avatar_to_slide(clip, audio_duration=None):
    """
    Overlays a talking avatar video onto the slide.
    If avatar video is missing, returns the slide unchanged.
    """

    avatar_path = os.path.join("assets", "avatar.mp4")

    if not os.path.exists(avatar_path):
        # Avatar optional — do not fail
        return clip

    try:
        avatar = (
            VideoFileClip(avatar_path)
            .resize(height=int(clip.h * 0.35))
            .set_position(("right", "bottom"))
        )

        if audio_duration:
            avatar = avatar.subclip(0, min(audio_duration, avatar.duration))

        return CompositeVideoClip([clip, avatar])

    except Exception as e:
        # Fail gracefully — never crash video generation
        print(f"[avatar] skipped: {e}")
        return clip
