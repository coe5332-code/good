"""Compatibility shim for `utils.service_utils`.

This imports and re-exports the implementations from
`training_video_generation.utils.service_utils` so older import paths
continue to work.
"""
try:
    from training_video_generation.utils.service_utils import *  # noqa: F401,F403
except Exception as e:
    raise ImportError(
        "Failed to import training_video_generation.utils.service_utils for compatibility shim."
    ) from e
