"""One GPU service shared by microphone, system audio and uploaded files."""
from pathlib import Path

from omnicommit.web_service import create_app as meeting_app
from Second.second_asr.service_web import SecondMediaCursor
from .service_session import ThirdStreamSession


def create_app(backend_factory, options, *, max_sessions=1, auto_reference_factory=None):
    if auto_reference_factory is not None:
        raise ValueError("Third transcribes plain text; speaker references are unsupported")
    return meeting_app(backend_factory, options, max_sessions=max_sessions,
                       session_factory=ThirdStreamSession, media_cursor_factory=SecondMediaCursor,
                       protocol_name="third", chunk_ms_choices=(), allowed_controls={"commit"},
                       web_directory=Path(__file__).resolve().parents[1] / "web")
