"""Live audio and playback-gated file ASR using the Second training protocol."""
from __future__ import annotations

from omnicommit.media import MediaCursor, MediaError
from omnicommit.runtime_protocol import chunk_milliseconds
from omnicommit.web_service import create_app as meeting_app

from .service_session import SecondStreamSession


class SecondMediaCursor(MediaCursor):
    def _chunk_end_offset(self, chunk_seconds):
        # Apply the recaster's ceiling rule to the grid starting at the selected sample.
        # At 44.1 kHz a millisecond boundary can be a fractional sample.
        try:
            chunk_ms = chunk_milliseconds(chunk_seconds)
        except ValueError as exc:
            raise MediaError(str(exc)) from exc
        rate = self.upload.sample_rate
        if chunk_ms * rate < 1000:
            raise MediaError("The selected step contains less than one source sample")
        return ((self.chunk_index + 1) * chunk_ms * rate + 999) // 1000


def create_app(backend_factory, options, *, max_sessions=1, auto_reference_factory=None,
               gap_backend_factory=None):
    def make_gap_service():
        from .gap_service import GapService
        return GapService(gap_backend_factory())

    return meeting_app(backend_factory, options, max_sessions=max_sessions,
                       auto_reference_factory=auto_reference_factory,
                       session_factory=SecondStreamSession, media_cursor_factory=SecondMediaCursor,
                       gap_service_factory=make_gap_service if gap_backend_factory else None,
                       protocol_name="second")
