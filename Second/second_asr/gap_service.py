"""Run local speech-gap recognition outside the foreground Omni decode queue."""
from __future__ import annotations

import asyncio
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import logging
import math

from .gap_recovery import GapTracker, LATE_MAIN_SECONDS, reconcile


LOG = logging.getLogger("omnicommit.gap")


def _spans(request, words):
    groups = []
    previous_kept = False
    for word in words:
        start = request.start_seconds + word["start"]
        end = request.start_seconds + word["end"]
        text = word["text"]
        if not text.strip() or any(ch in text for ch in "<>"):
            return []
        if not request.start_seconds <= start < end <= request.end_seconds + .001:
            return []
        # Word confidence is only a heuristic. Do not invent a transcript from
        # noise, nor join across a word that failed the gate.
        if not math.isfinite(word["probability"]) or not word["probability"] >= .40:
            previous_kept = False
            continue
        lang = word["language"] if word["language_probability"] >= .45 else "und"
        value = dict(text=text, start_seconds=start, end_seconds=end)
        if (not previous_kept or groups[-1]["language"] != lang
                or groups[-1]["end_seconds"] + .35 < start):
            groups.append(dict(text="", language=lang, words=[], start_seconds=start,
                               end_seconds=end))
        group = groups[-1]
        group["text"] += text
        group["words"].append(value)
        group["end_seconds"] = end
        previous_kept = True
    return groups


class GapService:
    def __init__(self, backend):
        self.backend = backend
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="second-gap-cpu")

    def new_session(self, session, ledger):
        return GapSession(self, session, ledger)

    def close(self):
        self.executor.shutdown(wait=True, cancel_futures=True)


class GapSession:
    def __init__(self, service, session, ledger):
        self.service = service
        self.session = session
        self.ledger = ledger
        self.tracker = GapTracker()
        self.main = deque()
        self.queue = deque()
        self.task = None
        self.request = None
        self.events = []
        self.added = []
        self.removed = []
        self.recent = []

    def _event(self, request, status, message=None):
        event = dict(status=status, start_seconds=request.start_seconds, end_seconds=request.end_seconds)
        if message:
            event["message"] = message
        self.events.append(event)

    def _launch(self, request):
        self.request = request
        self.task = asyncio.get_running_loop().run_in_executor(
            self.service.executor, self.service.backend.transcribe, request.audio)

    def _schedule(self, request):
        if request is None:
            return
        if self.task is None:
            self._launch(request)
        elif len(self.queue) < 2:
            self.queue.append(request)
        else:
            self._event(request, "unresolved", "补识别队列已满，主转录继续")
            return
        self._event(request, "pending")

    def observe(self, pcm, result):
        self.main.append({key: result[key] for key in ("turn", "action", "segments",
                                                     "start_seconds", "end_seconds")})
        while self.main and self.main[0]["end_seconds"] < result["end_seconds"] - 30:
            self.main.popleft()
        self._resolve_late_main(result)
        self._schedule(self.tracker.observe(pcm, result))
        self.collect()

    def _publish(self, request, span, index):
        when = next((r["turn"] for r in self.main
                     if r["end_seconds"] >= span["end_seconds"] - .001), request.last_turn)
        position = 10000 + index
        if "sort_before_turn" in span and when >= span["sort_before_turn"]:
            when = span["sort_before_turn"]
            position = span["sort_before_segment"] - .5
        elif "sort_after_turn" in span and when <= span["sort_after_turn"]:
            when = span["sort_after_turn"]
            position = span["sort_after_segment"] + .5
        item = dict(segment_id=len(self.ledger.segments) + 1, turn=when,
                    segment_index=position, origin="recovery", text=span["text"],
                    language=span["language"], model_speaker=None, display_speaker=None,
                    audio_start_seconds=span["start_seconds"], audio_end_seconds=span["end_seconds"])
        self.ledger.segments.append(item)
        self.added.append(dict(item))
        self.session.remember_recovery(item)
        return item

    def _resolve_late_main(self, result):
        # A delayed DIRECT may follow several WAITs; a later conversation can
        # also legitimately repeat these words. Match the first nearby DIRECT.
        remaining = []
        for request, span, item in self.recent:
            if result["end_seconds"] > request.end_seconds + LATE_MAIN_SECONDS + 1e-6:
                continue
            if result["action"] == "DIRECT":
                pieces = reconcile(replace(request, lead_text=""), [span], [result])
                if not pieces or pieces[0]["text"] != span["text"]:
                    item["superseded"] = True
                    self.removed.append(item["segment_id"])
                    self.session.forget_recovery(item["segment_id"])
                    for i, piece in enumerate(pieces):
                        self._publish(request, piece, i)
                continue
            remaining.append((request, span, item))
        self.recent = remaining

    def collect(self):
        if self.task is None or not self.task.done():
            return
        request = self.request
        try:
            words = self.task.result()
            spans = _spans(request, words)
            supported = getattr(self.session.backend, "allowed_tokens", None)
            for span in spans:
                if supported is not None and "<" + span["language"] + ">" not in supported:
                    span["language"] = "und"
            if not self.main or self.main[0]["turn"] > request.first_turn:
                self._event(request, "unresolved", "复核结果已超出近期音频窗口")
            else:
                kept = reconcile(request, spans, self.main)
                for i, span in enumerate(kept):
                    item = self._publish(request, span, i)
                    self.recent.append((request, span, item))
                if kept:
                    self._event(request, "recovered")
                elif spans:
                    self._event(request, "skipped", "没有可追加的文字")
                else:
                    self._event(request, "unresolved", "局部识别没有足够可信的文字")
        except Exception:
            LOG.exception("Gap ASR failed at %.2f–%.2f", request.start_seconds, request.end_seconds)
            self._event(request, "error", "局部识别失败，主转录继续")
        self.request = None
        self.task = None
        if self.queue:
            self._launch(self.queue.popleft())

    async def finish(self):
        # Give the final voiced fragment a chance even without a full WAIT streak.
        tail = self.tracker.finish()
        while self.task is not None:
            await asyncio.wait((self.task,))
            self.collect()
        self._schedule(tail)
        if self.task is not None:
            await asyncio.wait((self.task,))
            self.collect()

    def take_patch(self):
        result = {}
        if self.events:
            result["gap_events"], self.events = self.events, []
        if self.added:
            result["new_segments"], self.added = self.added, []
        if self.removed:
            result["removed_segment_ids"], self.removed = self.removed, []
        return result

    def close(self):
        self.queue.clear()
        if self.task is not None:
            self.task.cancel()
