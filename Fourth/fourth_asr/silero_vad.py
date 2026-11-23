"""CPU Silero streaming VAD for 16 kHz mono audio, with state per meeting."""
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import numpy as np


FRAME_SAMPLES = 512
CONTEXT_SAMPLES = 64
STATE_SHAPE = (2, 1, 128)
SAMPLE_RATE = np.array(16000, dtype=np.int64)


@dataclass(frozen=True)
class VadObservation:
    kind: Literal["quiet", "uncertain", "speech"]
    peak_probability: Optional[float]


class SileroVadModel:
    """Share one ONNX session between streams. No recurrent state lives here."""

    def __init__(self, path, *, quiet_threshold=0.10, speech_threshold=0.50):
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Silero VAD ONNX file is missing: {path}")
        if not 0 <= quiet_threshold < speech_threshold <= 1:
            raise ValueError("Silero thresholds require 0 <= quiet < speech <= 1")
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("Silero VAD requires onnxruntime in the web environment") from exc
        options = ort.SessionOptions()
        options.inter_op_num_threads = options.intra_op_num_threads = 1
        options.log_severity_level = 3
        self.session = ort.InferenceSession(
            str(path), sess_options=options, providers=["CPUExecutionProvider"])
        inputs = {arg.name for arg in self.session.get_inputs()}
        outputs = {arg.name for arg in self.session.get_outputs()}
        if inputs != {"input", "sr", "state"} or not {"output", "stateN"} <= outputs:
            raise ValueError(f"Unsupported Silero streaming ONNX graph {path}: "
                             f"inputs={sorted(inputs)}, outputs={sorted(outputs)}; "
                             "expected official silero_vad.onnx (input, sr, state -> output, stateN)")
        self.quiet_threshold = quiet_threshold
        self.speech_threshold = speech_threshold
        # A matching filename or signature alone cannot verify frame/state dimensions.
        try:
            self.new_stream().observe(np.zeros(FRAME_SAMPLES, dtype=np.float32))
        except Exception as exc:
            raise ValueError(f"Silero 16 kHz streaming probe failed for {path}: {exc}") from exc

    def new_stream(self):
        return SileroVadStream(self)

    def score(self, frame, context, state):
        inputs = {"input": np.concatenate((context, frame))[None, :],
                  "sr": SAMPLE_RATE, "state": state}
        score, next_state = self.session.run(["output", "stateN"], inputs)
        p = float(score.item())
        if not 0 <= p <= 1 or next_state.shape != STATE_SHAPE:
            raise ValueError(f"Unexpected Silero output: probability={p}, state={next_state.shape}")
        return p, next_state


class SileroVadStream:
    def __init__(self, model):
        self.model = model
        self.reset()

    def reset(self):
        self.state = np.zeros(STATE_SHAPE, dtype=np.float32)
        self.context = np.zeros(CONTEXT_SAMPLES, dtype=np.float32)
        self.tail = np.empty(0, dtype=np.float32)
        self.last_frame_peak = 0.

    def observe(self, audio, *, final=False):
        """Score each real frame once, including frames that span two ASR turns."""
        samples = np.asarray(audio, dtype=np.float32)
        if samples.ndim != 1 or not len(samples) or not np.isfinite(samples).all():
            raise ValueError("Silero expects nonempty finite mono audio")
        pcm = np.concatenate((self.tail, samples))
        end = len(pcm) // FRAME_SAMPLES * FRAME_SAMPLES
        peak = None
        for offset in range(0, end, FRAME_SAMPLES):
            frame = pcm[offset:offset + FRAME_SAMPLES]
            p, self.state = self.model.score(frame, self.context, self.state)
            self.context = frame[-CONTEXT_SAMPLES:]
            peak = p if peak is None else max(peak, p)
        if end:
            self.last_frame_peak = float(np.max(np.abs(pcm[end - FRAME_SAMPLES:end])))
            self.context = self.context.copy()
        self.tail = pcm[end:].copy()

        # At 16 kHz an ASR turn of 1 s ends 128 samples into a VAD frame.
        # Preview a non-silent remainder, but never advance recurrent state with
        # padding. A new onset there might be too short for a reliable score.
        possible_onset = False
        if self.tail.size and np.any(self.tail):
            padded = np.pad(self.tail, (0, FRAME_SAMPLES - len(self.tail)))
            p, _ = self.model.score(padded, self.context, self.state)
            peak = p if peak is None else max(peak, p)
            possible_onset = np.max(np.abs(self.tail)) > 2 * self.last_frame_peak
        if final:
            self.reset()
        # The recurrent model can remain active after the previous word. With
        # literal digital silence, advance its state but do not grant another
        # speech turn just for that residual score. The session owns tail grace.
        if not np.any(samples):
            return VadObservation("quiet", peak)
        if peak is not None and peak >= self.model.speech_threshold:
            return VadObservation("speech", peak)
        if possible_onset or (peak is not None and peak > self.model.quiet_threshold):
            return VadObservation("uncertain", peak)
        return VadObservation("quiet", peak)
