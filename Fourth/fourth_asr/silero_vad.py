"""CPU Silero streaming VAD for 16 kHz mono audio, with state per meeting."""
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Literal, Optional

import numpy as np


FRAME_SAMPLES = 512
CONTEXT_SAMPLES = 64
STATE_SHAPE = (2, 1, 128)
SPLIT_STATE_SHAPE = (2, 1, 64)
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
        inputs = {arg.name: arg for arg in self.session.get_inputs()}
        outputs = {arg.name for arg in self.session.get_outputs()}
        self.frame_samples = FRAME_SAMPLES
        if inputs.keys() == {"input", "sr", "state"} and {"output", "stateN"} <= outputs:
            self.split_state = False
            self.context_samples = CONTEXT_SAMPLES
        elif inputs.keys() == {"input", "sr", "h", "c"} and {"output", "hn", "cn"} <= outputs:
            # The v4 graph consumes raw audio and two LSTM states. Its
            # upstream 16 kHz streaming wrapper uses 512 for a dynamic window.
            self.split_state = True
            self.context_samples = 0
            shape = getattr(inputs["input"], "shape", None)
            if shape and isinstance(shape[-1], int):
                self.frame_samples = shape[-1]
                if self.frame_samples not in (512, 1024, 1536):
                    raise ValueError(f"Unsupported Silero h/c 16 kHz frame length {self.frame_samples}: {path}")
        else:
            raise ValueError(f"Unsupported Silero streaming ONNX graph {path}: "
                             f"inputs={sorted(inputs)}, outputs={sorted(outputs)}; "
                             "expected input, sr with either state -> stateN or h, c -> hn, cn and output")
        sr_shape = getattr(inputs["sr"], "shape", None)
        self.sample_rate = SAMPLE_RATE.reshape(1) if sr_shape is not None and len(sr_shape) == 1 else SAMPLE_RATE
        self.quiet_threshold = quiet_threshold
        self.speech_threshold = speech_threshold
        # Also exercise feedback of the graph's outputs into the next frame.
        try:
            self.new_stream().observe(np.zeros(2 * self.frame_samples, dtype=np.float32))
        except Exception as exc:
            raise ValueError(f"Silero 16 kHz streaming probe failed for {path}: {exc}") from exc
        layout = "h/c" if self.split_state else "state"
        print(f"[Silero VAD] ready: layout={layout} rate=16000 frame_samples={self.frame_samples} "
              f"provider=CPU model={path}", file=sys.stderr, flush=True)

    def new_stream(self):
        return SileroVadStream(self)

    def initial_state(self):
        if self.split_state:
            return (np.zeros(SPLIT_STATE_SHAPE, dtype=np.float32),
                    np.zeros(SPLIT_STATE_SHAPE, dtype=np.float32))
        return np.zeros(STATE_SHAPE, dtype=np.float32)

    def score(self, frame, context, state):
        if self.split_state:
            h, c = state
            score, hn, cn = self.session.run(["output", "hn", "cn"], {
                "input": frame[None, :], "sr": self.sample_rate, "h": h, "c": c})
            if hn.shape != SPLIT_STATE_SHAPE or cn.shape != SPLIT_STATE_SHAPE:
                raise ValueError(f"Unexpected Silero h/c state shapes: {hn.shape}, {cn.shape}")
            next_state = hn, cn
        else:
            inputs = {"input": np.concatenate((context, frame))[None, :],
                      "sr": self.sample_rate, "state": state}
            score, next_state = self.session.run(["output", "stateN"], inputs)
            if next_state.shape != STATE_SHAPE:
                raise ValueError(f"Unexpected Silero state shape: {next_state.shape}")
        p = float(score.item())
        if not 0 <= p <= 1:
            raise ValueError(f"Unexpected Silero speech probability: {p}")
        return p, next_state


class SileroVadStream:
    def __init__(self, model):
        self.model = model
        self.reset()

    def reset(self):
        self.state = self.model.initial_state()
        self.context = np.zeros(self.model.context_samples, dtype=np.float32)
        self.tail = np.empty(0, dtype=np.float32)
        self.last_frame_peak = 0.

    def observe(self, audio, *, final=False):
        """Score each real frame once, including frames that span two ASR turns."""
        samples = np.asarray(audio, dtype=np.float32)
        if samples.ndim != 1 or not len(samples) or not np.isfinite(samples).all():
            raise ValueError("Silero expects nonempty finite mono audio")
        pcm = np.concatenate((self.tail, samples))
        frame_size = self.model.frame_samples
        end = len(pcm) // frame_size * frame_size
        peak = None
        for offset in range(0, end, frame_size):
            frame = pcm[offset:offset + frame_size]
            p, self.state = self.model.score(frame, self.context, self.state)
            if self.model.context_samples:
                self.context = frame[-self.model.context_samples:]
            peak = p if peak is None else max(peak, p)
        if end:
            self.last_frame_peak = float(np.max(np.abs(pcm[end - frame_size:end])))
            self.context = self.context.copy()
        self.tail = pcm[end:].copy()

        # At 16 kHz an ASR turn of 1 s ends 128 samples into a VAD frame.
        # Preview a non-silent remainder, but never advance recurrent state with
        # padding. A new onset there might be too short for a reliable score.
        possible_onset = False
        if self.tail.size and np.any(self.tail):
            padded = np.pad(self.tail, (0, frame_size - len(self.tail)))
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
