"""Exercise the split hidden/cell state interface used by older Silero ONNX exports."""
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fourth_asr.silero_vad import SileroVadModel


class SplitGraph:
    def __init__(self, advertised_window="samples", sample_rate_shape=()):
        self.window = advertised_window if isinstance(advertised_window, int) else 512
        self.sr_shape = sample_rate_shape
        self.inputs = {"c": [2, "batch", 64], "h": [2, "batch", 64],
                       "input": ["batch", advertised_window], "sr": list(sample_rate_shape)}
        self.calls = []

    def get_inputs(self):
        return [SimpleNamespace(name=name, shape=shape) for name, shape in self.inputs.items()]

    def get_outputs(self):
        return [SimpleNamespace(name=name) for name in ("cn", "output", "hn")]

    def run(self, names, feeds):
        assert set(feeds) == set(self.inputs)
        audio, sr, h, c = (feeds[key] for key in ("input", "sr", "h", "c"))
        assert sr.dtype == np.int64 and sr.shape == self.sr_shape and sr.item() == 16000
        assert audio.shape == (1, self.window) and audio.dtype == np.float32
        assert h.shape == c.shape == (2, 1, 64) and h.dtype == c.dtype == np.float32
        self.calls.append((audio[0].copy(), h[0, 0, 0], c[0, 0, 0]))
        outputs = {"output": np.abs(audio[:, -1]), "hn": h + 1, "cn": c + 7}
        return [outputs[name] for name in names]


def load_graph(graph, tmp_path, monkeypatch):
    def load(path, *, sess_options, providers):
        assert providers == ["CPUExecutionProvider"]
        return graph

    ort = SimpleNamespace(SessionOptions=SimpleNamespace, InferenceSession=load)
    monkeypatch.setitem(sys.modules, "onnxruntime", ort)
    path = tmp_path / "silero_vad.onnx"
    path.touch()
    model = SileroVadModel(path)
    graph.calls.clear()  # Startup probes use a different stream from any meeting.
    return model


@pytest.mark.parametrize("window,sr_shape", [("samples", ()), (1024, (1,)), (1536, ())])
def test_split_graph_streaming_windows_and_independent_states(window, sr_shape, tmp_path, monkeypatch):
    graph = SplitGraph(window, sr_shape)
    model = load_graph(graph, tmp_path, monkeypatch)
    meeting, other = model.new_stream(), model.new_stream()
    frame = np.full(graph.window, .75, np.float32)

    assert meeting.observe(frame).kind == "speech"
    assert meeting.observe(frame).kind == "speech"
    assert other.observe(frame).kind == "speech"
    assert meeting.observe(frame, final=True).kind == "speech"
    assert meeting.observe(frame).kind == "speech"
    assert [(h, c) for _, h, c in graph.calls] == [(0, 0), (1, 7), (0, 0), (2, 14), (0, 0)]
    for audio, _, _ in graph.calls:
        np.testing.assert_array_equal(audio, frame)


def test_split_graph_does_not_commit_padded_frame_state(tmp_path, monkeypatch):
    graph = SplitGraph()
    stream = load_graph(graph, tmp_path, monkeypatch).new_stream()
    pcm = np.zeros(16000, np.float32)
    pcm[-5:] = .06
    assert stream.observe(pcm).kind == "uncertain"
    preview, h, c = graph.calls[-1]
    assert (h, c) == (31, 217)
    np.testing.assert_array_equal(preview[:128], pcm[-128:])
    assert not preview[128:].any()

    continuation = np.full(384, .7, np.float32)
    assert stream.observe(continuation).kind == "speech"
    actual, h, c = graph.calls[-1]
    assert (h, c) == (31, 217)
    np.testing.assert_array_equal(actual, np.concatenate((pcm[-128:], continuation)))
