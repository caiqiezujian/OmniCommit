"""Verify frame-level state and conservative turn decisions independently of model weights."""
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fourth_asr.silero_vad import FRAME_SAMPLES, STATE_SHAPE, SileroVadModel


class Graph:
    def __init__(self, *, inputs=("input", "sr", "state"), state_shape=STATE_SHAPE):
        self.inputs = inputs
        self.state_shape = state_shape
        self.calls = []

    def get_inputs(self):
        return [SimpleNamespace(name=name) for name in self.inputs]

    def get_outputs(self):
        return [SimpleNamespace(name=name) for name in ("output", "stateN")]

    def run(self, names, feeds):
        assert names == ["output", "stateN"] and int(feeds["sr"]) == 16000
        assert feeds["input"].shape == (1, 576) and feeds["state"].shape == STATE_SHAPE
        self.calls.append((feeds["input"].copy(), int(feeds["state"][0, 0, 0])))
        # Only the last sample counts so padded speculative frames can miss a
        # genuine onset. This exposes the real edge case without model weights.
        score = np.abs(feeds["input"][:, -1])
        return score, np.full(self.state_shape, feeds["state"][0, 0, 0] + 1, np.float32)


def load_model(tmp_path, monkeypatch, graph=None, **options):
    graph = graph or Graph()
    def load(path, *, sess_options, providers):
        assert providers == ["CPUExecutionProvider"] and sess_options.intra_op_num_threads == 1
        return graph
    stub = SimpleNamespace(SessionOptions=SimpleNamespace, InferenceSession=load)
    monkeypatch.setitem(sys.modules, "onnxruntime", stub)
    path = tmp_path / "silero_vad.onnx"
    path.touch()
    return SileroVadModel(path, **options)


def test_peak_in_short_answer_and_single_ambiguous_frame(tmp_path, monkeypatch):
    model = load_model(tmp_path, monkeypatch, quiet_threshold=0)
    stream = model.new_stream()
    audio = np.zeros(16000, np.float32)
    audio[511] = .8
    assert stream.observe(audio).kind == "speech"
    stream.reset()
    audio[511] = .12
    result = stream.observe(audio)
    assert result.kind == "uncertain" and result.peak_probability == pytest.approx(.12)


def test_frame_carry_does_not_advance_state_with_padding(tmp_path, monkeypatch):
    graph = Graph()
    model = load_model(tmp_path, monkeypatch, graph)
    stream = model.new_stream()
    graph.calls.clear()
    old = np.zeros(16000, np.float32)
    old[-8:] = .07
    assert stream.observe(old).kind == "uncertain"
    assert len(stream.tail) == 128
    assert graph.calls[-1][1] == 31
    new = np.full(384, .8, np.float32)
    assert stream.observe(new).kind == "speech"
    assert graph.calls[-1][1] == 31
    np.testing.assert_array_equal(graph.calls[-1][0][0, 64:192], old[-128:])
    np.testing.assert_array_equal(graph.calls[-1][0][0, 192:], new)
    assert not stream.tail.size


def test_meetings_share_graph_only_and_context_spans_turns(tmp_path, monkeypatch):
    graph = Graph()
    model = load_model(tmp_path, monkeypatch, graph)
    a, b = model.new_stream(), model.new_stream()
    x = np.full(FRAME_SAMPLES, .04, np.float32)
    a.observe(x)
    assert b.observe(x).kind == "quiet"
    assert not graph.calls[-1][0][0, :64].any() and graph.calls[-1][1] == 0
    assert a.observe(x).kind == "quiet"
    assert graph.calls[-1][1] == 1
    np.testing.assert_array_equal(graph.calls[-1][0][0, :64], x[-64:])


def test_final_partial_onset_and_steady_floor(tmp_path, monkeypatch):
    stream = load_model(tmp_path, monkeypatch).new_stream()
    assert stream.observe(np.zeros(1), final=True).kind == "quiet"
    assert stream.observe(np.array([.0002]), final=True).kind == "uncertain"
    # A stable background can be rejected even when a 1s packet leaves a tail.
    assert stream.observe(np.full(16000, .002)).kind == "quiet"
    stream.reset()
    assert not stream.state.any() and not stream.context.any() and not stream.tail.size


def test_wrong_model_fails_at_startup(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="Unsupported Silero"):
        load_model(tmp_path, monkeypatch, Graph(inputs=("input", "h", "c")))
    with pytest.raises(ValueError, match="streaming probe failed"):
        load_model(tmp_path, monkeypatch, Graph(state_shape=(1, 32)))


def test_digital_silence_after_speech_does_not_extend_tail_due_to_recurrent_score(tmp_path, monkeypatch):
    graph = Graph()
    model = load_model(tmp_path, monkeypatch, graph)
    stream = model.new_stream()
    monkeypatch.setattr(model, "score", lambda frame, context, state: (.9, state + 1))
    assert stream.observe(np.ones(16000, np.float32) * .05).kind == "speech"
    old_state = stream.state.copy()
    obs = stream.observe(np.zeros(16000, np.float32))
    assert obs.kind == "quiet" and obs.peak_probability == .9
    assert (stream.state > old_state).all()
