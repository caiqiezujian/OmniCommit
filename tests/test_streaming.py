"""Real session-state tests; no model, CUDA, or external services."""
import json
from pathlib import Path

import numpy as np
import pytest

from omnicommit.inference_backend import multimodal_messages, validate_token_manifest
from omnicommit.streaming import (StreamOptions, StreamSession, ReplayBackend,
                                 ContextBudgetError, parse_reply, WAIT, DIRECT)
from omnicommit.stream_cli import main


def pcm(seconds=.5):
    return np.full(round(16000*seconds), .1, dtype=np.float32)


@pytest.mark.parametrize("step", [.5, .8, 1.0])
def test_configured_chunk_step_and_real_short_tail(step):
    from argparse import ArgumentParser
    from omnicommit.stream_cli import add_runtime_arguments
    parser = ArgumentParser()
    add_runtime_arguments(parser)
    assert parser.parse_args(["--chunk-seconds", str(step)]).chunk_seconds == step
    session = StreamSession(ReplayBackend([DIRECT + "<en><spk1>First ", WAIT,
                                           DIRECT + "<en><spk1>tail.", WAIT]),
                            StreamOptions(chunk_seconds=step))
    assert session.step(pcm(step))["end_seconds"] == step
    assert session.step(pcm(step))["end_seconds"] == 2 * step
    assert session.step(pcm(.1))["end_seconds"] == pytest.approx(2 * step + .1)
    assert len(session.history[-1].audio) == 8000  # padding is not source time
    final = session.step(end_of_audio=True)
    assert final["end_seconds"] == pytest.approx(2 * step + .1)
    assert final["event"] == "end_of_audio"


def test_wait_and_full_attribution_parser():
    assert parse_reply(WAIT)["segments"] == []
    result = parse_reply(DIRECT + "<zh><spk1>你好。<en><spk1>Hello.\n<en><spk2>Yes.")
    assert [s["speaker"] for s in result["segments"]] == ["spk1", "spk1", "spk2"]
    assert result["segments"][1]["text"] == "Hello.\n"
    for bad in ("", WAIT+"x", DIRECT+"hello", DIRECT+"<en><spk1>", DIRECT+"<en><spk1>hi<|WAIT|>"):
        with pytest.raises(ValueError):
            parse_reply(bad)


def test_direct_preserves_boundary_whitespace():
    first=parse_reply(DIRECT+"<en><spk1>Please ")
    second=parse_reply(DIRECT+"<en><spk1>confirm.\n")
    assert first["segments"][0]["text"]+second["segments"][0]["text"]=="Please confirm.\n"
    assert parse_reply("  "+DIRECT+"<en><spk1> leading ")["segments"][0]["text"]==" leading "
    assert parse_reply(" \n"+WAIT+" \n")["text"]==WAIT


def test_direct_does_not_clear_unverified_audio():
    session = StreamSession(ReplayBackend([DIRECT+"<en><spk1>A", WAIT]),
                            StreamOptions(recent_seconds=.5,max_context_seconds=1,max_pending_seconds=1))
    session.step(pcm())
    session.step(pcm())
    with pytest.raises(ContextBudgetError):
        session.step(pcm())
    assert session.heard_seconds == 1
    assert len(session.history) == 2


def test_verified_window_preserves_pending_and_registry():
    session = StreamSession(ReplayBackend([DIRECT+"<en><spk1>A"]*6),
                            StreamOptions(recent_seconds=.5,max_context_seconds=1,max_pending_seconds=1))
    for i in range(6):
        session.step(pcm())
        session.verify_commit(session.heard_seconds, verified=True)
    assert len(session.history) == 2
    assert session.speakers == ["spk1"]
    assert not session.heuristic_eviction


def test_rolling_is_explicit_and_wait_chain_not_evicted():
    options=StreamOptions(context_policy="rolling",rolling_tail_seconds=.5,
        recent_seconds=.5,max_context_seconds=1.5,max_pending_seconds=1.5)
    session=StreamSession(ReplayBackend([DIRECT+"<en><spk1>A"]*4+[WAIT]*5),options)
    for _ in range(6):
        result=session.step(pcm())
    assert result["heuristic_eviction"] is True
    with pytest.raises(ContextBudgetError): session.step(pcm())


def test_rolling_budget_reports_the_blocked_audio_interval_after_waits():
    session = StreamSession(ReplayBackend([DIRECT + "<en><spk1>Hello"] * 64 + [WAIT] * 56),
                            StreamOptions(context_policy="rolling"))
    for _ in range(120):
        session.step(pcm())
    with pytest.raises(ContextBudgetError, match=r"30\.50s > 30s; rolling policy, commit watermark 30\.00s, attempted audio end 60\.50s"):
        session.step(pcm())
    assert session.heard_seconds == 60


def test_reference_is_causal_verified_and_separate():
    replies=[DIRECT+"<zh><spk1>你好",DIRECT+"<en><spk2>Hello",DIRECT+"<en><spk1>Hello again"]
    session=StreamSession(ReplayBackend(replies),StreamOptions(recent_seconds=.5))
    session.step(pcm())
    with pytest.raises(ValueError):
        session.register_reference("spk1","zh","你好",0,.5)
    with pytest.raises(ValueError):
        session.register_reference("spk1","zh","你好",0,1,verified=True,clean=True,overlap=False)
    session.register_reference("spk1","zh","你好",0,.5,verified=True,clean=True,overlap=False)
    session.verify_commit(.5,verified=True)
    session.step(pcm())
    session.verify_commit(1,verified=True)
    session.step(pcm())
    assert session.speakers==["spk1","spk2"]
    messages,audios=session._context(pcm())
    assert "Speaker reference" in messages[1]["content"]
    assert messages[1]["content"].count("<audio>")==2
    assert len(audios)==len(session.history)+2
    assert session.references["spk1"].audio.shape==(8000,)
    session.close()
    assert not session.references and not session.history


def test_unavailable_reference_cannot_be_fabricated():
    session=StreamSession(ReplayBackend([DIRECT+"<en><spk1>A"]*4),StreamOptions(recent_seconds=.5))
    for _ in range(4):
        session.step(pcm());session.verify_commit(session.heard_seconds,verified=True)
    with pytest.raises(ValueError):
        session.register_reference("spk1","en","A",0,.5,verified=True,clean=True,overlap=False)


def test_eos_is_text_only_and_short_tail_uses_true_source_duration():
    class Recording(ReplayBackend):
        def generate(self,messages,audios):
            self.messages,self.audios=messages,audios
            return super().generate(messages,audios)
    backend=Recording([WAIT,DIRECT+"<en><spk1>Hello"])
    session=StreamSession(backend)
    result=session.step(pcm(.1))
    assert result["end_seconds"]==.1
    assert len(backend.audios[-1])==8000
    with pytest.raises(ValueError): session.step(pcm())
    result=session.step(end_of_audio=True)
    assert backend.messages[-1]["content"]=="<end_of_audio>"
    assert len(backend.audios)==1
    assert result["start_seconds"]==result["end_seconds"]==.1
    with pytest.raises(RuntimeError):session.step(end_of_audio=True)


def test_model_error_never_becomes_wait_or_commits_audio():
    session=StreamSession(ReplayBackend(["broken"]))
    with pytest.raises(ValueError):session.step(pcm())
    assert session.heard_seconds==0 and session.failed


def test_speaker_registry_cannot_skip_or_recycle():
    session=StreamSession(ReplayBackend([DIRECT+"<en><spk2>No"]),StreamOptions(max_speakers=1))
    with pytest.raises(ValueError):session.step(pcm())


def test_noncausal_watermarks_rejected():
    session=StreamSession(ReplayBackend())
    session.step(pcm())
    for x in (1,float("nan"),True,-1):
        with pytest.raises(ValueError):session.verify_commit(x,verified=True)
    with pytest.raises(ValueError):session.verify_commit(.5)


def test_last_committed_attribution_survives_trim_wait_and_reference_change():
    from omnicommit.runtime_protocol import last_committed_text
    session=StreamSession(ReplayBackend([
        DIRECT+"<zh><spk1>您好。", DIRECT+"<en><spk1>Hello.",
        DIRECT+"<en><spk2>Yes.", WAIT, WAIT]),StreamOptions(recent_seconds=.5))
    assert last_committed_text(None,None)==""
    session.step(pcm());session.verify_commit(.5,verified=True)
    session.register_reference("spk1","zh","您好。",0,.5,verified=True,clean=True,overlap=False)
    session.step(pcm());session.verify_commit(1,verified=True)
    assert (session.last_committed_language,session.last_committed_speaker)==("en","spk1")
    session.step(pcm());session.verify_commit(1.5,verified=True)
    session.register_reference("spk2","en","Yes.",1,1.5,verified=True,clean=True,overlap=False)
    session.step(pcm());session.step(pcm())
    assert session.history[0].start>=1
    assert (session.last_committed_language,session.last_committed_speaker)==("en","spk2")
    messages,_=session._context(pcm())
    assert last_committed_text("en","spk2") in messages[0]["content"]
    assert "<zh><spk1>您好。" in messages[1]["content"]
    assert "not necessarily the latest heard" in messages[0]["content"]


def test_manifest_prevents_tokenizer_id_drift():
    tokens=[WAIT,DIRECT,"<en>","<spk1>"]
    class Tokenizer:
        all_special_tokens=tokens
        def __len__(self):return 20
        def encode(self,token,**kwargs):return [10+tokens.index(token)]
    manifest={"schema_version":1,"tokens":tokens,"token_ids":dict(zip(tokens,range(10,14))),"tokenizer_length":20}
    assert validate_token_manifest(Tokenizer(),manifest)==set(tokens)
    manifest["token_ids"][WAIT]=19
    with pytest.raises(ValueError):validate_token_manifest(Tokenizer(),manifest)


def test_multimodal_placeholder_order():
    messages=multimodal_messages([{"role":"user","content":"ref<audio>live<audio>"}])
    assert [p["type"] for p in messages[0]["content"]]==["text","audio","text","audio"]


def test_file_cli_partial_eof(tmp_path):
    import soundfile as sf
    audio=tmp_path/"input.wav";sf.write(audio,pcm(.6),16000)
    replies=tmp_path/"replies.json";replies.write_text(json.dumps([WAIT,WAIT,DIRECT+"<en><spk1>Hello"]),encoding="utf-8")
    out=tmp_path/"out.jsonl"
    assert main([str(audio),"--output",str(out),"--backend","replay","--replies",str(replies)])==0
    rows=[json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert [r["event"] for r in rows]==["audio","audio","end_of_audio"]
    assert rows[-1]["end_seconds"]==.6


def test_web_pcm_and_eof(tmp_path):
    pytest.importorskip("fastapi");pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    from omnicommit.web_service import create_app
    app=create_app(lambda:ReplayBackend([WAIT,DIRECT+"<en><spk1>Hello"]),StreamOptions())
    with TestClient(app) as client:
        assert client.get("/health").json()["ready"] is True
        with client.websocket_connect("/stream") as socket:
            socket.send_json({"type":"start","sample_rate":16000})
            assert socket.receive_json()["type"]=="ready"
            socket.send_bytes(np.zeros(8000,dtype="<i2").tobytes())
            assert socket.receive_json()["action"]=="WAIT"
            socket.send_json({"type":"end"})
            assert socket.receive_json()["action"]=="DIRECT"


def test_web_session_isolation_without_access_token():
    pytest.importorskip("fastapi");pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    from omnicommit.web_service import create_app
    app=create_app(ReplayBackend,StreamOptions(),max_sessions=1)
    with TestClient(app) as client:
        with client.websocket_connect("/stream") as first:
            first.send_json({"type":"start","sample_rate":16000})
            assert first.receive_json()["type"]=="ready"
            with client.websocket_connect("/stream") as second:
                second.send_json({"type":"start","sample_rate":16000})
                assert "capacity" in second.receive_json()["error"]
            first.send_bytes(np.zeros(8000,dtype="<i2").tobytes())
            assert first.receive_json()["end_seconds"]==.5
            first.send_json({"type":"end"})
            first.receive_json()
        with client.websocket_connect("/stream") as fresh:
            fresh.send_json({"type":"start","sample_rate":16000})
            assert fresh.receive_json()["type"]=="ready"
            fresh.send_bytes(np.zeros(8000,dtype="<i2").tobytes())
            assert fresh.receive_json()["start_seconds"]==0
            fresh.send_json({"type":"end"});fresh.receive_json()


def test_web_bad_control_keeps_session_and_serializes_model():
    pytest.importorskip("fastapi");pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    from omnicommit.web_service import create_app
    import threading,time
    lock=threading.Lock()
    class Backend(ReplayBackend):
        def generate(self,messages,audios):
            assert lock.acquire(blocking=False), "Concurrent model access"
            try:
                time.sleep(.02)
                return super().generate(messages,audios)
            finally:lock.release()
    with TestClient(create_app(Backend,StreamOptions(),max_sessions=2)) as client:
        with client.websocket_connect("/stream") as a,client.websocket_connect("/stream") as b:
            for socket in (a,b):
                socket.send_json({"type":"start","sample_rate":16000});socket.receive_json()
            a.send_json({"type":"control","op":"commit","through_seconds":100,"verified":True})
            assert a.receive_json()["type"]=="control_error"
            for socket in (a,b):socket.send_bytes(np.zeros(8000,dtype="<i2").tobytes())
            for socket in (a,b):assert socket.receive_json()["start_seconds"]==0
            for socket in (a,b):socket.send_json({"type":"end"})
            for socket in (a,b):assert socket.receive_json()["event"]=="end_of_audio"


@pytest.mark.parametrize("cache_stats", [{}, {"num_cached_tokens": 4}])
def test_vllm_request_cache_keys_exact_count_and_wait(cache_stats):
    from types import SimpleNamespace
    from omnicommit.vllm_backend import VLLMBackend
    backend=VLLMBackend.__new__(VLLMBackend)
    class Tok:
        def apply_chat_template(self,*args,**kwargs):return "prompt"
        def encode(self,*args,**kwargs):return [1,2,3]
        def decode(self,ids,**kwargs):
            assert kwargs["skip_special_tokens"] is False
            return WAIT if ids==[9] else "bad"
    class Engine:
        def generate(self,requests,*args,**kwargs):
            self.request=requests[0]
            return [SimpleNamespace(**cache_stats,prompt_token_ids=list(range(7)),outputs=[SimpleNamespace(
                token_ids=[9],finish_reason="stop")])]
    backend.tokenizer=Tok();backend.llm=Engine();backend.TokensPrompt=lambda **kw:kw
    backend.max_audio_turns=8;backend.max_audio_seconds=5;backend.max_new_tokens=8
    backend.max_context_tokens=32;backend.stop_ids=[8];backend.wait_id=9;backend.sampling=object()
    backend.audio_counter=SimpleNamespace(count=lambda a,fn:(2,False))
    result=backend.generate([{"role":"user","content":"ref<audio>live<audio>"}],[pcm(),pcm()])
    request=backend.llm.request
    assert request["multi_modal_uuids"]["audio"][0]==request["multi_modal_uuids"]["audio"][1]
    assert request["prompt_token_ids"]==[1,2,3]
    assert result=={"text":WAIT,"prompt_tokens":7,"num_cached_tokens":cache_stats.get("num_cached_tokens")}
    backend.max_context_tokens=10
    with pytest.raises(ContextBudgetError):backend.generate([{"role":"user","content":"<audio>"}],[pcm()])
