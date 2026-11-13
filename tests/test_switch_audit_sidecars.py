from copy import deepcopy
import json

from omnicommit.switch_audit_sidecars import Index, match_trajectory, pathkey


def test_index_seeks_exact_identity_and_preserves_duplicate_candidates(tmp_path):
    one = {"audio": "/data/a/file.wav", "id": 1}
    two = {"audios": ["a/file.wav"], "id": 2}
    other = {"audios": ["/data/b/file.wav"], "id": 3}
    path = tmp_path / "refs.jsonl"
    path.write_bytes(("\ufeff" + json.dumps(one) + "\n\nBAD\n" + json.dumps(two) + "\n" + json.dumps(other) +
                      '\n{"audio": "a.wav", "audios":["b.wav"]}\n').encode("utf-8"))
    with Index(path, audio_root="/data") as index:
        assert [i["line"] for i in index.issues] == [3, 6]
        assert index.get("file.wav") == []
        rows = index.get("/data/a/file.wav")
        assert rows == [one, {"audios": [one["audio"]], "id": 2}]
        assert index.get("/data/b/file.wav") == [other]
        assert all(isinstance(n, int) for offsets in index.offsets.values() for n in offsets)
    assert index.stream.closed
    assert pathkey(r"D:\meetings\A.wav") == pathkey("d:/meetings/a.wav")
    assert pathkey("/Meetings/A.wav") != pathkey("/meetings/a.wav")


def data():
    span = {"language": "ja", "speaker": "spk1", "text": "日本語。"}
    gold = [{"turn": 1, "event": "audio", "action": "WAIT", "start_seconds": 0, "end_seconds": .5, "segments": []},
            {"turn": 2, "event": "audio", "action": "DIRECT", "start_seconds": .5, "end_seconds": 1, "segments": [span]},
            {"turn": 3, "event": "end_of_audio", "action": "WAIT", "start_seconds": 1, "end_seconds": 1, "segments": []}]
    parsed = {"complete": True, "turns": gold, "audio": "/ja.wav"}
    trajectory = {"status": "ok", "audio": "/ja.wav", "source_segments": [{"attributed": [span]}],
                  "transcript": "<ja><spk1>日本語。", "duration_ms": 1000, "turns": [
                      {"start_ms": g["start_seconds"] * 1000, "end_ms": g["end_seconds"] * 1000, "event": g["event"],
                       "action": g["action"], "segments": g["segments"], "provenance": {"source_turn": g["turn"]}}
                      for g in gold]}
    return parsed, trajectory


def test_exact_trajectory_turns_only_and_aligned_reference():
    parsed, good = data()
    bad = deepcopy(good)
    bad["turns"][1]["segments"][0]["speaker"] = "spk2"
    prov, ref, issue = match_trajectory([bad, good], parsed)
    assert issue is None and prov == {k: {"source_turn": k} for k in (1, 2, 3)}
    assert ref["kind"] == "aligned_asr" and ref["segments"] == good["source_segments"]
    assert match_trajectory([good, good], parsed)[2] == "ambiguous_matching_trajectories"
    assert match_trajectory([bad], parsed) == ({}, None, "trajectory_grid_or_targets_do_not_match")
    assert match_trajectory([dict(good, audio="/other/ja.wav")], parsed)[2]
    assert match_trajectory([], parsed)[2] == "trajectory_not_found"


def test_window_uses_original_source_time_and_maps_provenance_to_local_turn():
    parsed, good = data()
    parsed.update(complete=False, turns=[dict(g, turn=i) for i, g in enumerate(parsed["turns"][1:], 1)])
    assert match_trajectory([good], parsed)[0] == {1: {"source_turn": 2}, 2: {"source_turn": 3}}
    shifted = deepcopy(good)
    shifted["turns"][1]["end_ms"] += 10
    assert match_trajectory([shifted], parsed)[2]
