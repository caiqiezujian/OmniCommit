"""Causal audio environment and outcome reward bridge for pinned Megatron-SWIFT."""
from __future__ import annotations

from hashlib import sha256
import json
import math
import os
from pathlib import Path

from .audio import encode_chunk_ref
from .protocol import EOS, WAIT, initial_messages, parse_reply
from .reward import score_trajectory


STATE_KEY = "_asr_rollout"


def _object(value, name):
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _task_key(task):
    canonical = json.dumps(task, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return sha256(canonical.encode("utf-8")).hexdigest()


def _finite(value):
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)


def _schedule(task):
    chunks = task.get("chunks")
    duration = task.get("duration_seconds")
    if not isinstance(chunks, list) or not chunks or not _finite(duration) or duration <= 0:
        raise ValueError("asr_task must have a positive duration and at least one audio chunk")
    refs, ends = [], []
    previous = 0.0
    for i, chunk in enumerate(chunks):
        start, end = chunk["start_seconds"], chunk["end_seconds"]
        if not _finite(start) or not _finite(end) or abs(start - previous) > 1e-8 or end <= start:
            raise ValueError(f"Invalid causal audio schedule at chunk {i}")
        refs.append(encode_chunk_ref(task, i))
        ends.append(end)
        previous = end
    if abs(ends[-1] - duration) > 1e-8:
        raise ValueError("The last audio observation must end at duration_seconds")
    return refs, ends


def _check_sample_details(choice):
    # Without these values, SWIFT can fall back to re-tokenizing text or omit
    # the per-token rollout probabilities. That would change the GSPO update.
    ids = choice.token_ids
    if not isinstance(ids, list) or not ids or any(type(i) is not int or i < 0 for i in ids):
        raise RuntimeError("ASR training requires exact sampled token IDs on every turn")
    probs = choice.logprobs.get("content") if isinstance(choice.logprobs, dict) else None
    if not isinstance(probs, list) or len(probs) != len(ids):
        raise RuntimeError("ASR training requires a sampled log probability for every generated token")
    if any(not isinstance(p, dict) or not _finite(p.get("logprob")) for p in probs):
        raise RuntimeError("Invalid sampled log probability")


class ASRStreamHooks:
    """Swift keeps sampled IDs, history and masks. The scheduler adds observations."""

    async def on_trajectory_start(self, requests):
        for req in requests:
            task = _object((req.data_dict or {}).get("asr_task"), "asr_task")
            refs, ends = _schedule(task)
            if self.max_turns is not None and len(refs) + 1 > self.max_turns:
                raise ValueError(f"{task['id']} requires {len(refs) + 1} turns including EOF")
            if req.messages != initial_messages() or req.audios != refs[:1] or req.images or req.videos:
                raise ValueError("First observation must be the explicit Third prompt with only the first chunk")
            # Keep references, switch/CTC annotations and future transcripts out
            # of the actor request. The trainer supplies its original asr_task
            # column to the ORM only after the rollout is complete.
            req.data_dict = {STATE_KEY: {"key": _task_key(task), "refs": refs, "ends": ends,
                                        "turns": [], "ready": 1, "done": False}}
            req.messages = initial_messages()
            req.audios = refs[:1]

    async def on_turn_end(self, infer_request, response_choice, current_turn):
        state = infer_request.data_dict[STATE_KEY]
        refs = state["refs"]
        if state["done"] or state["ready"] != current_turn or len(state["turns"]) != current_turn - 1:
            raise RuntimeError("Out-of-order ASR scheduler turn")
        eof = current_turn == len(refs) + 1
        observed = min(current_turn, len(refs))
        if infer_request.audios != refs[:observed]:
            raise RuntimeError("Observed audio differs from the causal chunk plan")
        observations = [m["content"] for m in infer_request.messages if m["role"] == "user"]
        if observations != ["<audio>"] * observed + ([EOS] if eof else []):
            raise RuntimeError("User observation history disagrees with the audio plan")
        _check_sample_details(response_choice)
        raw = response_choice.message.content
        if not isinstance(raw, str):
            raise RuntimeError("The inference backend did not provide generated text")
        finish = response_choice.finish_reason
        error = None
        if finish not in ("stop", "eos"):
            error = f"generation ended with {finish!r}"
        else:
            try:
                parse_reply(raw)
                if eof and raw != WAIT:
                    error = "Third EOF expects standalone WAIT"
            except ValueError as exc:
                error = str(exc)
        state["turns"].append({"kind": "eof" if eof else "audio", "response": raw,
                               "audio_end_seconds": state["ends"][observed - 1], "finish_reason": finish})
        state["ready"] = None
        state["done"] = eof or error is not None
        if not state["done"]:
            return {"done": False}
        # OnPolicySample.merge_rollout_output restores multimodal inputs only
        # via rollout_infos. This is required when training all generated turns.
        return {"done": True, "rollout_infos": {
            "audios": list(infer_request.audios), "asr_trajectory": {
                "task_key": state["key"], "turns": list(state["turns"]), "invalid_reason": error}}}

    def check_finished(self, infer_request, response_choice, current_turn):
        return infer_request.data_dict[STATE_KEY]["done"]

    def step(self, infer_request, response_choice, current_turn):
        state = infer_request.data_dict[STATE_KEY]
        if state["done"] or state["ready"] is not None or current_turn != len(state["turns"]):
            raise RuntimeError("Cannot advance ASR observation at this turn")
        if current_turn < len(state["refs"]):
            infer_request.audios.append(state["refs"][current_turn])
            observation = "<audio>"
        else:
            observation = EOS
        infer_request.messages.append({"role": "user", "content": observation})
        state["ready"] = current_turn + 1
        return {"infer_request": infer_request}


def score_batch(completions, asr_task, rollout_infos, *, mode, log_dir=None):
    """Score every full rollout. The last completion is normally just WAIT."""
    if not isinstance(completions, (list, tuple)):
        raise ValueError("Expected a batch of completions")
    for items in (asr_task, rollout_infos):
        if not isinstance(items, (list, tuple)) or len(items) != len(completions):
            raise ValueError("Expected one asr_task and one rollout_infos per completion")
    scores, records = [], []
    for task_raw, info_raw in zip(asr_task, rollout_infos):
        task = _object(task_raw, "asr_task")
        info = _object(info_raw, "rollout_infos")
        trajectory = _object(info.get("asr_trajectory"), "asr_trajectory")
        if trajectory.get("task_key") != _task_key(task):
            raise ValueError("Task and online trajectory do not belong to the same reference")
        turns = trajectory["turns"]
        # An early stop is legitimate only if an actual generated response was
        # truncated or violated the action grammar. A missing observation is a
        # pipeline failure and must not train the actor against a bogus reward.
        if not turns or len(turns) > len(task["chunks"]) + 1:
            raise ValueError("Missing ASR turns or unexpected extra observations")
        invalid_reason = trajectory["invalid_reason"]
        if invalid_reason is None and len(turns) != len(task["chunks"]) + 1:
            raise ValueError("An ASR trajectory without an invalid reply must reach EOF")
        score = score_trajectory(task, turns, mode=mode)
        if score["valid"] != (invalid_reason is None) or not _finite(score["reward"]):
            raise RuntimeError("Inconsistent scheduler and reward result")
        scores.append(score["reward"])
        if log_dir:
            records.append({"task_id": task["id"], "mode": mode, "turns": turns,
                            "invalid_reason": invalid_reason, **score})
    if records:
        path = Path(log_dir) / f"rollouts-rank{os.environ.get('RANK', '0')}-pid{os.getpid()}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as out:
            for record in records:
                out.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    return scores
