import ast
import asyncio
from abc import ABC
import copy
from dataclasses import dataclass, field, fields
import io
import json
import os
from pathlib import Path
import runpy
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from asr_gspo.audio import encode_chunk_ref, materialize_chunk_refs
from asr_gspo.online import ASRStreamHooks, score_batch
from asr_gspo.protocol import DIRECT, EOS, WAIT, initial_messages
from asr_gspo.rollout_sync import make_completion_gather


def task(path="/samples/bilingual.wav"):
    return {"id": "bilingual", "audio_path": str(path), "duration_seconds": 1.5,
            "chunks": [{"start_seconds": 0., "end_seconds": 1., "input_seconds": 1.},
                       {"start_seconds": 1., "end_seconds": 1.5, "input_seconds": .5}],
            "reference_text": "请点OK", "spans": [{"lang": "zh", "text": "请点"}, {"lang": "en", "text": "OK"}]}


def request(source):
    return SimpleNamespace(messages=initial_messages(), audios=[encode_chunk_ref(source, 0)],
                           data_dict={"asr_task": json.dumps(source)}, images=[], videos=[])


def choice(text, reason="stop"):
    return SimpleNamespace(message=SimpleNamespace(content=text), token_ids=[123, 789],
                           logprobs={"content": [{"logprob": -.75}, {"logprob": -.25}]},
                           finish_reason=reason)


class TestOnline(unittest.TestCase):
    def setUp(self):
        self.task = task()
        self.req = request(self.task)
        self.scheduler = ASRStreamHooks()
        self.scheduler.max_turns = 3

    def start(self):
        asyncio.run(self.scheduler.on_trajectory_start([self.req]))

    def submit(self, text, turn, reason="stop"):
        output = choice(text, reason)
        self.req.messages.append({"role": "assistant", "content": text})
        result = asyncio.run(self.scheduler.on_turn_end(self.req, output, turn))
        if not result["done"]:
            self.scheduler.step(self.req, output, turn)
        return result

    def test_observations_eof_restoration_and_full_reward(self):
        self.start()
        self.assertNotIn("reference_text", json.dumps(self.req.data_dict))
        self.assertEqual(self.req.audios, [encode_chunk_ref(self.task, 0)])
        self.assertFalse(self.submit(DIRECT + "请点", 1)["done"])
        self.assertEqual(self.req.messages[-1], {"role": "user", "content": "<audio>"})
        self.assertEqual(self.req.audios, [encode_chunk_ref(self.task, i) for i in range(2)])
        self.assertFalse(self.submit(DIRECT + "OK", 2)["done"])
        self.assertEqual(self.req.messages[-1], {"role": "user", "content": EOS})
        self.assertEqual(len(self.req.audios), 2)
        infos = self.submit(WAIT, 3)["rollout_infos"]
        self.assertEqual(infos["audios"], self.req.audios)
        self.assertEqual([turn["audio_end_seconds"] for turn in infos["asr_trajectory"]["turns"]], [1., 1.5, 1.5])
        with TemporaryDirectory() as folder:
            rewards = score_batch([WAIT], [self.task], [infos], mode="quality", log_dir=folder)
            self.assertEqual(rewards, [1.0])
            row = json.loads(next(Path(folder).glob("rollouts-*.jsonl")).read_text(encoding="utf8"))
            self.assertEqual([t["response"] for t in row["turns"]], [DIRECT + "请点", DIRECT + "OK", WAIT])
        with self.assertRaisesRegex(ValueError, "same reference"):
            score_batch([WAIT], [{**self.task, "reference_text": "wrong"}], [infos], mode="quality")

    def test_no_reward_for_missing_rollout_metadata(self):
        with self.assertRaises(ValueError):
            score_batch([WAIT], [self.task], [{}], mode="quality")

    def test_invalid_text_ends_early_and_is_scored(self):
        self.start()
        raw = WAIT + "guessed"
        infos = self.submit(raw, 1)["rollout_infos"]
        self.assertEqual(infos["audios"], [encode_chunk_ref(self.task, 0)])
        self.assertEqual(infos["asr_trajectory"]["turns"][0]["response"], raw)
        self.assertLess(score_batch([raw], [self.task], [infos], mode="quality")[0], -2.5)

    def test_truncation_is_invalid_even_if_text_is_well_formed(self):
        self.start()
        infos = self.submit(DIRECT + "well", 1, "length")["rollout_infos"]
        self.assertIn("length", infos["asr_trajectory"]["invalid_reason"])
        self.assertLess(score_batch([WAIT], [self.task], [infos], mode="quality")[0], -2.5)

    def test_reject_text_at_eof(self):
        self.start()
        self.submit(WAIT, 1)
        self.submit(WAIT, 2)
        infos = self.submit(DIRECT + self.task["reference_text"], 3)["rollout_infos"]
        self.assertIn("WAIT", infos["asr_trajectory"]["invalid_reason"])
        self.assertLess(score_batch([WAIT], [self.task], [infos], mode="quality")[0], -2.5)

    def test_reject_future_chunks_or_invalid_clock(self):
        self.req.audios.append(encode_chunk_ref(self.task, 1))
        with self.assertRaisesRegex(ValueError, "first chunk"):
            self.start()
        self.task["chunks"][1]["start_seconds"] = .9
        self.req = request(self.task)
        with self.assertRaisesRegex(ValueError, "causal audio"):
            self.start()

    def test_bad_logprobs_are_runtime_errors(self):
        self.start()
        output = choice(WAIT)
        output.logprobs["content"].pop()
        with self.assertRaisesRegex(RuntimeError, "log probability"):
            asyncio.run(self.scheduler.on_turn_end(self.req, output, 1))

    def test_audio_materialization_uses_only_supplied_chunk(self):
        import numpy as np
        import soundfile as sf
        with TemporaryDirectory() as folder:
            path = Path(folder) / "two.wav"
            sf.write(path, np.r_[np.zeros(16000), np.ones(8000) * .5], 16000, subtype="FLOAT")
            source = task(path)
            source["chunks"][1]["input_seconds"] = .75
            source.update(audio_mtime_ns=path.stat().st_mtime_ns, audio_size_bytes=path.stat().st_size)
            refs = [encode_chunk_ref(source, i) for i in range(2)]
            data = materialize_chunk_refs(refs)
            first, _ = sf.read(io.BytesIO(data[0]))
            second, _ = sf.read(io.BytesIO(data[1]))
            np.testing.assert_equal(first, np.zeros(16000))
            np.testing.assert_equal(second, np.r_[np.ones(8000) * .5, np.zeros(4000)])
            with path.open("ab") as out:
                out.write(b"change")
            with self.assertRaisesRegex(ValueError, "changed after preparation"):
                materialize_chunk_refs(refs)

    def test_idle_tp_rank_waits_for_active_peers(self):
        group = object()
        trainer = SimpleNamespace(vllm_mode="colocate", vllm_tensor_parallel_size=4, vllm_tp_group=group)
        def gather(values, *, group):
            self.assertIs(group, trainer.vllm_tp_group)
            return [*values, True, False, False]
        completion = make_completion_gather(trainer, gather, SimpleNamespace(get_world_size=lambda **kw: 4))
        self.assertTrue(any(completion([False])))


@unittest.skipUnless(os.environ.get("ASR_GSPO_SWIFT_SOURCE"), "Set ASR_GSPO_SWIFT_SOURCE for upstream CPU contract")
class TestPinnedSwift(unittest.TestCase):
    def test_actual_driver_masks_ids_logprobs_and_audios(self):
        # Execute the installed framework's own loop and sample merge. Only
        # inference outputs and import-time GPU dependencies are replaced.
        root = Path(os.environ["ASR_GSPO_SWIFT_SOURCE"])
        ns = {"ABC": ABC, "asyncio": asyncio, "copy": copy, "deepcopy": copy.deepcopy,
              "RolloutOutput": SimpleNamespace, "get_token_backed_response_ids":
              lambda value: value if isinstance(value, list) else None,
              "dataclass": dataclass, "field": field, "fields": fields,
              "StandardKeys": ["messages", "images", "videos", "audios", "tools", "objects"],
              "_MULTIMODAL_KEYS": ("images", "videos", "audios")}
        def load(relative, names, owner=None):
            path = root / "swift" / relative
            tree = ast.parse(path.read_text(encoding="utf-8"))
            body = tree.body if owner is None else next(
                n.body for n in tree.body if isinstance(n, ast.ClassDef) and n.name == owner)
            nodes = [n for n in body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names]
            self.assertEqual({n.name for n in nodes}, set(names))
            future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
            selected = ast.fix_missing_locations(ast.Module(body=[future, *nodes], type_ignores=[]))
            exec(compile(selected, str(path), "exec"), ns)
        load("rollout/multi_turn.py", ["MultiTurnScheduler", "RolloutScheduler"])
        load("rollout/agent_loop.py", ["run_multi_turn", "_run_multi_turn_impl", "_identity_gather",
                                      "extract_logprobs_from_choice"])
        load("rl_core/data.py", ["OnPolicySample"])
        registrations = []
        qwen = SimpleNamespace(Qwen3OmniTemplate=object, QwenTemplateMeta=lambda *a, **kw: SimpleNamespace(
            stop_words=["<|endoftext|>"], **kw))
        fake = {"swift.rewards": SimpleNamespace(ORM=object, orms={}),
                "swift.megatron.callbacks": SimpleNamespace(MegatronCallback=object, megatron_callbacks_map={}),
                "swift.rollout": SimpleNamespace(MultiTurnScheduler=ns["MultiTurnScheduler"], multi_turns={}),
                "swift.template": SimpleNamespace(register_template=registrations.append),
                "swift.template.templates.qwen": qwen}
        with patch.dict(sys.modules, fake):
            plugin = runpy.run_path(str(Path(__file__).resolve().parents[1] / "swift_plugin.py"))
        self.assertIn(WAIT, registrations[0].stop_words)
        self.assertIn("omnicommit_export", fake["swift.megatron.callbacks"].megatron_callbacks_map)
        load("infer_engine/infer_engine.py", ["_get_stop_token_ids"], owner="InferEngine")
        tk = SimpleNamespace(encode=lambda t, **kw: [2] if t == WAIT else [6])
        self.assertIn(2, ns["_get_stop_token_ids"](SimpleNamespace(tokenizer=tk), registrations[0].stop_words))
        class Scheduler(ASRStreamHooks, ns["MultiTurnScheduler"]):
            pass
        tokens = {1: DIRECT, 2: WAIT, 3: "请点", 4: "OK", 5: "bad", 6: "<|im_end|>"}
        tok = SimpleNamespace(decode=lambda ids, **kw: "".join(tokens[i] for i in ids))
        scheduler = Scheduler(tokenizer=tok, max_turns=3)
        def output(ids):
            reply = choice(tok.decode(ids[:-1] if ids[-1] == 6 else ids))
            reply.token_ids = ids
            reply.logprobs = {"content": [{"logprob": -i / 10} for i in ids]}
            return SimpleNamespace(response=SimpleNamespace(choices=[reply]))
        source = task()
        reqs = [request(source) for _ in range(2)]
        asyncio.run(scheduler.on_trajectory_start(reqs))
        seen = []
        def generate(batch, config):
            seen.append(len(batch))
            if not batch:
                return []
            self.assertEqual(len(batch), 1)
            req = batch[0]
            self.assertEqual(req.messages[2]["content"], [2])
            return [output([2] if req.messages[-1]["content"] == EOS else [1, 3, 4, 6])]
        good, bad = ns["run_multi_turn"](reqs, [output([2]), output([5, 6])],
                                         scheduler, generate, None, max_turns=3)
        self.assertEqual(seen, [1, 1, 0])
        self.assertEqual(good.response_token_ids, [[2], [1, 3, 4, 6], [2]])
        self.assertEqual(good.response_loss_mask, [[1], [1, 1, 1, 1], [1]])
        self.assertEqual(bad.response_loss_mask, [[1, 1]])
        self.assertEqual(good.rollout_logprobs, [[-.2], [-.1, -.3, -.4, -.6], [-.2]])
        self.assertLess(score_batch(["bad"], [source], [bad.rollout_infos], mode="quality")[0], -2.5)
        first = request(source)
        sample = ns["OnPolicySample"].from_row({"audios": first.audios, "messages": first.messages,
                                                "asr_task": json.dumps(source)})
        sample.apply_rollout_output(rollout_output=good)
        self.assertEqual(sample.audios, [encode_chunk_ref(source, i) for i in range(2)])
        self.assertNotIn("asr_task", sample.to_template_dict())
        self.assertIn("asr_task", sample.to_reward_row())

        # Exercise the training-side token and mask conversion, including the
        # actual Megatron last_round default and explicit earlier-turn masks.
        import torch
        ns.update(torch=torch, ALL_BASE_STRATEGY=["all", "default", "last_round"],
                  ContextType=SimpleNamespace(RESPONSE="response", SUFFIX="suffix", OTHER="other"))
        load("rlhf_trainers/utils.py", ["replace_assistant_response_with_ids", "build_rollout_logps"])
        load("template/utils.py", ["get_last_user_round"])
        load("loss_scale/base.py", ["LossScale"])
        load("template/base.py", ["_encode_context_list"], owner="Template")
        messages = ns["replace_assistant_response_with_ids"](
            copy.deepcopy(sample.messages[1:]), sample.response_token_ids, sample.response_loss_mask)
        contexts = [m["content"] for m in messages]
        kinds = ["response" if m["role"] == "assistant" else "other" for m in messages]
        contexts, scales = ns["LossScale"]("last_round")(contexts, kinds, messages)
        # The middle user prompt contains a rendered (unsampled) chat separator
        # after WAIT. Upstream's regular SFT template promotes it into a label.
        def tokenize(text):
            return [6, 999] if text == "<audio>" else [999]
        ids, labels, _ = ns["_encode_context_list"](
            SimpleNamespace(is_binary_loss_scale=True, _tokenize=tokenize), contexts, scales)
        expected = [-100, -100, 2, -100, -100, 1, 3, 4, 6, -100, 2]
        plugin["OmniCommitASRTemplate"]._add_dynamic_eos(ids, labels, None, [6])
        self.assertEqual(labels, expected)
        load("template/base.py", ["_add_dynamic_eos"], owner="Template")
        sft_labels = labels.copy()
        ns["_add_dynamic_eos"].__func__(ids, sft_labels, None, [6])
        self.assertNotEqual(sft_labels, labels)
        mask = torch.tensor([labels]).ne(-100)
        logps = ns["build_rollout_logps"]([good.rollout_logprobs], mask, torch.device("cpu"))
        torch.testing.assert_close(logps[mask], torch.tensor([-.2, -.1, -.3, -.4, -.6, -.2]))
        self.assertEqual([i for i, y in zip(ids, labels) if y >= 0], [2, 1, 3, 4, 6, 2])


if __name__ == "__main__":
    unittest.main()
