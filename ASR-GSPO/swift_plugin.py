"""Online ASR extensions for ms-swift f15bb91439d756d4a366b51956d72d1f088b7af6."""
import os
from pathlib import Path
import sys

project = str(Path(__file__).resolve().parent)
if project not in sys.path:
    sys.path.insert(0, project)

from swift.rewards import ORM, orms
from swift.rollout import MultiTurnScheduler, multi_turns
from swift.template import register_template
from swift.template.templates.qwen import Qwen3OmniTemplate, QwenTemplateMeta

from asr_gspo.audio import SAMPLE_RATE, materialize_chunk_refs
from asr_gspo.online import ASRStreamHooks, score_batch
from asr_gspo.protocol import WAIT
from asr_gspo.rollout_sync import install_rollout_sync_patch


class OmniCommitASRTemplate(Qwen3OmniTemplate):
    def _preprocess_inputs(self, inputs):
        if getattr(self, "sampling_rate", SAMPLE_RATE) != SAMPLE_RATE:
            raise ValueError("OmniCommit Third requires 16 kHz audio")
        inputs.audios = materialize_chunk_refs(inputs.audios)
        return super()._preprocess_inputs(inputs)

    @staticmethod
    def _add_dynamic_eos(input_ids, labels, loss_scale, suffix_tokens_id):
        # Third returns immediately after a sampled WAIT. A later prompt adds
        # the normal ChatML separator; that separator has no policy logprob.
        # Keep its existing mask. Real sampled EOS tokens already carry masks.
        return None


class ASRStreamScheduler(ASRStreamHooks, MultiTurnScheduler):
    def __init__(self, *args, **kwargs):
        install_rollout_sync_patch()
        super().__init__(*args, **kwargs)


class OnlineASRReward(ORM):
    def __init__(self, **kwargs):
        super().__init__()
        self.mode = os.environ.get("ASR_GSPO_REWARD_MODE", "quality_latency")
        self.log_dir = os.environ.get("ASR_GSPO_LOG_DIR")

    def __call__(self, completions, asr_task=None, rollout_infos=None, **kwargs):
        return score_batch(completions, asr_task, rollout_infos, mode=self.mode, log_dir=self.log_dir)


meta = QwenTemplateMeta(
    "omnicommit_qwen3_omni", template_cls=OmniCommitASRTemplate, default_system=None,
    thinking_prefix="", non_thinking_prefix="", is_thinking=False)
# Native vLLM maps one-token stop words to stop_token_ids. The GRPO engine
# decodes exact sampled IDs and only strips the template suffix/native EOS.
meta.stop_words.append(WAIT)
register_template(meta)
multi_turns["asr_stream"] = ASRStreamScheduler
orms["asr_online"] = OnlineASRReward
