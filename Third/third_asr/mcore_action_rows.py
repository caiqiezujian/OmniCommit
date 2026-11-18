"""Train Third's action rows with SWIFT Megatron and export native PEFT adapters."""

from contextlib import contextmanager
from functools import wraps
import logging
import os
from pathlib import Path

import torch
from torch import nn

from .action_rows import ActionRows
from .action_rows_manifest import read_action_rows_manifest, write_action_rows_manifest
from .protocol import protocol_for_tokens


class _VocabularyRows(nn.Module):
    def __init__(self, embedding_weight, head_weight, ids):
        super().__init__()
        self.input_rows = ActionRows(embedding_weight, ids)
        self.output_rows = ActionRows(head_weight, ids)

    def sharded_state_dict(self, *args, **kwargs):
        # The MCore base has no residuals. This recipe exports the effective
        # matrices in HF safetensors, and does not save a native optimizer.
        return {}


def _vocab_modules(model):
    language_model = getattr(model, "language_model", model)
    if getattr(language_model, "share_embeddings_and_output_weights", False):
        raise ValueError("Third action rows require an untied vocabulary head")
    embedding = language_model.embedding.word_embeddings
    head = language_model.output_layer
    for wrapper in (embedding, head):
        if not hasattr(wrapper, "modules_to_save") or "default" not in wrapper.modules_to_save:
            raise ValueError("Expected PEFT modules_to_save for word_embeddings and output_layer")
    return embedding, head


def _embedding_hook(rows):
    def add_rows(module, args, output):
        ids = args[0]
        # MCore transposes to [seq, batch, hidden] with reduce-scatter.
        if getattr(module, "reduce_scatter_embeddings", False):
            ids = ids.transpose(0, 1)
        return rows.embedding(ids, output)
    return add_rows


def _head_hook(rows):
    def add_rows(module, args, output):
        logits, bias = output
        return rows.logits(args[0], logits), bias
    return add_rows


def attach_action_rows(model, token_ids):
    if hasattr(model, "omnicommit_rows"):
        raise RuntimeError("Action rows already installed")
    embedding, head = _vocab_modules(model)
    native_embedding = embedding.modules_to_save["default"]
    native_head = head.modules_to_save["default"]
    if native_embedding.weight is native_head.weight:
        raise ValueError("Third action rows require separate input and output weights")
    rows = _VocabularyRows(native_embedding.weight, native_head.weight, list(token_ids.values()))
    # Keep the small parameters outside the PEFT wrappers. SWIFT loads its
    # standard MCore base and copies active/original modules by strict state dict.
    model.add_module("omnicommit_rows", rows)
    for wrapper in (embedding, head):
        wrapper.requires_grad_(False)
    native_embedding.register_forward_hook(_embedding_hook(rows.input_rows))
    native_head.register_forward_hook(_head_hook(rows.output_rows))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.getLogger("swift").info(
        "[action-rows] token_ids=%s vocabulary_trainable=%d local_model_trainable=%d",
        token_ids, sum(p.numel() for p in rows.parameters()), trainable,
    )


def _tables(model):
    embedding, head = _vocab_modules(model)
    return ((embedding.modules_to_save["default"].weight, model.omnicommit_rows.input_rows),
            (head.modules_to_save["default"].weight, model.omnicommit_rows.output_rows))


@torch.no_grad()
def initialize_action_rows(model):
    """SWIFT normally randomizes *all* zero vocab rows. Limit this to action IDs."""
    if getattr(model.omnicommit_rows, "loaded_hf_adapter", False):
        return
    for weight, rows in _tables(model):
        ids = rows.token_ids
        current = weight.index_select(0, ids)
        zero = (current == 0).all(dim=1)
        if zero.any():
            current[zero] = model.config.init_method(torch.empty_like(current[zero]))
        # Row initialization must agree across replicas even under different RNGs.
        if torch.distributed.is_initialized():
            torch.distributed.broadcast(current, src=0)
        weight.index_copy_(0, ids, current)


@contextmanager
def materialized_action_rows(models):
    backups = []
    try:
        with torch.no_grad():
            for model in models:
                for weight, rows in _tables(model):
                    original = weight.index_select(0, rows.token_ids)
                    backups.append((weight, rows.token_ids, original))
                    weight.index_copy_(0, rows.token_ids, rows.effective_rows(weight))
        yield
    finally:
        with torch.no_grad():
            for weight, ids, original in reversed(backups):
                weight.index_copy_(0, ids, original)


def wrap_bridge_save(bridge, token_ids):
    if getattr(bridge.save_weights, "_omnicommit_rows", False):
        return
    original = bridge.save_weights

    @wraps(original)
    def save(models, output_dir, *args, **kwargs):
        # Both supported bridges synchronously finalize safetensor shards.
        with materialized_action_rows(models):
            result = original(models, output_dir, *args, **kwargs)
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            write_action_rows_manifest(output_dir, token_ids)
        return result

    save._omnicommit_rows = True
    bridge.save_weights = save


def _resolve_ids(trainer):
    args = trainer.args
    for axis in ("tensor_model_parallel_size", "pipeline_model_parallel_size", "context_parallel_size"):
        if getattr(args, axis, 1) != 1:
            raise ValueError(f"Third action rows require {axis}=1")
    if not (args.no_save_optim and args.save_safetensors) or args.merge_lora:
        raise ValueError(
            "Action rows require HF adapter export with save_safetensors=true, "
            "no_save_optim=true and merge_lora=false"
        )
    if getattr(args, "tuner_type", getattr(args, "train_type", None)) != "lora":
        raise ValueError("Third action rows require LoRA training")
    if any(getattr(args, k, None) for k in ("adapter_load", "mcore_adapter", "ref_adapters", "ref_adapter_load")):
        raise ValueError("Action-row warm starts require an exported HF adapter")
    tokens = Path(os.environ["OMNICOMMIT_ACTION_TOKENS_FILE"]).read_text(encoding="utf-8-sig").splitlines()
    if list(args.new_special_tokens) != tokens:
        raise ValueError("SWIFT new_special_tokens differs from the Third training token inventory")
    tokenizer = trainer.template.tokenizer
    ids = {}
    for token in protocol_for_tokens(tokens).tokens:
        token_id = tokenizer.convert_tokens_to_ids(token)
        encoded = tokenizer.encode(token, add_special_tokens=False)
        if encoded != [token_id] or tokenizer.convert_ids_to_tokens(token_id) != token:
            raise ValueError(f"Training tokenizer does not encode {token} as one dedicated ID")
        ids[token] = token_id
    adapters = getattr(args, "adapters", None)
    if adapters:
        if len(adapters) != 1 or getattr(args, "load", None) or getattr(args, "mcore_model", None):
            raise ValueError("Warm start requires one action-row HF adapter and its original HF base")
        read_action_rows_manifest(adapters[0], ids, required=True)
    return ids


def _install_trainer_hooks(base):
    trainer_cls = base.BaseMegatronTrainer
    if getattr(trainer_cls, "_omnicommit_rows", False):
        return
    if hasattr(trainer_cls, "_prepare_peft_model"):
        # SWIFT 4: PEFT/native HF loads finish just before wrap_model builds DDP.
        original = trainer_cls._prepare_peft_model

        @wraps(original)
        def prepare(self, models):
            ids = _resolve_ids(self)
            peft_models = original(self, models)
            for model in models:
                attach_action_rows(model, ids)
                if getattr(self.args, "adapters", None):
                    model.omnicommit_rows.loaded_hf_adapter = True
            wrap_bridge_save(self.bridge, ids)
            return peft_models

        trainer_cls._prepare_peft_model = prepare
    else:
        # SWIFT 3: its model-provider closure injects PEFT before construction
        # of DDP and the optimizer, then loads the distributed base in place.
        original = trainer_cls.setup_model_and_optimizer

        @wraps(original)
        def setup(self, *args, **kwargs):
            ids = _resolve_ids(self)
            native_prepare = base.prepare_mcore_model

            def prepare(model):
                peft_model = native_prepare(model)
                attach_action_rows(model, ids)
                if getattr(self.args, "adapters", None):
                    model.omnicommit_rows.loaded_hf_adapter = True
                return peft_model

            wrap_bridge_save(self.bridge, ids)
            base.prepare_mcore_model = prepare
            try:
                return original(self, *args, **kwargs)
            finally:
                base.prepare_mcore_model = native_prepare

        trainer_cls.setup_model_and_optimizer = setup
    trainer_cls._initialize_embedding = staticmethod(initialize_action_rows)
    trainer_cls._omnicommit_rows = True


def install_action_rows():
    from swift.megatron.trainers import base
    _install_trainer_hooks(base)
