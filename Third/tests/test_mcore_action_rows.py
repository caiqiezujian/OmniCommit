"""Check the training and HF checkpoint boundary with a tiny untied language model."""
from copy import deepcopy
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from third_asr.action_rows_manifest import FILENAME, read_action_rows_manifest
from third_asr.mcore_action_rows import attach_action_rows, initialize_action_rows, wrap_bridge_save
from third_asr.protocol import DIRECT, WAIT


TOKEN_IDS = {WAIT: 4, DIRECT: 6}


class SavedModule(nn.Module):
    """Expose the active PEFT copy used by the SWIFT bridge."""

    def __init__(self, module):
        super().__init__()
        self.original_module = module.requires_grad_(False)
        self.modules_to_save = nn.ModuleDict({"default": deepcopy(module).requires_grad_(True)})

    @property
    def weight(self):
        return self.modules_to_save["default"].weight

    def forward(self, *args, **kwargs):
        return self.modules_to_save["default"](*args, **kwargs)


class Head(nn.Linear):
    def forward(self, hidden, weight=None, runtime_gather_output=None):
        return F.linear(hidden, self.weight), None


class TinyOmni(nn.Module):
    def __init__(self, wrap=SavedModule):
        super().__init__()
        self.config = SimpleNamespace(init_method=lambda tensor: nn.init.constant_(tensor, .125))
        lm = self.language_model = nn.Module()
        lm.share_embeddings_and_output_weights = False
        lm.config = self.config
        lm.embedding = nn.Module()
        lm.embedding.word_embeddings = wrap(nn.Embedding(9, 3))
        lm.output_layer = wrap(Head(3, 9, bias=False))
        self.lora = nn.Linear(3, 3, bias=False)

    def forward(self, ids):
        hidden = self.language_model.embedding.word_embeddings(ids)
        return self.language_model.output_layer(self.lora(hidden), weight=None)[0]

    def vocab(self):
        return (self.language_model.embedding.word_embeddings.weight, self.language_model.output_layer.weight)


def test_initialization_skips_zero_rows_outside_the_action_tokens():
    model = TinyOmni()
    attach_action_rows(model, TOKEN_IDS)
    for weight in model.vocab():
        with torch.no_grad():
            weight[[2, 4]] = 0
    before = [weight.clone() for weight in model.vocab()]
    initialize_action_rows(model)
    for weight, original in zip(model.vocab(), before):
        assert torch.equal(weight[2], original[2])
        assert torch.equal(weight[6], original[6])
        assert torch.equal(weight[4], torch.full_like(weight[4], .125))


def check_optimizer_and_bridge(tmp_path, wrapper=SavedModule):
    torch.manual_seed(91)
    model = TinyOmni(wrapper)
    ids = torch.tensor([[1, 4, 6, 4]])
    reference = model(ids).detach().clone()
    before = [w.detach().clone() for w in model.vocab()]
    attach_action_rows(model, TOKEN_IDS)
    torch.testing.assert_close(model(ids), reference)
    trainable = [p for p in model.parameters() if p.requires_grad]
    assert sorted(p.numel() for p in trainable) == [6, 6, 9]
    optimizer = torch.optim.AdamW(trainable, lr=.03, weight_decay=.1)
    F.cross_entropy(model(ids).flatten(0, 1), torch.tensor([6, 1, 2, 4])).backward()
    optimizer.step()
    active_logits = model(ids).detach().clone()

    class Bridge:
        fail = True

        def save_weights(self, models, output_dir, **kwargs):
            self.exported = [weight.detach().clone() for weight in models[0].vocab()]
            if self.fail:
                raise OSError("writer failed")
            return "written"

    bridge = Bridge()
    wrap_bridge_save(bridge, TOKEN_IDS)
    with pytest.raises(OSError, match="writer failed"):
        bridge.save_weights([model], tmp_path, peft_format=True)
    assert not (tmp_path / FILENAME).exists()
    for weight, original in zip(model.vocab(), before):
        assert torch.equal(weight, original)
    bridge.fail = False
    bridge.save_weights([model], tmp_path, peft_format=True)
    assert read_action_rows_manifest(tmp_path, TOKEN_IDS, required=True)
    for trained, weight, original in zip(bridge.exported, model.vocab(), before):
        assert torch.equal(weight, original)
        assert (trained != original).any(dim=1).nonzero().flatten().tolist() == [4, 6]
    dense_hidden = model.lora(F.embedding(ids, bridge.exported[0]))
    torch.testing.assert_close(F.linear(dense_hidden, bridge.exported[1]), active_logits)
    torch.testing.assert_close(model(ids), active_logits)


def test_optimizer_and_bridge_roundtrip(tmp_path):
    check_optimizer_and_bridge(tmp_path)


def test_native_peft_adapter_reload_and_merge(tmp_path):
    peft = pytest.importorskip("peft")
    from transformers import PretrainedConfig

    torch.manual_seed(18)
    model = TinyOmni(wrap=lambda layer: layer)
    model.config = PretrainedConfig()
    base = deepcopy(model)
    vocab_keys = ("language_model.embedding.word_embeddings.weight", "language_model.output_layer.weight")
    original = [base.state_dict()[k].clone() for k in vocab_keys]
    cfg = peft.LoraConfig(r=2, lora_alpha=4, target_modules=["lora"],
                          modules_to_save=["word_embeddings", "output_layer"])
    adapter = peft.get_peft_model(model, cfg)
    attach_action_rows(model, TOKEN_IDS)
    ids = torch.tensor([[6, 3, 4, 1]])
    trainable = (p for p in model.parameters() if p.requires_grad)
    optimizer = torch.optim.AdamW(trainable, lr=.03, weight_decay=.1)
    F.cross_entropy(adapter(ids).flatten(0, 1), torch.tensor([4, 2, 6, 5])).backward()
    optimizer.step()
    expected = adapter(ids).detach()
    bridge = SimpleNamespace(save_weights=lambda models, path: adapter.save_pretrained(
        path, safe_serialization=True, save_embedding_layers=False))
    wrap_bridge_save(bridge, TOKEN_IDS)
    bridge.save_weights([model], tmp_path)

    merged = peft.PeftModel.from_pretrained(base, tmp_path).merge_and_unload()
    torch.testing.assert_close(merged(ids), expected)
    for key, initial in zip(vocab_keys, original):
        final = merged.state_dict()[key]
        assert (final != initial).any(dim=1).nonzero().flatten().tolist() == [4, 6]
