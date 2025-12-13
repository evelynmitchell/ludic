from __future__ import annotations

from types import SimpleNamespace
from typing import Dict, Any, List

import pytest
import torch
from torch import nn

from ludic.training.loss import PPOLoss
from ludic.training.trainer import _collate_saw_items
from ludic.training.types import SAWItem, SAWBatch
from ludic.training.algorithm import PPOAlgorithm
from ludic.training.credit_assignment import MonteCarloReturn


class DummyModel(nn.Module):
    """
    Minimal model that returns constant logits for teacher-forced backfill.
    """

    def __init__(self, vocab_size: int = 3):
        super().__init__()
        self.vocab_size = vocab_size
        # Add a dummy parameter so `.parameters()` is non-empty
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, input_ids, attention_mask=None):
        # logits shape: [B, T, V]
        B, T = input_ids.shape
        logits = torch.zeros((B, T, self.vocab_size), dtype=torch.float32, device=input_ids.device)
        return SimpleNamespace(logits=logits)


def make_item(input_ids: List[int], action_mask: List[int], meta: Dict[str, Any] | None = None) -> SAWItem:
    L = len(input_ids)
    return SAWItem(
        input_ids=input_ids,
        attention_mask=[1] * L,
        action_mask=action_mask,
        weight=1.0,
        meta=meta or {},
    )


def test_ppoloss_requires_old_logp():
    loss_fn = PPOLoss()
    logits = torch.zeros((1, 3, 2))
    batch = {
        "input_ids": torch.tensor([[0, 1, 0]], dtype=torch.long),
        "action_mask": torch.tensor([[0, 1, 1]], dtype=torch.float32),
        "weight": torch.tensor([1.0], dtype=torch.float32),
    }
    with pytest.raises(KeyError):
        loss_fn.compute(logits, batch)


def test_collate_sums_behavior_logprobs():
    # Two tokens in the action; pre-computed behavior logprobs sum to -3.0
    logps = [-1.0, -2.0]
    items = [
        make_item([0, 1, 2], [0, 1, 1], meta={"old_token_logprobs": logps}),
        make_item([0, 2, 1], [0, 1, 1], meta={"old_token_logprobs": logps}),
    ]
    batch = _collate_saw_items(items, pad_token_id=0, device=torch.device("cpu"))
    assert "old_logp_action" in batch
    expected = torch.tensor([-3.0, -3.0], dtype=torch.float32)
    assert torch.allclose(batch["old_logp_action"], expected)


def test_collate_mixed_behavior_logprobs_raises():
    items = [
        make_item([0, 1, 2], [0, 1, 1], meta={"old_token_logprobs": [-1.0, -2.0]}),
        make_item([0, 2, 1], [0, 1, 1], meta={}),  # missing
    ]
    with pytest.raises(ValueError):
        _collate_saw_items(items, pad_token_id=0, device=torch.device("cpu"))


def test_collate_mismatched_behavior_logprobs_raises():
    # action_mask has two action tokens but only one logprob provided
    items = [
        make_item([0, 1, 2], [0, 1, 1], meta={"old_token_logprobs": [-1.0]}),
    ]
    with pytest.raises(ValueError):
        _collate_saw_items(items, pad_token_id=0, device=torch.device("cpu"))


def test_ppopreprocess_backfills_missing_logprobs():
    model = DummyModel(vocab_size=3)
    algo = PPOAlgorithm(
        name="ppo",
        credit_assigner=MonteCarloReturn(),
        loss=PPOLoss(),
        backfill_chunk_size=1,
    )
    items = [
        make_item([0, 1, 2], [0, 1, 1], meta={}),
        make_item([0, 2, 1], [0, 1, 1], meta={}),
    ]
    saw_batch = SAWBatch(items=items, meta={})
    processed = algo.preprocess_batch(
        saw_batch,
        model=model,
        pad_token_id=0,
    )
    for it in processed.items:
        token_logps = it.meta.get("old_token_logprobs")
        assert isinstance(token_logps, list)
        # With uniform logits over vocab_size=3, logprob = log(1/3) per action token
        assert len(token_logps) == 2
        for lp in token_logps:
            assert pytest.approx(lp, rel=1e-4) == -torch.log(torch.tensor(3.0)).item()


def test_ppopreprocess_async_batch_missing_logprobs_raises():
    model = DummyModel(vocab_size=3)
    algo = PPOAlgorithm(
        name="ppo",
        credit_assigner=MonteCarloReturn(),
        loss=PPOLoss(),
    )
    items = [
        make_item([0, 1, 2], [0, 1, 1], meta={"policy_version": 1}),
    ]
    saw_batch = SAWBatch(items=items, meta={"source": "pipeline_redis"})
    with pytest.raises(ValueError):
        algo.preprocess_batch(
            saw_batch,
            model=model,
            pad_token_id=0,
        )
