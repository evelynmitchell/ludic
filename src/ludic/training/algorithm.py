from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, List

from torch import nn, Tensor
import torch

from ludic.training.types import CreditAssigner, SAWBatch
from ludic.training.loss import (
    Loss,
    ReinforceLoss,
    ReinforceBaselineLoss,
    selective_log_softmax,
)
from ludic.training.credit_assignment import MonteCarloReturn


Batch = Mapping[str, Tensor]


@dataclass
class RLAlgorithm:
    """
    Full RL algorithm = credit assignment + loss.

    - credit_assigner: maps Rollouts -> per-step scalar credits
                 (e.g. discounted returns / advantages)
    - loss:      consumes a collated batch (built from SAWBatch) and produces
                 a scalar loss and stats.
    - name:      identifier for logging / checkpoints
    """

    name: str
    credit_assigner: CreditAssigner
    loss: Loss

    def compute_loss(
        self,
        model: nn.Module,
        batch: Batch,
    ) -> tuple[Tensor, Dict[str, Any]]:
        """
        Runs the forward pass once and delegates to the Loss object.
        """
        # --- Run the forward pass ---
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        logits: Tensor = outputs.logits

        # Pass the resulting logits to the loss function
        return self.loss.compute(logits, batch)

    # ------------------------------------------------------------------
    # Optional preprocessing hook
    # ------------------------------------------------------------------
    def preprocess_batch(
        self,
        saw_batch: SAWBatch,
        *,
        model: Optional[nn.Module] = None,
        pad_token_id: Optional[int] = None,
    ) -> SAWBatch:
        """
        Algorithm-specific preprocessing on the SAWBatch before collation.

        Default: no-op. Override in algorithms (e.g., PPO) that need to
        materialize extra fields such as behavior logprobs.
        """
        return saw_batch


# ---------------------------------------------------------------------------
# PPO helper algorithm with behavior logprob backfill
# ---------------------------------------------------------------------------


class PPOAlgorithm(RLAlgorithm):
    """
    RLAlgorithm variant that can backfill behavior logprobs via teacher forcing
    when the rollout source did not provide them.

    - If all items already have old_token_logprobs, this is a no-op.
    - If some items are missing and a model is provided, it computes per-token
      logprobs of the sampled sequence under the current model (no grad) and
      stores them on the items.
    - Mixed presence without a model raises.
    - Recompute is only allowed for synchronous (on-policy) batches; async /
      pipeline batches must ship behavior logprobs with the data.

    Design note: this keeps behavior-logprob plumbing out of the generic
    Trainer/collator. PPO owns the backfill, and collation stays CPU-only.
    """

    def __init__(
        self,
        name: str,
        credit_assigner: CreditAssigner,
        loss: Loss,
        *,
        backfill_chunk_size: Optional[int] = None,
    ) -> None:
        super().__init__(name=name, credit_assigner=credit_assigner, loss=loss)
        self.backfill_chunk_size = backfill_chunk_size

    def preprocess_batch(
        self,
        saw_batch: SAWBatch,
        *,
        model: Optional[nn.Module] = None,
        pad_token_id: Optional[int] = None,
    ) -> SAWBatch:
        """
        Optional backfill of behavior logprobs.

        Backfill is only allowed for synchronous on-policy batches. Async /
        pipeline batches are expected to carry rollout-time logprobs already.
        """
        items = saw_batch.items
        # Extract any pre-computed behavior logprobs from meta
        for it in items:
            if "completion_logprobs" in it.meta and isinstance(it.meta["completion_logprobs"], list):
                # store a copy to avoid mutating shared structures
                it.meta["old_token_logprobs"] = list(it.meta["completion_logprobs"])

        missing_indices: List[int] = [
            i for i, it in enumerate(items) if not isinstance(it.meta.get("old_token_logprobs"), list)
        ]
        if not missing_indices:
            return saw_batch

        # Recompute is only allowed for synchronous (on-policy) batches.
        # Async / pipelined batches should carry logprobs tagged with policy_version.
        is_async = any("policy_version" in it.meta for it in items)
        if is_async:
            raise ValueError(
                "PPOAlgorithm missing old_token_logprobs on a batch tagged with policy_version; "
                "recomputation is only supported for synchronous on-policy batches."
            )

        if model is None or pad_token_id is None:
            raise ValueError(
                "PPOAlgorithm requires model and pad_token_id to backfill behavior logprobs."
            )

        device = next(model.parameters()).device
        subset = [items[i] for i in missing_indices]
        chunk_size = self.backfill_chunk_size or len(subset)

        restore_train_state = model.training
        if restore_train_state:
            model.eval()

        try:
            for start in range(0, len(subset), chunk_size):
                chunk_items = subset[start : start + chunk_size]
                lengths = [len(it.input_ids) for it in chunk_items]
                max_len = max(lengths)

                input_ids_list = []
                attn_mask_list = []
                action_mask_list = []
                for it in chunk_items:
                    L = len(it.input_ids)
                    ids = torch.full((max_len,), pad_token_id, dtype=torch.long)
                    am = torch.zeros((max_len,), dtype=torch.long)
                    actm = torch.zeros((max_len,), dtype=torch.float32)
                    ids[:L] = torch.tensor(it.input_ids, dtype=torch.long)
                    am[:L] = torch.tensor(it.attention_mask, dtype=torch.long)
                    actm[:L] = torch.tensor(it.action_mask, dtype=torch.float32)
                    input_ids_list.append(ids)
                    attn_mask_list.append(am)
                    action_mask_list.append(actm)

                batch_input_ids = torch.stack(input_ids_list, dim=0).to(device)
                batch_attn = torch.stack(attn_mask_list, dim=0).to(device)
                batch_action_mask = torch.stack(action_mask_list, dim=0).to(device)

                with torch.inference_mode():
                    logits = model(input_ids=batch_input_ids, attention_mask=batch_attn).logits
                    logits_shifted = logits[:, :-1, :]
                    target_ids = batch_input_ids[:, 1:]
                    token_logp = selective_log_softmax(logits_shifted, target_ids)  # [B, T-1]
                    action_mask_shifted = batch_action_mask[:, 1:]  # align with targets

                # Scatter back into items as per-token logprobs over the action region
                for idx, it in enumerate(chunk_items):
                    mask = action_mask_shifted[idx].bool()
                    per_token = token_logp[idx][mask].detach().cpu().tolist()
                    it.meta["old_token_logprobs"] = [float(v) for v in per_token]

                # Allow tensors to go out of scope; no explicit empty_cache to keep the path light.
        finally:
            if restore_train_state:
                model.train()

        return saw_batch


# ---------------------------------------------------------------------------
# Presets: REINFORCE and REINFORCE+baseline
# ---------------------------------------------------------------------------


def make_reinforce(
    *,
    gamma: float = 1.0,
    name: str = "reinforce",
) -> RLAlgorithm:
    """
    REINFORCE without baseline.

    - Credit assignment: Monte Carlo discounted return-to-go with discount `gamma`
          G_t = r_t + gamma * r_{t+1} + gamma^2 * r_{t+2} + ...
    - Loss:              ReinforceLoss using `batch["weight"]` as the return

    The orchestrator will use this algorithm's `credit_assigner` (MonteCarloReturn)
    to compute G_t per step, store it in SAWItem.weight, and collate that
    into `batch["weight"]` for the loss.
    """
    credit_assigner: CreditAssigner = MonteCarloReturn(gamma=gamma)
    loss: Loss = ReinforceLoss()

    return RLAlgorithm(
        name=name,
        credit_assigner=credit_assigner,
        loss=loss,
    )


def make_reinforce_baseline(
    *,
    gamma: float = 1.0,
    name: str = "reinforce_baseline",
    normalize_adv: bool = False,
) -> RLAlgorithm:
    """
    REINFORCE with batch-mean baseline:

        G_t = discounted return-to-go from step t
        b   = mean(G_t) over the batch
        A_t = G_t - b
        loss = - E[ A_t * log π(a_t|s_t) ]

    Here:
      - MonteCarloReturn(gamma) computes G_t and feeds it into SAWItem.weight
      - the collated batch exposes this as `batch["weight"]`

    If `normalize_adv=True`, A_t is additionally normalized to zero mean /
    unit variance within the batch before being used in the loss.
    """
    credit_assigner: CreditAssigner = MonteCarloReturn(gamma=gamma)
    loss: Loss = ReinforceBaselineLoss(
        normalize=normalize_adv,
    )

    return RLAlgorithm(
        name=name,
        credit_assigner=credit_assigner,
        loss=loss,
    )
