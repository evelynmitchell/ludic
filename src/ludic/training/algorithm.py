from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Protocol

from jaxtyping import Float
from torch import nn, Tensor

from ludic.training.types import CreditAssigner, SAWBatch
from ludic.training.loss import (
    Loss,
    ReinforceLoss,
    ReinforceBaselineLoss,
    ClippedSurrogateLoss,
    MaskedCausalLMCrossEntropyLoss,
)
from ludic.training.credit_assignment import MonteCarloReturn, GroupNormalizedReturn, ConstantCredit


Batch = Mapping[str, Tensor]
Logits = Float[Tensor, "B T V"]

class PreprocessFn(Protocol):
    def __call__(self, saw_batch: SAWBatch) -> SAWBatch: ...


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
    preprocess: Optional[PreprocessFn] = None

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
        logits: Logits = outputs.logits

        # Pass the resulting logits to the loss function
        return self.loss.compute(logits, batch)


# ---------------------------------------------------------------------------
# Behavior logprob requirements (ratio-based objectives)
# ---------------------------------------------------------------------------


def validate_actor_logps(
    saw_batch: SAWBatch,
) -> SAWBatch:
    """
    Validate SAWItems carry per-token logprobs under the behavior policy (actor).

    Contract:
      - For ratio objectives (PPO/GRPO/KL-to-behavior), each SAWItem must carry
        `item.actor_logps` (backed by `item.attachments.actor_logps`), computed at rollout time by the inference
        client and propagated through batching/collation.

    This function validates the contract; it does not backfill or recompute logprobs.
    """
    items = saw_batch.items

    missing = []
    for i, it in enumerate(items):
        actor = it.actor_logps
        if actor is None:
            missing.append(i)
            continue
        expected_len = int(sum(int(x) for x in it.action_mask))
        if len(actor.token_logps) != expected_len:
            raise ValueError(
                "ActorTokenLogps length mismatch for item "
                f"(index={i}, rollout_id={it.meta.get('rollout_id')!r}, step_index={it.meta.get('step_index')!r}): "
                f"expected {expected_len}, got {len(actor.token_logps)}."
            )
        if not isinstance(actor.token_logps, list) or not all(
            isinstance(v, (int, float)) for v in actor.token_logps
        ):
            raise TypeError("ActorTokenLogps.token_logps must be a List[float].")

    if missing:
        raise ValueError(
            "Missing SampleAttachments.actor_logps for a ratio-based objective. "
            "Ensure your inference client returns chosen-token logprobs and your batch collation "
            "populates SAWItem.attachments.actor_logps (e.g., via ReturnSpec.for_rl()). "
            f"Missing indices: {missing}."
        )

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


def make_grpo(
    *,
    group_size: int,
    group_normalize_adv: bool = True,
    positive_only: bool = False,
    clip_eps: float = 0.2,
    length_normalize: bool = False,
    name: str = "grpo",
) -> RLAlgorithm:
    """
    GRPO-style preset (clipped surrogate):

      - Credit assignment: group-normalized returns (per-group baseline)
      - Loss: PPO-style clipped surrogate (policy term only)

    Rollouts must carry `group_id` in their metadata and each group must
    have exactly `group_size` members. Raises ValueError otherwise.

    Args:
        group_size: Number of rollouts per group.
        group_normalize_adv: Whether to normalize advantages within each group.
        positive_only: If True, clip negative advantages to zero (reinforce-only).
        clip_eps: PPO clipping epsilon for the surrogate objective.
        length_normalize: Whether to normalize log-probs by action length.
        name: Algorithm name for logging/metrics.
    Note: For the clipped ratio objective, we need behavior-policy logprobs.
    This preset installs a preprocessor that validates
    `item.actor_logps` is present.
    """
    credit_assigner: CreditAssigner = GroupNormalizedReturn(
        group_size=group_size,
        normalize_adv=group_normalize_adv,
        positive_only=positive_only,
    )
    loss: Loss = ClippedSurrogateLoss(clip_eps=clip_eps, length_normalize=length_normalize)
    preprocess = validate_actor_logps

    return RLAlgorithm(
        name=name,
        credit_assigner=credit_assigner,
        loss=loss,
        preprocess=preprocess,
    )


# ---------------------------------------------------------------------------
# SFT (Supervised Fine-Tuning / Behavioral Cloning)
# ---------------------------------------------------------------------------


def make_sft(
    *,
    length_normalize: bool = False,
    name: str = "sft",
) -> RLAlgorithm:
    """
    Supervised Fine-Tuning (SFT) / Behavioral Cloning.

    This is offline RL with trivial credit assignment:
      - Credit assignment: constant weight=1.0 for all steps
      - Loss: ReinforceLoss (which with uniform weights is just NLL)

    SFT treats all actions in the dataset equally, making it suitable for:
      - Cold-start training on rejection-sampled successful trajectories
      - Behavioral cloning from expert demonstrations
      - Pre-training before RL fine-tuning

    Args:
        length_normalize: If True, normalize log-probs by action length.
            This prevents the loss from being dominated by long sequences.
        name: Algorithm name for logging/metrics.

    Usage with OfflineBatchSource:
        ```python
        from ludic.training import OfflineBatchSource, Trainer, make_sft

        algo = make_sft()
        batch_source = OfflineBatchSource(
            jsonl_paths=[Path("data/winners.jsonl")],
            tokenize=tokenizer.encode,
            credit_assigner=algo.credit_assigner,
            batch_size=32,
        )
        trainer = Trainer(model=model, algorithm=algo, ...)
        ```
    """
    credit_assigner: CreditAssigner = ConstantCredit(value=1.0)
    # Use standard token-level CE over the action region for stability.
    loss: Loss = MaskedCausalLMCrossEntropyLoss(length_normalize=length_normalize)

    return RLAlgorithm(
        name=name,
        credit_assigner=credit_assigner,
        loss=loss,
    )
