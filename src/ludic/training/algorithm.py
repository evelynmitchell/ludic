from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping

from torch import nn, Tensor

from ludic.training.types import WeightingStrategy
from ludic.training.loss import (
    Loss,
    ReinforceLoss,
    ReinforceBaselineLoss,
)
from ludic.training.credit_assignment import MonteCarloReturn


Batch = Mapping[str, Tensor]


@dataclass
class RLAlgorithm:
    """
    Full RL algorithm = credit assignment + loss.

    - weighting: maps Rollouts -> per-step scalar credits
                 (e.g. discounted returns / advantages)
    - loss:      consumes a collated batch (built from SAWBatch) and produces
                 a scalar loss and stats.
    - name:      identifier for logging / checkpoints
    """

    name: str
    weighting: WeightingStrategy
    loss: Loss

    def compute_loss(
        self,
        model: nn.Module,
        batch: Batch,
    ) -> tuple[Tensor, Dict[str, Any]]:
        """
        Convenience wrapper so trainers can just call algo.compute_loss(...)
        instead of algo.loss.compute(...).
        """
        return self.loss.compute(model, batch)


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

    The orchestrator will use this algorithm's `weighting` (MonteCarloReturn)
    to compute G_t per step, store it in SAWItem.weight, and collate that
    into `batch["weight"]` for the loss.
    """
    weighting: WeightingStrategy = MonteCarloReturn(gamma=gamma)
    loss: Loss = ReinforceLoss()

    return RLAlgorithm(
        name=name,
        weighting=weighting,
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
    weighting: WeightingStrategy = MonteCarloReturn(gamma=gamma)
    loss: Loss = ReinforceBaselineLoss(
        normalize=normalize_adv,
    )

    return RLAlgorithm(
        name=name,
        weighting=weighting,
        loss=loss,
    )
