from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from ludic.types import Rollout
from ludic.training.types import RolloutStepKey


@dataclass
class MonteCarloReturn:
    """
    Monte Carlo return per step:

        G_t = r_t + gamma * r_{t+1} + gamma^2 * r_{t+2} + ...

    For each rollout and each step, assigns the discounted sum of *future*
    rewards including the current step.

    This is the standard REINFORCE-style return (optionally discounted).
    """

    gamma: float = 1.0

    def compute(
        self,
        rollouts: List[Rollout],
    ) -> Dict[RolloutStepKey, float]:
        out: Dict[RolloutStepKey, float] = {}

        for r in rollouts:
            # process steps in reverse to accumulate returns
            G = 0.0
            returns: List[float] = []

            for step in reversed(r.steps):
                G = float(step.reward) + self.gamma * G
                returns.append(G)

            returns.reverse()  # now aligned with r.steps order

            for step, g in zip(r.steps, returns):
                key: RolloutStepKey = (r.id, step.index)
                out[key] = g

        return out


@dataclass
class PerStepReward:
    """
    Assigns each step's weight equal to its immediate reward:

        w_t = r_t

    This is sometimes useful for simple bandit-style or myopic settings.
    """

    def compute(
        self,
        rollouts: List[Rollout],
    ) -> Dict[RolloutStepKey, float]:
        out: Dict[RolloutStepKey, float] = {}

        for r in rollouts:
            for step in r.steps:
                key: RolloutStepKey = (r.id, step.index)
                out[key] = float(step.reward)

        return out

@dataclass
class EpisodicReturn:
    """
    Assigns the same episodic return to every step in a rollout:

        R_ep = sum_t r_t   (undiscounted total reward for the episode)
        w_t  = R_ep        for all steps t in that rollout

    This is useful when you care only about the overall episode score and
    want each action in a successful episode to receive identical credit.
    """

    def compute(
        self,
        rollouts: List[Rollout],
    ) -> Dict[RolloutStepKey, float]:
        out: Dict[RolloutStepKey, float] = {}

        for r in rollouts:
            R_ep = float(r.total_reward)
            for step in r.steps:
                key: RolloutStepKey = (r.id, step.index)
                out[key] = R_ep

        return out
