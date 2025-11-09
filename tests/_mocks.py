from __future__ import annotations
from typing import Any, Optional, List, Tuple
from ludic.types import Message, SamplingArgs, StepOutcome, Observation, Info
from ludic.agent.base import Agent
from ludic.env import Env

class MockAgent(Agent):
    """
    A trivial agent for testing. Always replies with "1".
    """

    async def act(self, messages, sampling_args, **kwargs) -> str:
        return "1"


class MockEnv(Env):
    """
    Minimal env for exercising the interaction loop only.

    - reset() -> returns a fixed instruction.
    - step("1") -> reward=1.0, terminated=True, obs="✅ done".
    - step(other) -> reward=-0.1, keep going with obs="❌ try again {t}/{max}".
    - Truncates after `max_steps`.
    - No snapshot support on purpose.
    """

    def __init__(self, *, max_steps: int = 8, target: str = "1") -> None:
        self.max_steps = max_steps
        self.target = target
        self._t = 0
        self._obs: Observation = "Reply with '1' to finish."

    @property
    def suggested_sysprompt(self) -> Optional[str]:
        return "You are terse. Output only the final answer."

    def reset(self, *, seed: Optional[int] = None) -> Tuple[Observation, Info]:
        self._t = 0
        self._obs = "Reply with '1' to finish."
        return self._obs, {}

    def step(self, action: str) -> StepOutcome:
        if self._t >= self.max_steps:
            # already truncated on the previous call
            return StepOutcome(obs=self._obs, reward=0.0, truncated=True, terminated=False, info={})

        self._t += 1
        if action.strip() == self.target:
            self._obs = "✅ done"
            return StepOutcome(obs=self._obs, reward=1.0, truncated=False, terminated=True, info={})

        # wrong answer
        self._obs = f"❌ try again {self._t}/{self.max_steps}"
        truncated = self._t >= self.max_steps
        return StepOutcome(obs=self._obs, reward=-0.1, truncated=truncated, terminated=False,
                           info={"attempt": self._t})

    def current_obs(self) -> Observation:
        return self._obs
