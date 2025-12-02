from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional, Tuple
from ludic.types import Observation, StepOutcome, Info, Snapshot

class Env(ABC):
    @property
    def suggested_sysprompt(self) -> Optional[str]:
        return None

    @abstractmethod
    def reset(self, *, seed: Optional[int] = None) -> Tuple[Observation, Info]:
        ...

    @abstractmethod
    def step(self, action: str) -> StepOutcome:
        ...

    @abstractmethod
    def current_obs(self) -> Observation:
        ...

    # Snapshots -----------------------------------------------------------
    def snapshot(self) -> Snapshot:
        """
        Return a resumable snapshot of the environment. Default implementation
        raises; concrete envs MUST override to be snapshot-capable.
        """
        raise NotImplementedError("snapshot() not supported")

    def reset_from(self, snap: Snapshot) -> Tuple[Observation, Info]:
        """
        Restore the environment from a snapshot and start a NEW episode fork.
        Returns (obs, info). Concrete envs SHOULD override; default refuses.
        """
        raise NotImplementedError("reset_from() not supported")
