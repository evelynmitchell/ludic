from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Union, Optional
import logging
import time
import uuid
import json

log = logging.getLogger(__name__)

JSON = Union[str, int, float, bool, None, Dict[str, Any], List[Any]]


# Chat-style message schema
Message = Dict[str, str]  # {"role": "system|user|assistant", "content": "..."}

@dataclass
class ChatResponse:
    """
    Normalized inference output for training/logging.
    Keep this minimal. Put transport/vendor junk in the returned `info` dict.
    """
    text: str
    completion_token_ids: Optional[List[int]] = None
    completion_logprobs: Optional[List[float]] = None
    finish_reason: Optional[str] = None
    prompt_token_ids: Optional[List[int]] = None

    def __post_init__(self) -> None:
        if (
            self.completion_token_ids is not None
            and self.completion_logprobs is not None
            and len(self.completion_token_ids) != len(self.completion_logprobs)
        ):
            log.warning(
                "ChatResponse completion_token_ids/completion_logprobs length mismatch "
                "(%d vs %d).",
                len(self.completion_token_ids),
                len(self.completion_logprobs),
            )

    def to_info(self) -> Dict[str, Any]:
        """
        Canonical serialization of "training-relevant" fields into the shared `info`
        dict shape used throughout the project.
        """
        info: Dict[str, Any] = {
            "completion": self.text,
        }
        if self.prompt_token_ids is not None:
            info["prompt_token_ids"] = list(self.prompt_token_ids)
        if self.completion_token_ids is not None:
            info["completion_token_ids"] = list(self.completion_token_ids)
        if self.completion_logprobs is not None:
            info["completion_logprobs"] = list(self.completion_logprobs)
        if self.finish_reason is not None:
            info["finish_reason"] = self.finish_reason
        return info

    def merge_into_info(self, info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Merge this response's canonical fields into an existing `info` dict.
        """
        if info is None:
            info = {}
        info.update(self.to_info())
        return info

# ----- Environment level types -----

Observation = str
Info   = Dict[str, JSON]

@dataclass
class StepOutcome:
    obs: str
    reward: float
    truncated: bool
    terminated: bool
    info: Info = field(default_factory=dict)

@dataclass(frozen=True)
class Snapshot:
    """
    A portable checkpoint of an Environment, not just the outward State.
    `env_kind` and `version` let you refuse incompatible restores.
    """
    env_kind: str
    version: str           # bump on breaking schema changes
    episode_id: str        # source episode lineage
    obs: str           # outward state snapshot
    world: Dict[str, JSON] # extra hidden/internal stuff needed to resume
    created_ns: int = field(default_factory=lambda: time.time_ns())

    def to_json(self) -> str:
        return json.dumps(asdict(self))

# ----- Environment-Agent-Interaction level types -----

@dataclass
class Step:
    index: int
    prev_obs: Observation
    action: str
    next_obs: Optional[Observation]  # may be None on terminal steps
    reward: float
    truncated: bool
    terminated: bool
    info: Info = field(default_factory=dict)
    ts_ns: int = field(default_factory=lambda: time.time_ns())

@dataclass
class Rollout:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    steps: List[Step] = field(default_factory=list)
    meta: Dict[str, JSON] = field(default_factory=dict)

    @property
    def total_reward(self) -> float:
        return sum(s.reward for s in self.steps)

    @property
    def length(self) -> int:
        return len(self.steps)

    @property
    def start_ns(self) -> Optional[int]:
        return self.steps[0].ts_ns if self.steps else None

    @property
    def end_ns(self) -> Optional[int]:
        return self.steps[-1].ts_ns if self.steps else None

    @property
    def duration_ns(self) -> Optional[int]:
        start = self.start_ns
        end = self.end_ns
        if start is None or end is None:
            return None
        return end - start

    @property
    def duration_s(self) -> Optional[float]:
        ns = self.duration_ns
        return None if ns is None else ns / 1e9
