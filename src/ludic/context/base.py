from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Optional
from ludic.types import Message, Observation, Info, ChatResponse

class ContextStrategy(ABC):
    """
    Owns the chat transcript & how it's exposed to the agent at each step.
    """

    def __init__(self, system_prompt: Optional[str] = None) -> None:
        self._messages: List[Message] = []
        if system_prompt:
            self._messages.append({"role": "system", "content": system_prompt})

    # ---- lifecycle ------------------------------------------------------
    def reset(self, *, system_prompt: Optional[str] = None) -> None:
        self._messages = []
        if system_prompt:
            self._messages.append({"role": "system", "content": system_prompt})

    # ---- event hooks ----------------------------------------------------
    @abstractmethod
    def on_env_reset(self, obs: Observation, info: Info) -> None:
        """Called after env.reset(). Typically appends a user message."""

    @abstractmethod
    def on_before_act(self) -> List[Message]:
        """Return messages to send to the agent for the *next* act() call."""

    @abstractmethod
    def on_after_act(self, assistant_response: ChatResponse) -> None:
        """Record assistant output (and optionally transform it)."""

    @abstractmethod
    def on_after_step(self, next_obs: Observation, info: Info) -> None:
        """Record the new observation (usually as a user message)."""

    # ---- convenience ----------------------------------------------------
    @property
    def messages(self) -> List[Message]:
        return list(self._messages)
