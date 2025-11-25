from __future__ import annotations
import asyncio
from typing import Any, Dict, List, Optional, Tuple, Mapping

import torch

from ludic.types import Message, SamplingArgs, ChatResponse
from ludic.inference.client import ChatClient
from ludic.inference.sampling import SamplingConfig, resolve_sampling_args


class Agent:
    """
    Concrete agent that defines an interface that can be relied on.
    Defers transport + low-level invocation to the underlying client, but
    is responsible for resolving partial SamplingArgs into a fully-specified
    SamplingConfig so backends do not have to guess or inject defaults.
    """
    name: str = "agent"

    def __init__(self, *, client: ChatClient, model: str) -> None:
        self._client = client
        self._model = model
        self.last_info: Dict[str, Any] = {}

    async def act(
        self,
        messages: List[Message],
        sampling_args: SamplingArgs,
        timeout_s: Optional[float] = None,
    ) -> Tuple[ChatResponse, Dict[str, Any]]:
        # Resolve partial SamplingArgs -> fully-specified SamplingConfig.
        sampling: SamplingConfig = resolve_sampling_args(sampling_args)

        coro = self._client.complete(
            model=self._model,
            messages=messages,
            sampling=sampling,
        )
        if timeout_s is None:
            resp, info = await coro
        else:
            resp, info = await asyncio.wait_for(coro, timeout=timeout_s)

        self.last_info = dict(info)
        return resp, info

    def push_policy_update(
        self,
        params: Mapping[str, torch.Tensor],
        *,
        timeout_s: float = 600.0,
        reset_cache: bool = True,
        version: Optional[str] = None,
    ) -> str:
        """
        Push updated policy parameters into the underlying runtime.

        Delegates to the ChatClient's push_update_atomic implementation.
        """
        if not hasattr(self._client, "push_update_atomic"):
            raise RuntimeError(
                "Underlying ChatClient does not support policy weight updates "
                "(missing push_update_atomic)."
            )

        return self._client.push_update_atomic(
            params,
            timeout_s=timeout_s,
            reset_cache=reset_cache,
            version=version,
        )
