from __future__ import annotations
import asyncio
from typing import Any, Dict, List, Optional, Tuple, Mapping

import torch

from ludic.types import SamplingArgs, Observation, Info, Message, ChatResponse
from ludic.inference.client import ChatClient
from ludic.inference.sampling import SamplingConfig, resolve_sampling_args
from ludic.context.base import ContextStrategy
from ludic.parsers import Parser, ParseResult

_DEFAULT_INCOMPLETE_FEEDBACK = (
    "Your response was cut off because it exceeded the token limit. "
    "Please provide a shorter, more concise response."
)


class Agent:
    """
    A stateful, logical actor that bundles inference, context, and parsing.

    It holds a reference to a (potentially shared) ChatClient and manages
    its own internal state via its ContextStrategy.
    """

    name: str = "agent"

    def __init__(
        self,
        *,
        client: ChatClient,
        model: str,
        ctx: ContextStrategy,
        parser: Parser,
        reject_incomplete_completions: bool = True,
        incomplete_completion_penalty: float = -0.1,
        incomplete_completion_feedback: str = _DEFAULT_INCOMPLETE_FEEDBACK,
    ) -> None:
        """
        Initializes the Agent.

        Args:
            client: The ChatClient for inference.
            model: The model name this agent should use.
            ctx: An instance of a ContextStrategy for managing memory.
            parser: An instance of a Parser for decoding actions.
            reject_incomplete_completions: If True, completions that hit max_tokens
                (finish_reason="length") are treated as parse failures with feedback.
            incomplete_completion_penalty: Reward penalty for incomplete completions.
            incomplete_completion_feedback: Feedback message shown to agent when
                its completion is cut off.
        """
        self._client = client
        self._model = model
        self._ctx = ctx
        self._parser = parser
        self._reject_incomplete = reject_incomplete_completions
        self._incomplete_penalty = incomplete_completion_penalty
        self._incomplete_feedback = incomplete_completion_feedback
        self.last_info: Dict[str, Any] = {}

    async def _infer_once(
        self,
        *,
        messages: List[Message],
        sampling_args: SamplingArgs,
        timeout_s: Optional[float] = None,
    ) -> Tuple[ChatResponse, Dict[str, Any], Dict[str, Any]]:
        """
        Shared single inference helper.

        Resolves sampling args, runs the client call (optionally with timeout),
        merges token IDs/logprobs into the returned info dict, and updates self.last_info.
        """
        sampling: SamplingConfig = resolve_sampling_args(sampling_args)
        coro = self._client.complete(
            model=self._model,
            messages=messages,
            sampling=sampling,
        )
        if timeout_s is None:
            resp, client_info = await coro
        else:
            resp, client_info = await asyncio.wait_for(coro, timeout=timeout_s)

        last_info: Dict[str, Any] = dict(client_info)
        # Store prompt and completion for logging/training
        last_info["chat_prompt_messages"] = messages
        last_info["chat_completion"] = {"role": "assistant", "content": resp.text}
        resp.merge_into_info(last_info)

        self.last_info = last_info
        return resp, client_info, last_info

    def reset(self, system_prompt: Optional[str] = None) -> None:
        """Resets the agent's internal context."""
        self._ctx.reset(system_prompt=system_prompt)
        
    def on_env_reset(self, obs: Observation, info: Info):
        """Called by the protocol *after* env.reset()."""
        self._ctx.on_env_reset(obs, info)
        
    def on_after_step(self, obs: Observation, info: Info):
        """Called by the protocol *after* env.step()."""
        self._ctx.on_after_step(obs, info)

    async def act(
        self,
        sampling_args: SamplingArgs,
        timeout_s: Optional[float] = None,
    ) -> Tuple[ParseResult, str, Dict[str, Any]]:
        """
        Runs the think -> act -> parse cycle based on current context.

        This method does *not* take obs/info, as those are fed to the
        agent via on_env_reset() and on_after_step().

        Args:
            sampling_args: The sampling configuration for this step.
            timeout_s: Optional timeout for the inference call.

        Returns:
            A tuple of (ParseResult, raw_action_text, client_info_dict).
        """
        # 1. Think (prepare prompt messages from context)
        messages: List[Message] = self._ctx.on_before_act()

        # 2. Act (run inference)
        resp, _client_info, last_info = await self._infer_once(
            messages=messages,
            sampling_args=sampling_args,
            timeout_s=timeout_s,
        )

        # 3. Update memory with the agent's own response
        self._ctx.on_after_act(resp)

        raw_action = resp.text

        # 4. Check for incomplete completion (hit max_tokens)
        if self._reject_incomplete and last_info.get("finish_reason") == "length":
            parse_result = ParseResult(
                action=None,
                reward=self._incomplete_penalty,
                obs=self._incomplete_feedback,
            )
            # Mark this in info for downstream tracking
            last_info["incomplete_completion"] = True
            return parse_result, raw_action, last_info

        # 5. Parse (format the raw text action)
        parse_result = self._parser(raw_action)

        return parse_result, raw_action, last_info

    def push_policy_update(
        self,
        params: Mapping[str, torch.Tensor],
        *,
        timeout_s: float = 600.0,
        version: Optional[str] = None,
    ) -> str:
        """Pushes updated policy parameters to the underlying runtime."""
        if not hasattr(self._client, "sync_weights"):
            raise RuntimeError(
                "Underlying ChatClient does not support policy weight updates "
                "(missing sync_weights)."
            )
        return self._client.sync_weights(
            params,
            timeout_s=timeout_s,
            version=version,
        )
