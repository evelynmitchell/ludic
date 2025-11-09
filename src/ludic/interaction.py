from __future__ import annotations
from typing import Optional
from ludic.env import Env
from ludic.agent.base import Agent
from ludic.types import Rollout, Step, StepOutcome, SamplingArgs
from ludic.context.base import ContextStrategy
from ludic.context.full_dialog import FullDialog

async def run_episode(
    env: Env,
    agent: Agent,
    *,
    max_steps: int,
    sampling_args: Optional[SamplingArgs] = None,
    ctx: Optional[ContextStrategy] = None,
    system_prompt: Optional[str] = None,
    timeout_s: Optional[float] = None,
) -> Rollout:
    # context instance per episode
    if ctx is None:
        ctx = FullDialog()

    # choose system prompt priority: explicit > env-suggested > none
    sys = system_prompt or env.suggested_sysprompt
    ctx.reset(system_prompt=sys)

    rollout = Rollout(meta={
        "agent": getattr(agent, "name", "unknown"),
        "env": env.__class__.__name__,
        "ctx": ctx.__class__.__name__,
    })

    obs, info = env.reset()
    ctx.on_env_reset(obs, info)

    sargs: SamplingArgs = sampling_args or {}

    for t in range(max_steps):
        messages = ctx.on_before_act()
        text = await agent.call(messages=messages, sampling_args=sargs, timeout_s=timeout_s)
        ctx.on_after_act(text)
        outcome: StepOutcome = env.step(text)

        rollout.steps.append(Step(
            index=t,
            prev_obs=obs,
            action=text,
            next_obs=outcome.obs,
            reward=outcome.reward,
            truncated=outcome.truncated,
            terminated=outcome.terminated,
            info=outcome.info,
        ))

        obs = outcome.obs
        ctx.on_after_step(obs, outcome.info)

        if outcome.terminated or outcome.truncated:
            break

    return rollout
