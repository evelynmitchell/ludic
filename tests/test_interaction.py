import pytest
from ludic.context.full_dialog import FullDialog
from ludic.interaction import run_episode
from tests._mocks import MockAgent, MockEnv

@pytest.mark.asyncio
async def test_happy_path_terminates_immediately():
    env = MockEnv(max_steps=3, target="1")
    agent = MockAgent()
    rollout = await run_episode(env=env, agent=agent, max_steps=5, sampling_args={}, ctx=FullDialog())
    assert rollout.steps[-1].terminated is True
    assert rollout.total_reward == pytest.approx(1.0)

@pytest.mark.asyncio
async def test_truncation_when_agent_is_wrong():
    class WrongAgent(MockAgent):
        async def act(self, *a, **k) -> str:
            return "nope"

    env = MockEnv(max_steps=2, target="1")
    agent = WrongAgent()
    rollout = await run_episode(env=env, agent=agent, max_steps=10, sampling_args={}, ctx=FullDialog())
    assert rollout.steps[-1].truncated is True
    assert rollout.total_reward < 0.0
