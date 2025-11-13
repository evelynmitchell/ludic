import pytest
from ludic.training.orchestrator import Orchestrator, OrchestratorConfig
from tests._mocks import MockEnv, MockAgent

@pytest.mark.asyncio
async def test_generate_async_n_rollouts():
    o = Orchestrator(lambda: MockEnv(max_steps=2), MockAgent(),
                     cfg=OrchestratorConfig(episodes=4, max_steps=3, concurrency=2))
    rs = await o.generate()
    assert len(rs) == 4
    assert all(r.length >= 1 for r in rs)

def test_generate_sync_n_rollouts():
    o = Orchestrator(lambda: MockEnv(max_steps=2), MockAgent(),
                     cfg=OrchestratorConfig(episodes=3, max_steps=3, concurrency=1))
    rs = o.generate_sync()
    assert len(rs) == 3
