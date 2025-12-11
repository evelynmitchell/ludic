import logging
import asyncio

# Ludic Imports
from ludic.agents.base_agent import Agent
from ludic.context.full_dialog import FullDialog
from ludic.inference.vllm_client import VLLMChatClient
from ludic.parsers import xml_move_parser
from ludic.training.batching.rollout_engine import RolloutEngine
from ludic.training.batching import run_pipeline_actor
from ludic.training.types import EnvSpec, ProtocolSpec, RolloutRequest
from ludic.training.algorithm import make_reinforce
from ludic.interaction.single_agent import SingleAgentSyncProtocol

# Env Import
from examples.envs.tic_tac_toe import TicTacToeEnv

# ---------------------------------------------------------------------------
# Configuration (Must match Trainer!)
# ---------------------------------------------------------------------------
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
VLLM_HOST = "127.0.0.1"
VLLM_PORT = 8000
REDIS_URL = "redis://localhost:6379/0"
QUEUE_KEY = "ludic_tictactoe_queue"

logging.basicConfig(level=logging.INFO)

def create_engine(client: VLLMChatClient) -> RolloutEngine:
    """Setup the Environment and Agent logic."""
    
    # 1. Environment Registry
    env_registry = {"tictactoe": lambda **kwargs: TicTacToeEnv(**kwargs)}
    
    # 2. Protocol Factory
    def create_protocol(system_prompt: str = None):
        return SingleAgentSyncProtocol(
            agent=Agent(
                client=client, 
                model=MODEL_NAME, 
                ctx=FullDialog(), 
                parser=xml_move_parser
            ),
            prompt=system_prompt 
        )

    protocol_registry = {"single_agent": create_protocol}
    
    return RolloutEngine(
        env_registry=env_registry,
        protocol_registry=protocol_registry,
    )

def make_requests() -> list[RolloutRequest]:
    """Curriculum Definition."""
    base_prompt = TicTacToeEnv().suggested_sysprompt or ""
    training_prompt = base_prompt + "\n\nOutput your move as a single XML tag, e.g., <move>A1</move>."

    sampling_args = {
        "temperature": 1.0, 
        "max_tokens": 100,
        "extras": {"extra_body": {"return_token_ids": True}} 
    }
    
    return [RolloutRequest(
        env=EnvSpec(kind="tictactoe", kwargs={"agent_starts": True}),
        protocol=ProtocolSpec(kind="single_agent", kwargs={"system_prompt": training_prompt}),
        sampling_args=sampling_args, 
        num_episodes=1, 
    )]

async def main():
    print("🎬 Starting ACTOR Node")
    print(f"   vLLM:  {VLLM_HOST}:{VLLM_PORT}")
    print(f"   Redis: {REDIS_URL}")
    
    # Lightweight client (Inference only, no weight updates)
    client = VLLMChatClient(host=VLLM_HOST, port=VLLM_PORT)
    
    engine = create_engine(client)
    credit_assigner = make_reinforce(gamma=0.99).credit_assigner

    # Run Infinite Generation Loop
    await run_pipeline_actor(
        engine=engine,
        requests_fn=make_requests,
        credit_assigner=credit_assigner,
        redis_url=REDIS_URL,
        queue_key=QUEUE_KEY,
        max_steps=5,
        concurrency=16,      # Higher concurrency = higher throughput
        client=client,       # Needed to tag data with policy version
        retokenize=False
    )

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Actor stopped.")