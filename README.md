# Ludic
## A hackable training library for the era of experience

This repo is the result of a full rewrite: a library (not a framework) where you can rip out parts, swap in your own, and keep the pieces loosely coupled. Training and inference stay decoupled, training runs async, and everything else is designed to be composable rather than prescriptive.

### Why this design
- Agent vs. Environment: envs are pure state-transition functions with optional rewards and prompts; agents own the LLM harness (prompting, context, parsing, auxiliary tools, intrinsic format rewards).
- Episode vs. Rollout: an episode is an env run; a rollout is an agent’s perspective on that episode. Multi-agent setups get clean, per-agent rollouts via trajectory collectors.
- InteractionProtocols: the explicit agent–env loop lives outside both, so you can rewrite how interaction works without touching agent/env code.
- Rollouts → SAW tensors: `RolloutEngine` produces trajectories; `BatchSource` turns them into State–Action–Weight batches for the GPU.
- Algorithms as plug-ins: the Trainer is just an optimization loop; swap credit assigners (MC return, GRPO group normalization, per-step/episodic) and losses without new trainer classes.
- Async pipeline ready: run actors on inference nodes pushing SAWItems into Redis while the trainer on your GPU never waits for generation.

### What’s here
- Agents: base `Agent`, `ToolAgent`, and `ReActAgent`; parser utilities (`xml_tag_parser`, `think_prefix_parser`, `compose_parsers`) with intrinsic rewards/penalties for format.
- Context: `FullDialog`, `TruncatedThinkingContext` (keeps full history while trimming `<think>...</think>` in prompts).
- Environments: `LudicEnv` for multi-agent, `SingleAgentEnv` for gym-like single-agent; examples for QA and Tic-Tac-Toe.
- Interaction: `SingleAgentSyncProtocol`, multi-agent loop + collectors for per-agent rollouts.
- Rollouts & batching: `RolloutEngine`, GRPO/identity request strategies, token-aware collation into SAW batches.
- Training: `Trainer` (FSDP-aware, grad accumulation, optional grad checkpointing), REINFORCE/baseline losses, checkpointing, reducers/loggers.
- Inference/distribution: `VLLMChatClient` + weight publishing to the bundled `ludic.inference.vllm_server`, Redis pipeline RL (actor/trainer split), policy version tagging to reject stale data.
- Examples: Tic-Tac-Toe GRPO + LoRA (`examples/tic_tac_toe/train_tic_tac_toe.py`), GSM8K GRPO (`examples/gsm8k/*`), FSDP2 math (`examples/fsdp2_training/*`), pipeline RL actor/trainer, rejection sampling data generation.
- Tests: unit/integration suites (`integration`, `gpu` markers).

### Status
This is aspirational and built for researchers who want clean abstractions and hackability. It is not production-grade; expect to modify it for your own runs.

## Repository layout
- `src/ludic/agents`: base `Agent`, `ToolAgent`, and `ReActAgent` (tool-calling loop).
- `src/ludic/context`: conversation memory strategies (`FullDialog`, truncated thinking).
- `src/ludic/envs`: environment interfaces plus `DatasetQAEnv` and `TicTacToeEnv`.
- `src/ludic/interaction`: protocols that wire agents to envs (`SingleAgentSyncProtocol`).
- `src/ludic/inference`: OpenAI-compatible clients (`VLLMChatClient`), sampling utilities.
- `src/ludic/training`: RL algorithms, losses, batching (`RolloutEngine`, `RolloutBatchSource`, pipeline actor/redis queue), trainer, checkpointing, logging.
- `environments/`: runnable example envs (Tic-Tac-Toe, GSM8K-style QA).
- `examples/`: entry points for Tic-Tac-Toe GRPO + LoRA, GSM8K training/eval, FSDP2 math training, pipeline RL actor/trainer, and rejection sampling.
- `tests/`: unit/integration coverage (markers: `integration`, `gpu`).

### Examples at a glance (what each one is for)
- Tic-Tac-Toe (`examples/tic_tac_toe/train_tic_tac_toe.py`): LoRA fine-tuning + GRPO-style grouped advantages on a small env. Intended as the quickest end-to-end RL run; works well with 2 GPUs (1 vLLM inference, 1 training).
- GSM8K (`examples/gsm8k/train_gsm8k.py` + `eval_gsm8k_vllm.py`): GRPO with PPO-style clip to showcase standard question-answer training. Also laid out for 2 GPUs (1 inference, 1 training).
- FSDP2 Math (`examples/fsdp2_training/train_math_fsdp2.py`): Multi-GPU (4 GPU template: 1 vLLM, 3 training) showing FSDP2 wrapping, NCCL weight pushes, and GRPO-style credit assignment on a 7B model.
- Pipeline RL (`examples/pipeline_rl/run_actor.py`, `run_trainer.py`): actor/learner split over Redis for async sampling.
- Rejection sampling: `examples/rejection_sampling.py` for data generation via filtering.

## Requirements
- Python 3.11+
- PyTorch >= 2.8.0 with CUDA for training examples
- vLLM server exposing the OpenAI API; NCCL available if you want live weight pushes
- Redis for the pipeline RL actor/trainer example

## Installation
Using [uv](https://github.com/astral-sh/uv) (recommended because `uv.lock` is checked in):
```bash
uv sync --group dev   # installs runtime + dev/test deps
```

## Running tests
```bash
uv run pytest              # unit suite
uv run pytest -m gpu       # GPU-marked tests (if available)
```

## Architectural flow (story edition)
1) Define envs as pure state machines; keep agent harnesses separate so you can reuse prompts, parsers, and tools across envs.  
2) Wire them with an `InteractionProtocol` that owns the loop and emits rollouts.  
3) Turn rollouts into SAWItems via a `BatchSource`: synchronous (`RolloutBatchSource`) for debugging, or pipeline (`PipelineBatchSource`) where actors push SAWItems to Redis and the trainer just pops and optimizes.  
4) Let the Trainer stay dumb: it only sees tensors and a `weight` field; swap credit assigners (MC return, GRPO’s `GroupNormalizedReturn`, per-step, episodic) and losses without new trainer classes.  
5) For GRPO, use `GRPORequestStrategy` to expand requests into groups sharing env seeds but varying sampling seeds; normalize returns by group and train without touching the Trainer.  
6) Curriculum and heterogeneity are just different lists of `RolloutRequest` objects—mix Tic-Tac-Toe and GSM8K in one pass, or shift distributions over time.

## Quickstart: synchronous training loop
1. Start the bundled vLLM server with weight-update endpoints (no `--enable-lora`; Ludic will push merged weights itself), e.g.:
   ```bash
   uv run python -m ludic.inference.vllm_server \
     --model Qwen/Qwen2.5-7B-Instruct --port 8000 --host 0.0.0.0 --dtype bfloat16
   ```
2. Run the Tic-Tac-Toe GRPO + LoRA trainer (requires a GPU):
   ```bash
   PYTHONPATH=. uv run python examples/tic_tac_toe/train_tic_tac_toe.py
   ```
   This wires `SingleAgentSyncProtocol` + `TicTacToeEnv` through `RolloutEngine`, expands rollouts with `GRPORequestStrategy`, and pushes updated weights to vLLM via NCCL. Defaults target `Qwen/Qwen2.5-7B-Instruct`; override `--model/--port/--host` as needed.

## Pipeline RL (actor/trainer split)
1. Start Redis locally (`redis-server`).
2. Launch vLLM as above.
3. In one shell start the actor (generates and pushes SAW samples to Redis):
   ```bash
   uv run python examples/pipeline_rl/run_actor.py
   ```
4. In another shell start the trainer (pulls batches from Redis, trains, and publishes weights):
   ```bash
   uv run python examples/pipeline_rl/run_trainer.py
   ```

## Data generation example
Use rejection sampling to keep only winning Tic-Tac-Toe trajectories:
```bash
uv run python examples/rejection_sampling.py
```

## Extending Ludic
- Build an environment: subclass `SingleAgentEnv`, implement `env_reset`, `env_step`, and optionally `suggested_sysprompt`.
- Configure an agent: pick a context (`FullDialog`, `TruncatedThinkingContext`), a parser (`xml_tag_parser`, `think_prefix_parser`, compose as needed), and a `ChatClient` (vLLM or your own implementation of `ChatClient`).
- Train: register env/protocol factories with `RolloutEngine`, feed `RolloutBatchSource` (synchronous) or `PipelineBatchSource` (Redis) into `Trainer`, and choose an algorithm (`make_reinforce`, `make_reinforce_baseline`).

### Behavior notes
- Parser failures: if a parser returns `ParseResult.action=None`, protocols will not call `env.step()` for that agent. They log a synthetic step (`info["parse_error"]=True`) with reward from the parser and feed the synthetic observation back into the agent context.
- Tool failures: tool-calling agents record missing tools, invalid JSON args, and tool exceptions as tool messages in the context and continue the loop.
- Algorithms shown in examples today: GSM8K uses GRPO (grouped advantages + PPO-style clip). Tic-Tac-Toe and the FSDP2 Math script use grouped-advantage REINFORCE (GRPO-style `GroupNormalizedReturn`).
