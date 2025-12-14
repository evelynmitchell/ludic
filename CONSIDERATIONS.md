# Considerations (Notes for Later)

## DCP (Distributed Checkpointing) and Resuming With Different GPU Counts

We use PyTorch **DCP** (“Distributed Checkpointing”) APIs for FSDP2:

- Saving gathers a **full (unsharded)** state dict with `full_state_dict=True` (and `cpu_offload=True`).
- Loading uses `set_model_state_dict(..., full_state_dict=True, broadcast_from_rank0=True)` so rank0 provides the full weights and each rank **re-shards locally**. But this behavior is untested and I intend to explicitly test it.

Implication:

- **Resuming with a different `world_size` / GPU count should work**, because we are saving/loading a **full** state dict (not a sharded DTensor checkpoint).

When it would *not* work:

- If we saved a **sharded** checkpoint (DTensor placements tied to a device mesh), then the checkpoint would generally be coupled to the original sharding/world-size (unless additional resharding logic is implemented).

## Checkpoint Format Notes (HF Sharded Saving)

Our checkpoint code collects a full state dict via DCP, but then writes it using HuggingFace `save_pretrained(..., state_dict=...)` when available (see `src/ludic/training/checkpoint.py`).

Notes:

- `save_pretrained()` may write **sharded** weights for large models (e.g. `model-00001-of-000xx.safetensors` + `model.safetensors.index.json`).
- Our loader (`CheckpointManager._read_state_dict`) supports:
  - single-file `pytorch_model.bin` / `pytorch_model.pt`
  - sharded torch checkpoints via `pytorch_model.bin.index.json`
  - single-file `model.safetensors`
  - sharded safetensors via `model.safetensors.index.json` (requires `safetensors`)

## Truncation Semantics (Env vs LLM vs Protocol)

There are multiple "things that look like truncation" in the system. For correctness and for clean training stats, these are now tracked as distinct signals.

### Current Implementation

- **Env-level episode endings:** `StepOutcome` and `Step` carry `terminated` and `truncated` booleans (`src/ludic/types.py`), and interaction protocols stop when either is true.
- **Protocol time-limit truncation:** When a protocol hits `max_steps` without env termination/truncation:
  - The last step's `truncated` is set to `True`
  - `Step.info["truncation_reason"] = "max_steps"` is added
  - `Rollout.meta["episode_truncated"] = True` and `Rollout.meta["truncation_reason"] = "max_steps"` are set
  - For parser failure steps, the synthetic `next_obs` is preserved (not cleared to None)
- **LLM "finish reason" is separate:** the vLLM/OpenAI response `finish_reason` is captured into `ChatResponse.finish_reason` and merged into `Step.info` (via `ChatResponse.to_info()`), but this is *not* the same as env truncation.
- **Incomplete LLM completions (default = reject):** if the model returns `finish_reason == "length"` and the `Agent` has `reject_incomplete_completions=True` (default), the step is treated as a parse failure:
  - `ParseResult.action=None` so the protocol does not call `env.step()`
  - `Step.info["incomplete_completion"] = True` is set (and `finish_reason` is preserved in `Step.info`)
  - the raw truncated assistant message is still appended to the agent context, and the agent also receives the synthetic feedback observation for the next turn
- **Training sees truncation:** `RolloutEngine.generate_batch()` now propagates `Step.truncated` and `Step.terminated` into `SAWItem.meta`, along with `episode_truncated` and `truncation_reason` from `Rollout.meta`.

### Definitions

Three concepts are kept separate:

- `terminated`: the environment reached a true terminal state (success/failure terminal).
- `truncated`: the environment (or protocol) ended an episode due to a time limit / external cutoff (not necessarily terminal in the MDP sense).
  - `truncation_reason = "max_steps"`: protocol time limit hit
  - `truncation_reason = "env"`: environment-initiated truncation
- `llm_finish_reason`: the model stopped generating for some reason (`stop`, `length`, `tool_calls`, etc.).

### Should we mask out truncated samples for training?

Not universally; it depends on what `truncated` means in the environment and the algorithm:

- If truncation is a **Gym-style time limit** (episode cut off but task is "continuing"), then pure Monte Carlo returns (no value bootstrap) treat truncation like terminal and can bias learning. In that case:
  - a "correct" RL treatment is to **bootstrap** from the value function at the truncation boundary (requires a critic/value model), or
  - if you're doing pure Monte Carlo without bootstrapping, it can be safer to **drop** truncated episodes or downweight them to avoid bias (at the cost of data efficiency).
- If truncation is a **meaningful failure mode** ("timed out" is a real terminal condition for the task), then you should **keep** truncated samples and ensure the reward signal reflects the timeout.

### LLM `finish_reason == "length"` is not env truncation

`finish_reason=="length"` typically means the model hit `max_tokens` and produced an incomplete completion/action.

In Ludic (by default), these incomplete completions are rejected at the `Agent` level and treated like parse failures (see above), so they can be tracked and filtered separately from env/protocol truncation.
