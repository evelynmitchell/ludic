import asyncio
import os
import signal
from argparse import Namespace
from typing import Any, Awaitable, Sequence, Set

import torch
import uvloop
from fastapi import FastAPI, Request
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
from vllm.distributed.parallel_state import get_world_group
from vllm.distributed.utils import StatelessProcessGroup
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.launcher import serve_http
from vllm.entrypoints.openai.api_server import (
    build_app,
    create_server_socket,
    init_app_state,
)
from vllm.entrypoints.openai.cli_args import (
    make_arg_parser,
    validate_parsed_serve_args,
)
from vllm.usage.usage_lib import UsageContext
from vllm.utils import FlexibleArgumentParser, set_ulimit

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"


# ---------------------------------------------------------------------------
# Global state for weight updates & background tasks
# ---------------------------------------------------------------------------

MAX_CONCURRENT_WEIGHT_UPDATES = 10
weight_update_semaphore = asyncio.Semaphore(MAX_CONCURRENT_WEIGHT_UPDATES)

background_tasks: Set[asyncio.Task[Any]] = set()

RUNTIME_VERSION: int = 0
RUNTIME_VERSION_LOCK = asyncio.Lock()


def create_background_task(coro: Awaitable[Any]) -> asyncio.Task[Any]:
    """Create an async task and track it so we can wait/cancel on shutdown."""
    task = asyncio.create_task(coro)
    background_tasks.add(task)
    task.add_done_callback(background_tasks.discard)
    return task


# ---------------------------------------------------------------------------
# Worker extension: NCCL-based weight sync
# ---------------------------------------------------------------------------


class WeightSyncWorkerExtension:
    """
    vLLM worker extension for weight synchronization.

    Each worker:
      - joins a StatelessProcessGroup (TCP)
      - wraps it in a PyNcclCommunicator (NCCL)
      - receives updated weights via broadcast() from the client rank
      - calls `model_runner.model.load_weights` with the new tensors
    """

    pynccl_comm: PyNcclCommunicator | None = None
    client_rank: int | None = None
    device: torch.device | None = None

    def init_communicator(self, host: str, port: int, world_size: int) -> None:
        """
        Called via engine.collective_rpc on all workers.
        Creates the NCCL communicator for weight updates.
        """
        if self.pynccl_comm is not None:
            raise RuntimeError(
                "Weight update group already initialized. "
                "Call close_communicator() first."
            )

        rank = get_world_group().rank
        pg = StatelessProcessGroup.create(
            host=host,
            port=port,
            rank=rank,
            world_size=world_size,
        )
        assert self.device is not None, "WeightSyncWorkerExtension.device must be set"
        self.pynccl_comm = PyNcclCommunicator(pg, device=self.device)
        # client rank is the last rank in the world (host process)
        self.client_rank = world_size - 1

    def update_named_param(self, name: str, dtype: str, shape: Sequence[int]) -> None:
        """
        Called via engine.collective_rpc on all workers.
        Receives a single parameter tensor via NCCL broadcast and loads it.
        """
        if self.pynccl_comm is None or self.client_rank is None:
            raise RuntimeError(
                "Communicator not initialized. Call `init_communicator` first."
            )

        torch_dtype = getattr(torch, dtype.split(".")[-1])
        assert self.device is not None
        weight = torch.empty(shape, dtype=torch_dtype, device=self.device)
        self.pynccl_comm.broadcast(weight, src=self.client_rank)
        self.pynccl_comm.group.barrier()
        # vLLM model runner will apply the incoming weights
        self.model_runner.model.load_weights(weights=[(name, weight)])  # type: ignore[attr-defined]

    def close_communicator(self) -> None:
        """
        Called via engine.collective_rpc to tear down communicator state.
        """
        if self.pynccl_comm is not None:
            del self.pynccl_comm
            self.pynccl_comm = None
            self.client_rank = None


# ---------------------------------------------------------------------------
# Server / app setup
# ---------------------------------------------------------------------------


async def run_server(args: Namespace) -> None:
    sock_addr = (args.host or "0.0.0.0", args.port)
    sock = create_server_socket(sock_addr)

    set_ulimit()

    def signal_handler(*_: Any) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, signal_handler)

    # Build vLLM engine with our worker extension
    engine_args = AsyncEngineArgs.from_cli_args(args)
    # Adjust this string to the real import path of WeightSyncWorkerExtension
    engine_args.worker_extension_cls = (
        "ludic.inference.vllm_server.WeightSyncWorkerExtension"
    )
    engine = AsyncLLMEngine.from_engine_args(
        engine_args, usage_context=UsageContext.OPENAI_API_SERVER
    )

    app: FastAPI = build_app(args)

    # ------------------------ control-plane endpoints ---------------------
    
    #TODO: override 
    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/get_world_size")
    async def get_world_size() -> dict[str, int]:
        return {
            "world_size": args.tensor_parallel_size * args.data_parallel_size
        }

    @app.get("/runtime_version")
    async def runtime_version() -> dict[str, int]:
        return {"version": RUNTIME_VERSION}

    @app.post("/init_communicator")
    async def init_communicator(request: Request) -> dict[str, str]:
        """
        Client tells all workers to join a weight-sync process group.
        """
        data = await request.json()
        host = data.get("host")
        port = data.get("port")
        world_size = data.get("world_size")

        create_background_task(
            engine.collective_rpc(
                "init_communicator", args=(host, port, world_size)
            )
        )
        return {"status": "ok"}

    @app.post("/update_named_param")
    async def update_named_param(request: Request) -> dict[str, str]:
        """
        Update a single named parameter.

        Client side:
          1) POST name/dtype/shape here
          2) Immediately run NCCL broadcast(weights, src=client_rank)

        Worker side:
          - allocates empty tensor of given shape/dtype
          - calls broadcast(empty, src=client_rank)
          - loads the received tensor into the model
        """
        data = await request.json()
        name = data.get("name")
        dtype = data.get("dtype")
        shape = data.get("shape")
        shape_tuple = tuple(shape)

        async def throttled_update() -> None:
            async with weight_update_semaphore:
                await engine.collective_rpc(
                    "update_named_param", args=(name, dtype, shape_tuple)
                )

        create_background_task(throttled_update())
        return {"status": "ok"}

    @app.post("/push_update_atomic")
    async def push_update_atomic(request: Request) -> dict[str, Any]:
        """
        Optional batched update endpoint.

        Body: { "params": [ {name, dtype, shape}, ... ], "version": "optional-tag" }

        Semantics:
          - Schedules a background task that calls update_named_param for each param.
          - Bumps RUNTIME_VERSION once all workers have processed the batch.
          - HTTP returns immediately after scheduling; client can poll
            /get_num_background_tasks or /runtime_version if it wants to wait.
        """
        data = await request.json()
        params = data.get("params", [])
        requested_version = data.get("version")

        async def do_update() -> None:
            async with weight_update_semaphore:
                for p in params:
                    name = p["name"]
                    dtype = p["dtype"]
                    shape = tuple(p["shape"])
                    await engine.collective_rpc(
                        "update_named_param", args=(name, dtype, shape)
                    )
                global RUNTIME_VERSION
                async with RUNTIME_VERSION_LOCK:
                    RUNTIME_VERSION += 1

        create_background_task(do_update())
        return {"status": "ok", "requested_version": requested_version}

    @app.post("/reset_prefix_cache")
    async def reset_prefix_cache() -> dict[str, str]:
        """
        Reset any KV/prefix caches on the engine.
        """
        create_background_task(engine.reset_prefix_cache())
        return {"status": "ok"}

    @app.post("/get_num_background_tasks")
    async def get_num_background_tasks() -> dict[str, int]:
        return {"num_background_tasks": len(background_tasks)}

    @app.post("/close_communicator")
    async def close_communicator() -> dict[str, str]:
        """
        Tear down NCCL communicator on all workers.
        """
        await engine.collective_rpc("close_communicator")
        return {"status": "ok"}

    # ------------------------ start HTTP server --------------------------

    vllm_config = await engine.get_vllm_config()
    await init_app_state(engine, vllm_config, app.state, args)

    shutdown_task = await serve_http(
        app,
        sock,
        host=args.host,
        port=args.port,
        log_level=args.uvicorn_log_level,
        ssl_keyfile=args.ssl_keyfile,
        ssl_certfile=args.ssl_certfile,
        ssl_ca_certs=args.ssl_ca_certs,
        ssl_cert_reqs=args.ssl_cert_reqs,
    )
    await shutdown_task

    # graceful shutdown of background tasks
    for task in list(background_tasks):
        task.cancel()
    if background_tasks:
        await asyncio.gather(*background_tasks, return_exceptions=True)

    sock.close()


def main() -> None:
    parser = FlexibleArgumentParser(
        description="vLLM OpenAI-compatible server with weight synchronization"
    )
    parser = make_arg_parser(parser)
    args = parser.parse_args() or Namespace()
    validate_parsed_serve_args(args)
    print(args)
    uvloop.run(run_server(args))


if __name__ == "__main__":
    main()
