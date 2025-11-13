import atexit
import logging
import time
from typing import Any, Dict, List, Mapping, Optional, Tuple

import requests
import torch  # type: ignore
from openai import AsyncOpenAI
from requests import ConnectionError
from requests.adapters import HTTPAdapter
from requests.exceptions import RequestException, Timeout
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
from vllm.distributed.utils import StatelessProcessGroup

from ludic.types import Message
from ludic.inference.client import ChatClient, ChatResponse
from ludic.inference.sampling import SamplingConfig

log = logging.getLogger(__name__)


class VLLMChatClient(ChatClient):
    """
    vLLM ChatClient backed by a vLLM OpenAI-compatible server
    + a NCCL-based weight update path.

    Design:
      - The client can run in two modes:
          * inference-only (enable_weight_updates=False):
                - Only uses the OpenAI-compatible HTTP interface.
                - Does NOT initialize NCCL or require a GPU on the client.
                - Any attempt to push updates fails explicitly.
          * training / update-capable (enable_weight_updates=True):
                - Initializes a NCCL communicator to participate as an extra rank.
                - Allows push_update_atomic() to broadcast weights to server workers.
      - This keeps the inference-only use case lightweight while preserving
        the full training/update path for specialized clients.
    """

    def __init__(
        self,
        *,
        host: str = "0.0.0.0",
        port: int = 8000,
        group_port: int = 51216,
        connection_timeout_s: float = 0.0,
        enable_weight_updates: bool = False,
    ) -> None:

        # Store configuration parameters
        self.host = host
        self.port = port
        self.group_port = group_port
        self.connection_timeout_s = connection_timeout_s
        self.enable_weight_updates = enable_weight_updates

        # AsyncOpenAI handles /v1/chat/completions, etc.
        self._async_client = AsyncOpenAI(
            base_url=f"http://{self.host}:{self.port}/v1",
            api_key="local",
        )

        # Sync HTTP for control-plane / health / weight-update RPC triggers
        self._session = requests.Session()
        adapter = HTTPAdapter(
            pool_connections=10,
            pool_maxsize=10,
            max_retries=3,
        )
        self._session.mount("http://", adapter)
        self._session.mount("https://", adapter)

        self.server_url = f"http://{self.host}:{self.port}"
        self._pynccl_comm: Optional[PyNcclCommunicator] = None
        self._rank: Optional[int] = None

        # Ensure server is reachable
        self._check_server(self.connection_timeout_s)

        # In inference-only mode, we intentionally skip NCCL initialization so
        # the client can run without a local GPU and without any distributed
        # setup. Weight updates are blocked in that mode.
        if self.enable_weight_updates:
            self._init_communicator()
            atexit.register(self.close_communicator)

    # ---- ChatClient.complete ------------------------------------

    async def complete(
        self,
        *,
        model: str,
        messages: List[Message],
        sampling: SamplingConfig,
    ) -> Tuple[ChatResponse, Dict[str, Any]]:
        # Map SamplingConfig -> OpenAI params
        request_kwargs: Dict[str, Any] = dict(
            model=model,
            messages=messages,
        )
        request_kwargs.update(sampling.to_openai_kwargs())

        resp = await self._async_client.chat.completions.create(**request_kwargs)

        choice = resp.choices[0]
        text = choice.message.content or ""
        finish_reason = choice.finish_reason

        # vLLM OpenAI server can optionally return logprobs, tokens, etc. in extras;
        # for now, keep it minimal and put raw response into info.
        chat_resp = ChatResponse(text=text, finish_reason=finish_reason)
        info: Dict[str, Any] = {
            "raw_response": resp.model_dump(exclude_none=True),
            "used_args": request_kwargs,
        }
        return chat_resp, info

    # ---- ChatClient.push_update_atomic --------------------------

    def push_update_atomic(
        self,
        params: Mapping[str, torch.Tensor],
        *,
        timeout_s: float = 600.0,
        reset_cache: bool = True,
        version: Optional[str] = None,
        check_shapes: bool = True,
    ) -> str:
        """
        Example implementation:
          - for each named param:
              * POST /update_named_param (name, dtype, shape)
              * NCCL broadcast tensor from this client to workers
          - optional reset_prefix_cache
          - returns a version string (dummy or supplied)
        """

        if self._pynccl_comm is None or self._rank is None:
            if not self.enable_weight_updates:
                raise RuntimeError(
                    "push_update_atomic() called on an inference-only client "
                    "(enable_weight_updates=False). Construct the client with "
                    "enable_weight_updates=True to enable weight updates."
                )
            raise RuntimeError(
                "Communicator not initialized (NCCL setup failed or not completed)."
            )

        start = time.time()
        for name, tensor in params.items():
            dtype, shape = str(tensor.dtype), tuple(tensor.shape)
            url = f"{self.server_url}/update_named_param"

            try:
                resp = self._session.post(
                    url,
                    json={
                        "name": name,
                        "dtype": dtype,
                        "shape": shape,
                    },
                    timeout=timeout_s,
                )
            except Timeout:
                raise TimeoutError(
                    f"HTTP timeout when sending metadata for {name}"
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Error sending metadata for {name}: {exc}"
                ) from exc

            if resp.status_code != 200:
                raise RuntimeError(
                    f"Server rejected update_named_param({name}): "
                    f"{resp.status_code} {resp.text}"
                )

            # Now broadcast the actual tensor to all workers.
            self._pynccl_comm.broadcast(tensor, src=self._rank)
            self._pynccl_comm.group.barrier()

            if (time.time() - start) > timeout_s:
                raise TimeoutError(
                    f"push_update_atomic exceeded {timeout_s} seconds"
                )

        if reset_cache:
            self.reset_prefix_cache()

        # Optional: wait until server background tasks drained
        while self.get_num_background_tasks() > 0:
            time.sleep(0.2)

        return version or f"vllm-{int(time.time())}"

    # ---- Control-plane helpers ---------------------------------

    def _check_server(
        self,
        total_timeout: float = 0.0,
        retry_interval: float = 2.0,
    ):
        url = f"{self.server_url}/health"
        start_time = time.time()
        while True:
            try:
                r = self._session.get(url, timeout=5.0)
                if r.status_code == 200:
                    log.info("vLLM server is up")
                    return
            except RequestException:
                pass

            if total_timeout and (time.time() - start_time) >= total_timeout:
                raise ConnectionError(
                    f"vLLM server not reachable at {self.host}:{self.port} "
                    f"after {total_timeout} seconds"
                )
            log.info("vLLM server not ready, retrying...")
            time.sleep(retry_interval)

    def _init_communicator(self) -> None:
        # 1) query world size
        r = self._session.get(
            f"{self.server_url}/get_world_size",
            timeout=10.0,
        )
        r.raise_for_status()
        vllm_world_size = r.json()["world_size"]
        world_size = vllm_world_size + 1
        self._rank = vllm_world_size

        # 2) ask server workers to init their communicators
        r = self._session.post(
            f"{self.server_url}/init_communicator",
            json={
                "host": self.host,
                "port": self.group_port,
                "world_size": world_size,
            },
            timeout=30.0,
        )
        r.raise_for_status()

        time.sleep(0.1)  # let server side spin up NCCL

        # 3) create client-side process group
        pg = StatelessProcessGroup.create(
            host=self.host,
            port=self.group_port,
            rank=self._rank,
            world_size=world_size,
        )
        device = 0
        self._pynccl_comm = PyNcclCommunicator(pg, device=device)

    def reset_prefix_cache(self) -> None:
        r = self._session.post(
            f"{self.server_url}/reset_prefix_cache",
            timeout=30.0,
        )
        r.raise_for_status()

    def get_num_background_tasks(self) -> int:
        r = self._session.post(
            f"{self.server_url}/get_num_background_tasks",
            timeout=10.0,
        )
        r.raise_for_status()
        return r.json()["num_background_tasks"]

    def close_communicator(self) -> None:
        try:
            r = self._session.post(
                f"{self.server_url}/close_communicator",
                timeout=10.0,
            )
            if r.status_code != 200:
                log.warning(
                    "close_communicator responded with %s %s",
                    r.status_code,
                    r.text,
                )
        except ConnectionError:
            # server might already be down
            pass
