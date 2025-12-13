from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from ludic.inference.vllm_client import VLLMChatClient
from ludic.inference.sampling import get_default_sampling_config

pytestmark = [pytest.mark.integration, pytest.mark.gpu]


@pytest.mark.asyncio
async def test_vllm_client_returns_logprobs(
    vllm_client: VLLMChatClient,
    vllm_model_name: str,
) -> None:
    """
    Ensure vLLM returns per-token logprobs when requested (logprobs=1 by default).
    """
    sampling = get_default_sampling_config()

    messages = [
        {"role": "system", "content": "You are a test assistant. Answer concisely."},
        {"role": "user", "content": "Say the word 'ping'."},
    ]

    resp, _ = await vllm_client.complete(
        model=vllm_model_name,
        messages=messages,
        sampling=sampling,
        return_token_ids=True,
    )

    assert resp.text.strip() != ""
    assert resp.completion_token_ids is not None
    assert resp.completion_logprobs is not None
    assert len(resp.completion_logprobs) == len(resp.completion_token_ids)

    # Cross-check by teacher-forcing the same model locally.
    # Release client resources before loading HF model to avoid GPU OOM.
    if hasattr(vllm_client, "_session"):
        try:
            vllm_client._session.close()
        except Exception:
            pass
    if hasattr(vllm_client, "_pynccl_comm") and vllm_client._pynccl_comm is not None:
        try:
            vllm_client.close_communicator()
        except Exception:
            pass

    # Load HF model on CPU (fp32) to avoid GPU OOM and keep scoring stable.
    hf_dtype = torch.float32
    hf_device = torch.device("cpu")
    model = AutoModelForCausalLM.from_pretrained(
        vllm_model_name,
        dtype=hf_dtype,
        device_map={"": hf_device},
        torch_dtype=hf_dtype,
    )
    device = hf_device
    model.eval()

    prompt_ids = resp.prompt_token_ids
    completion_ids = resp.completion_token_ids
    assert prompt_ids is not None

    combined = prompt_ids + completion_ids
    input_ids = torch.tensor(combined, dtype=torch.long, device=device).unsqueeze(0)
    attn = torch.ones_like(input_ids)

    with torch.inference_mode():
        logits = model(input_ids=input_ids, attention_mask=attn).logits
        token_logp = F.log_softmax(logits[:, :-1, :], dim=-1)  # [1, T-1, V]
        targets = input_ids[:, 1:]  # [1, T-1]
        gathered = torch.gather(token_logp, dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)  # [1, T-1]

    prompt_len = len(prompt_ids)
    completion_mask = torch.zeros_like(gathered, dtype=torch.bool)
    completion_mask[:, prompt_len - 1 :] = True  # predictions corresponding to completion tokens
    hf_logprobs = gathered[completion_mask].cpu().tolist()

    assert len(hf_logprobs) == len(resp.completion_logprobs)
    # Allow small numeric drift between vLLM and HF scoring.
    for got, expected in zip(hf_logprobs, resp.completion_logprobs):
        assert got == pytest.approx(expected, rel=1e-2, abs=0.3)
