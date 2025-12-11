from __future__ import annotations

import re
from typing import Optional

from ludic.envs.dataset_qa_env import DatasetQAEnv, Sample


def math_answer_parser(text: str) -> str:
    """
    Normalize MATH-style answers:
      - strip whitespace
      - unbox \\boxed{...}
      - take text after '####' if present
      - drop commas and grab the last numeric/fraction token
    """
    cleaned = text.strip()

    boxed = re.search(r"\\boxed\{([^}]*)\}", cleaned)
    if boxed:
        cleaned = boxed.group(1).strip()

    if "####" in cleaned:
        cleaned = cleaned.split("####")[-1].strip()

    cleaned = cleaned.replace(",", "").strip()
    numeric_tokens = re.findall(r"-?\d+(?:/\d+)?(?:\.\d+)?", cleaned)
    if numeric_tokens:
        return numeric_tokens[-1].strip()
    return cleaned


class MATHEnv(DatasetQAEnv):
    """
    Convenience wrapper for MATH-style QA.
    """

    DEFAULT_SYSTEM_PROMPT = (
        "You are a careful math tutor. Think step by step. "
        "Put your final numeric answer in \\boxed{...} or after ####."
    )

    def __init__(
        self,
        sample: Sample,
        *,
        system_prompt: Optional[str] = DEFAULT_SYSTEM_PROMPT,
    ) -> None:
        super().__init__(
            sample=sample,
            prompt_key="problem" if "problem" in sample else "question",
            answer_key="solution" if "solution" in sample else "answer",
            system_prompt=system_prompt,
            answer_parser=math_answer_parser,
            target_parser=math_answer_parser,
        )
