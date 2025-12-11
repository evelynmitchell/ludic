from __future__ import annotations

from typing import Any, Callable, Mapping, Optional, Tuple

from ludic.envs.single_agent_env import SingleAgentEnv
from ludic.types import Info, Observation, StepOutcome


Sample = Mapping[str, Any]
ParserFn = Callable[[str], str]
ComparatorFn = Callable[[str, str], bool]
PromptBuilder = Callable[[Sample], str]


def _identity_parser(text: str) -> str:
    return text.strip()


def _default_comparator(a: str, b: str) -> bool:
    """Simple case-insensitive comparison."""
    return a.strip().lower() == b.strip().lower()


class DatasetQAEnv(SingleAgentEnv):
    """
    A one-shot QA environment for a single sample.

    Each reset() exposes the sample's question as the observation, and the
    episode ends after a single agent answer is graded against the sample's
    ground truth.

    Parsing/normalization is caller-provided via `answer_parser` and
    `target_parser`; by default answers are only stripped.
    """

    def __init__(
        self,
        sample: Sample,
        *,
        prompt_key: str = "question",
        answer_key: str = "answer",
        system_prompt: Optional[str] = None,
        prompt_builder: Optional[PromptBuilder] = None,
        answer_parser: ParserFn = _identity_parser,
        target_parser: ParserFn = _identity_parser,
        comparator: ComparatorFn = _default_comparator,
        correct_reward: float = 1.0,
        incorrect_reward: float = 0.0,
        parse_error_reward: float = -1.0,
    ) -> None:
        super().__init__()
        self._sample: Sample = sample
        self._prompt_key = prompt_key
        self._answer_key = answer_key
        self._system_prompt = system_prompt
        self._prompt_builder = prompt_builder
        self._answer_parser = answer_parser
        self._target_parser = target_parser
        self._comparator = comparator
        self._correct_reward = correct_reward
        self._incorrect_reward = incorrect_reward
        self._parse_error_reward = parse_error_reward
        self._current_prompt: Observation = ""
        self._current_answer: str = ""
        self._current_id: Optional[str] = None
        self._done: bool = False
        self._latest_obs: Observation = ""

    # ------------------------------------------------------------------
    # SingleAgentEnv implementation
    # ------------------------------------------------------------------

    @property
    def suggested_sysprompt(self) -> Optional[str]:
        return self._system_prompt

    def _build_prompt(self, sample: Sample) -> Observation:
        if self._prompt_builder is not None:
            return self._prompt_builder(sample)

        if self._prompt_key not in sample:
            raise KeyError(f"Sample missing prompt_key {self._prompt_key!r}: {sample}")

        return str(sample[self._prompt_key])

    def _get_answer(self, sample: Sample) -> str:
        if self._answer_key not in sample:
            raise KeyError(f"Sample missing answer_key {self._answer_key!r}: {sample}")
        return str(sample[self._answer_key])

    def env_reset(self, *, seed: Optional[int] = None) -> Tuple[Observation, Info]:
        prompt = self._build_prompt(self._sample)
        answer = self._get_answer(self._sample)

        self._current_prompt = prompt
        self._current_answer = answer
        self._current_id = str(self._sample.get("id") or self._sample.get("uid") or 0)
        self._done = False
        self._latest_obs = prompt

        info: Info = {
            "question_id": self._current_id,
        }
        return prompt, info

    def env_step(self, action: str) -> StepOutcome:
        if self._done:
            raise RuntimeError("env_step called after episode finished. Call reset().")

        info: Info = {
            "question_id": self._current_id,
            "raw_action": action,
        }

        try:
            parsed_answer = self._answer_parser(action)
        except Exception as e:
            self._done = True
            info.update({"parse_error": True, "error": str(e)})
            obs = f"Could not parse answer: {e}"
            self._latest_obs = obs
            return StepOutcome(
                obs=obs,
                reward=self._parse_error_reward,
                truncated=True,
                terminated=False,
                info=info,
            )

        parsed_target = self._target_parser(self._current_answer)
        correct = self._comparator(parsed_answer, parsed_target)
        self._done = True

        info.update(
            {
                "parsed_answer": parsed_answer,
                "target_answer": parsed_target,
                "correct": correct,
            }
        )

        obs = "✅ Correct." if correct else f"❌ Incorrect. Expected {parsed_target}."
        reward = self._correct_reward if correct else self._incorrect_reward
        self._latest_obs = obs

        return StepOutcome(
            obs=obs,
            reward=reward,
            truncated=False,
            terminated=True,
            info=info,
        )

    def env_current_obs(self) -> Observation:
        return self._latest_obs
