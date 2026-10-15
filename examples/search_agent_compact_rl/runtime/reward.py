from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import string
import time
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from filelock import FileLock, Timeout
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    RateLimitError,
)

JUDGE_SYSTEM_PROMPT = """You are an expert evaluator. Given a question, a ground truth answer, and a predicted answer, judge whether the predicted answer correctly and completely matches the ground truth answer for the question. Accept equivalent wording, aliases, abbreviations, and semantically identical answers. Reply with only a number: 1 if correct, 0 if incorrect."""

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class JudgeResult:
    score: float
    metrics: dict[str, float]


def _judge_tracking_path() -> str | None:
    return os.environ.get("SA_JUDGE_TRACKING_FILE") or None


class _JudgeUsageLogger:
    async def log(self, record: dict[str, Any]) -> None:
        path = _judge_tracking_path()
        if path is None:
            return

        line = json.dumps(record, ensure_ascii=True) + "\n"
        lock = FileLock(f"{path}.lock", timeout=5.0)

        def _write() -> None:
            try:
                Path(path).parent.mkdir(parents=True, exist_ok=True)
                with lock:
                    with open(path, "a", encoding="utf-8") as output:
                        output.write(line)
            except Timeout:
                warnings.warn(
                    f"Judge tracking logger could not acquire lock for {path} within 5s; " "dropping log line.",
                    stacklevel=2,
                )
            except OSError as exc:
                warnings.warn(
                    f"Judge tracking logger could not write to {path}: {exc!r}; dropping log line.",
                    stacklevel=2,
                )

        await asyncio.to_thread(_write)


_judge_usage_logger = _JudgeUsageLogger()

# ---------------------------------------------------------------------------
# F1 fallback helpers (Chinese-aware)
# ---------------------------------------------------------------------------


def _normalize_answer(s: str) -> str:
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def _bool_mapping(s: str) -> str:
    if s == "True":
        return "yes"
    if s == "False":
        return "no"
    return s


def _contains_chinese(text: str) -> bool:
    for char in text:
        if "一" <= char <= "鿿":
            return True
        if "㐀" <= char <= "䶿":
            return True
        if "豈" <= char <= "﫿":
            return True
    return False


def _normalize_text(text: str) -> str:
    for punct in string.punctuation:
        text = text.replace(punct, " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def _extract_answer(solution_str: str) -> str | None:
    """Extract the last <answer>...</answer> content."""
    matches = list(re.finditer(r"<answer>(.*?)</answer>", solution_str, re.DOTALL))
    for match in reversed(matches):
        answer = match.group(1).strip()
        if answer:
            return answer
    return None


def _f1_score(answer_content: str, gt: str) -> float:
    answer_content = _normalize_text(_bool_mapping(answer_content))
    gt = _normalize_text(_bool_mapping(gt))

    if _contains_chinese(gt):

        def parse_chinese_str(s):
            numbers = []
            for i, c in enumerate(s):
                if c.isdigit():
                    if i > 0 and s[i - 1].isdigit():
                        numbers[-1] = numbers[-1] + c
                    else:
                        numbers.append(c)
            for c in "0123456789，。 ,.-":
                s = s.replace(c, "")
            return set(list(s) + numbers)

        pred_tokens = parse_chinese_str(answer_content)
        gt_tokens = parse_chinese_str(gt)
    else:
        pred_tokens = set(answer_content.split())
        gt_tokens = set(gt.split())

    if not gt_tokens or not pred_tokens:
        return 0.0

    common = pred_tokens & gt_tokens
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gt_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def _compute_f1(solution_str: str, ground_truth: Any) -> tuple[str | None, float]:
    """Compute F1 against the answer extracted from the response."""
    if ground_truth is None:
        return None, 0.0

    answer = _extract_answer(solution_str)
    if answer is None:
        return None, 0.0

    if isinstance(ground_truth, (list, tuple)):
        if not ground_truth:
            return answer, 0.0
        scores = [_f1_score(answer, str(g)) for g in ground_truth]
        return answer, max(scores)

    return answer, _f1_score(answer, str(ground_truth))


def _get_ground_truth(sample: Any) -> Any:
    ground_truth = getattr(sample, "label", None)
    if ground_truth is None and hasattr(sample, "metadata"):
        metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
        ground_truth = metadata.get("label")
    if isinstance(ground_truth, (list, tuple)) and not ground_truth:
        return None
    return ground_truth


def _format_ground_truth(ground_truth: Any) -> str:
    if isinstance(ground_truth, (list, tuple)):
        return "\n".join(f"- {item}" for item in ground_truth)
    return str(ground_truth)


_judge_client: AsyncOpenAI | None = None
_judge_semaphore: asyncio.Semaphore | None = None


def _normalize_openai_base_url(base_url: str) -> str:
    """Ensure OpenAI-compatible providers expose the /v1 path."""
    base_url = base_url.strip()
    if not base_url:
        return base_url
    # xiaoma.best and other New-API gateways serve the dashboard at the root;
    # the API lives under /v1.
    if "xiaoma.best" in base_url and not base_url.rstrip("/").endswith("/v1"):
        base_url = base_url.rstrip("/") + "/v1"
    return base_url


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if value is None or value == "":
        raise ValueError(f"Missing required environment variable: {name}")
    return value


def _require_int_env(name: str) -> int:
    raw = _require_env(name)
    try:
        return int(raw)
    except ValueError as e:
        raise ValueError(f"Invalid {name}: {raw!r}") from e


def _require_float_env(name: str) -> float:
    raw = _require_env(name)
    try:
        return float(raw)
    except ValueError as e:
        raise ValueError(f"Invalid {name}: {raw!r}") from e


def _get_judge_client() -> AsyncOpenAI:
    """Return a cached AsyncOpenAI client for the judge."""
    global _judge_client
    if _judge_client is None:
        base_url = _normalize_openai_base_url(_require_env("JUDGE_BASE_URL"))
        api_key = _require_env("JUDGE_API_KEY")
        timeout = _require_float_env("JUDGE_TIMEOUT_SECONDS")
        _judge_client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
            max_retries=0,
        )
    return _judge_client


def _get_judge_semaphore() -> asyncio.Semaphore:
    """Return a cached semaphore to bound concurrent judge calls."""
    global _judge_semaphore
    if _judge_semaphore is None:
        concurrency = _require_int_env("JUDGE_MAX_CONCURRENCY")
        _judge_semaphore = asyncio.Semaphore(concurrency)
    return _judge_semaphore


def _is_retryable_error(exc: Exception) -> bool:
    """Decide whether a failed judge call is worth retrying."""
    if isinstance(exc, RateLimitError):
        return True
    if isinstance(exc, (APITimeoutError, APIConnectionError)):
        return True
    if isinstance(exc, APIStatusError):
        return exc.status_code in (429, 500, 502, 503, 504, 524)
    return False


def judge_extra_body(model_name: str) -> dict[str, Any] | None:
    """Return provider-specific extra request body for judge calls."""
    if model_name == "deepseek-v4-flash":
        return {"thinking": {"type": "disabled"}}
    return None


def _parse_score(response: Any) -> float | None:
    """Parse a 0/1 score from a ChatCompletion response, or None if invalid."""
    if not hasattr(response, "choices") or not response.choices:
        logger.warning(
            "Judge returned a non-standard response (type=%s): %.200r",
            type(response).__name__,
            response,
        )
        return None

    content = (response.choices[0].message.content or "").strip()
    score_match = re.search(r"\b(0|1)\b", content)
    if score_match:
        return 1.0 if score_match.group(1) == "1" else 0.0

    logger.warning("Judge returned an unparseable score: %r", content)
    return None


def _usage_value(value: Any, name: str) -> Any:
    if value is None:
        return None
    if isinstance(value, dict):
        return value.get(name)
    return getattr(value, name, None)


def _judge_attempt_record(
    *,
    model_name: str,
    attempt: int,
    max_attempts: int,
    attempt_start: float,
    status: str,
    response: Any = None,
    error: Exception | None = None,
) -> dict[str, Any]:
    usage = getattr(response, "usage", None) if response is not None else None
    prompt_details = _usage_value(usage, "prompt_tokens_details")
    http_status = getattr(error, "status_code", None) if error is not None else None
    if http_status is None and response is not None:
        http_status = getattr(response, "status_code", None)
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model": model_name,
        "attempt": attempt,
        "max_attempts": max_attempts,
        "status": status,
        "http_status": http_status,
        "latency_sec": time.perf_counter() - attempt_start,
        "error_type": type(error).__name__ if error is not None else None,
        "prompt_tokens": _usage_value(usage, "prompt_tokens"),
        "completion_tokens": _usage_value(usage, "completion_tokens"),
        "total_tokens": _usage_value(usage, "total_tokens"),
        "cached_tokens": _usage_value(prompt_details, "cached_tokens"),
    }


async def track_judge_attempt(
    *,
    model_name: str,
    attempt: int,
    max_attempts: int,
    attempt_start: float,
    status: str,
    response: Any = None,
    error: Exception | None = None,
) -> None:
    await _judge_usage_logger.log(
        _judge_attempt_record(
            model_name=model_name,
            attempt=attempt,
            max_attempts=max_attempts,
            attempt_start=attempt_start,
            status=status,
            response=response,
            error=error,
        )
    )


async def _judge_one_with_metrics(args: Any, sample: Any) -> JudgeResult:
    """Judge a single sample with bounded concurrency, retries, and metrics."""
    metadata = getattr(sample, "metadata", None) or {}
    question = metadata.get("original_question", sample.prompt)
    if isinstance(question, list):
        question = "\n".join(m.get("content", "") for m in question if m.get("role") == "user")

    answer = _extract_answer(sample.response)
    if answer is None:
        return JudgeResult(
            score=0.0,
            metrics={
                "attempts": 0.0,
                "retries": 0.0,
                "latency_sec": 0.0,
                "success": 0.0,
                "fallback": 0.0,
                "timeout": 0.0,
                "status_524": 0.0,
                "retryable_error": 0.0,
                "unparseable": 0.0,
                "skipped_no_answer": 1.0,
            },
        )
    ground_truth = _get_ground_truth(sample)
    if ground_truth is None:
        return JudgeResult(
            score=0.0,
            metrics={
                "attempts": 0.0,
                "retries": 0.0,
                "latency_sec": 0.0,
                "success": 0.0,
                "fallback": 1.0,
                "timeout": 0.0,
                "status_524": 0.0,
                "retryable_error": 0.0,
                "unparseable": 0.0,
                "skipped_no_answer": 0.0,
                "skipped_no_ground_truth": 1.0,
            },
        )

    model_name = _require_env("JUDGE_MODEL")
    client = _get_judge_client()
    semaphore = _get_judge_semaphore()
    max_retries = _require_int_env("JUDGE_MAX_RETRIES")
    max_attempts = max_retries + 1
    start_time = time.perf_counter()
    attempts = 0
    timeout_count = 0
    status_524_count = 0
    retryable_error_count = 0
    unparseable_count = 0

    def build_metrics(*, source: str) -> dict[str, float]:
        latency_sec = time.perf_counter() - start_time
        fallback = 1.0 if source == "f1" else 0.0
        return {
            "attempts": float(attempts),
            "retries": float(max(0, attempts - 1)),
            "latency_sec": float(latency_sec),
            "success": 1.0 - fallback,
            "fallback": fallback,
            "timeout": float(timeout_count),
            "status_524": float(status_524_count),
            "retryable_error": float(retryable_error_count),
            "unparseable": float(unparseable_count),
        }

    async with semaphore:
        last_error: Exception | None = None
        for attempt in range(max_attempts):
            attempt_start = time.perf_counter()
            try:
                attempts += 1
                request_kwargs: dict[str, Any] = {
                    "model": model_name,
                    "messages": [
                        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": (
                                f"Question:\n{question}\n\n"
                                f"Ground Truth Answer:\n{_format_ground_truth(ground_truth)}\n\n"
                                f"Predicted Answer:\n{answer}\n\n"
                                "Score (0 or 1):"
                            ),
                        },
                    ],
                    "max_tokens": 10,
                    "temperature": 0.0,
                }
                extra_body = judge_extra_body(model_name)
                if extra_body is not None:
                    request_kwargs["extra_body"] = extra_body
                response = await client.chat.completions.create(**request_kwargs)
            except Exception as exc:
                await track_judge_attempt(
                    model_name=model_name,
                    attempt=attempt + 1,
                    max_attempts=max_attempts,
                    attempt_start=attempt_start,
                    status="error",
                    error=exc,
                )
                last_error = exc
                is_retryable = _is_retryable_error(exc)
                if isinstance(exc, APITimeoutError):
                    timeout_count += 1
                if isinstance(exc, APIStatusError):
                    if exc.status_code == 524:
                        status_524_count += 1
                        timeout_count += 1
                if is_retryable:
                    retryable_error_count += 1
                if is_retryable and attempt < max_attempts - 1:
                    backoff = 2**attempt
                    logger.warning(
                        "Judge API call failed (attempt %d/%d), retrying in %ds: %s",
                        attempt + 1,
                        max_attempts,
                        backoff,
                        exc,
                    )
                    await asyncio.sleep(backoff)
                    continue
                logger.warning(
                    "Judge API call failed (attempt %d/%d), not retryable: %s",
                    attempt + 1,
                    max_attempts,
                    exc,
                )
                break

            score = _parse_score(response)
            await track_judge_attempt(
                model_name=model_name,
                attempt=attempt + 1,
                max_attempts=max_attempts,
                attempt_start=attempt_start,
                status="success" if score is not None else "unparseable",
                response=response,
            )
            if score is not None:
                return JudgeResult(score=score, metrics=build_metrics(source="llm"))

            # Response was malformed / unparseable; retry once if we have attempts left.
            unparseable_count += 1
            if attempt < max_attempts - 1:
                backoff = 2**attempt
                logger.warning(
                    "Judge returned unparseable score (attempt %d/%d), retrying in %ds",
                    attempt + 1,
                    max_attempts,
                    backoff,
                )
                await asyncio.sleep(backoff)
                continue

            logger.warning("Judge returned unparseable score after %d attempts", max_attempts)
            break

    # Fall back to F1 when the LLM judge is unavailable or unparseable.
    if last_error is not None:
        logger.warning("Falling back to F1 reward after judge failed: %s", last_error)
    else:
        logger.warning("Falling back to F1 reward after judge returned no valid score")

    _, f1 = _compute_f1(sample.response, ground_truth)
    return JudgeResult(score=f1, metrics=build_metrics(source="f1"))


async def _judge_one(args: Any, sample: Any) -> float:
    """Judge a single sample and return only its reward score."""
    result = await _judge_one_with_metrics(args, sample)
    return result.score


async def browsecomp_judge_with_metrics(args, sample: Any) -> JudgeResult | list[JudgeResult]:
    """BrowseComp-style LLM judge returning reward scores plus judge metrics."""
    if isinstance(sample, list):
        return await asyncio.gather(*[_judge_one_with_metrics(args, s) for s in sample])
    return await _judge_one_with_metrics(args, sample)


async def browsecomp_judge(args, sample: Any) -> float | list[float]:
    """BrowseComp-style LLM judge.

    Supports both the slime batched custom reward interface (``sample`` is a
    list of samples) and the single-sample call used by the agent loop.
    """
    if isinstance(sample, list):
        return await asyncio.gather(*[_judge_one(args, s) for s in sample])
    return await _judge_one(args, sample)
