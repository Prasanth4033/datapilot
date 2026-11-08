"""One place to call any LLM: timeouts, retries with backoff, rate limiting, logging."""
from __future__ import annotations

import time
from dataclasses import dataclass

from tenacity import RetryError, Retrying, retry_if_exception, stop_after_attempt, wait_random_exponential

from core import tokens
from core.config import Settings, load_domain
from core.tokens import chat_breakdown
from core.db import get_engine, log_call
from core.rate_limiter import RateLimiter

# USD per 1M tokens (input, output). Check your provider's pricing page and update.
PRICES = {"gpt-4o-mini": (0.15, 0.60)}


@dataclass
class LLMResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    attempts: int


class RetryableError(Exception):
    """Rate limits, timeouts and 5xx errors: worth trying again."""


# ---------- providers: each returns (text, prompt_tokens, completion_tokens) ----------

class OpenAIProvider:
    def __init__(self, model: str, timeout_s: float):
        import openai
        self._openai = openai
        self.client = openai.OpenAI(timeout=timeout_s, max_retries=0)  # we do retries ourselves
        self.model = model

    def complete(self, system: str, user: str, max_tokens: int):
        o = self._openai
        try:
            r = self.client.chat.completions.create(
                model=self.model, max_tokens=max_tokens,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            )
        except (o.RateLimitError, o.APITimeoutError, o.APIConnectionError, o.InternalServerError) as e:
            raise RetryableError(str(e)) from e
        return r.choices[0].message.content, r.usage.prompt_tokens, r.usage.completion_tokens


class FakeProvider:
    """No API key needed. Used by tests and the burst test."""
    def __init__(self, fail_first: int = 0):
        self.calls, self.fail_first = 0, fail_first

    def complete(self, system: str, user: str, max_tokens: int):
        self.calls += 1
        if self.calls <= self.fail_first:
            raise RetryableError("simulated 429 rate limit")
        answer = f"echo: {user}"
        return answer, chat_breakdown(system, user)["total"], tokens.count(answer)


def make_provider(s: Settings):
    if s.provider == "openai":
        return OpenAIProvider(s.model, s.timeout_s)
    if s.provider == "fake":
        return FakeProvider()
    raise ValueError(f"Unknown LLM_PROVIDER '{s.provider}'")


def estimate_tokens(text: str) -> int:
    return max(1, tokens.count(text))  # real tokenizer if available, else approximate


# ------------------------------------- client -------------------------------------

class LLMClient:
    def __init__(self, settings: Settings | None = None, provider=None, limiter: RateLimiter | None = None,
                 engine=None, wait=None):
        self.s = settings or Settings()
        self.provider = provider or make_provider(self.s)
        self.limiter = limiter or RateLimiter(self.s.rpm_limit, self.s.tpm_limit)
        self.engine = engine or get_engine(self.s.database_url)
        self.domain = load_domain(self.s.domain)
        self._wait = wait or wait_random_exponential(multiplier=1, max=30)  # backoff with jitter

    def ask(self, question: str, max_tokens: int = 400) -> LLMResult:
        system = self.domain.system_prompt
        estimate = estimate_tokens(system + question) + max_tokens
        waited = self.limiter.acquire(estimate)
        start, attempts = time.perf_counter(), 0
        try:
            for attempt in Retrying(stop=stop_after_attempt(self.s.max_retries), wait=self._wait,
                                    retry=retry_if_exception(lambda e: isinstance(e, RetryableError)),
                                    reraise=False):
                with attempt:
                    attempts = attempt.retry_state.attempt_number
                    text, p_tok, c_tok = self.provider.complete(system, question, max_tokens)
        except (RetryError, Exception) as e:
            cause = e.last_attempt.exception() if isinstance(e, RetryError) else e
            self._log(attempts, waited, start, 0, 0, "error", repr(cause))
            raise RuntimeError(f"LLM call failed after {attempts} attempt(s): {cause}") from cause
        self.limiter.correct(estimate, p_tok + c_tok)
        latency = self._log(attempts, waited, start, p_tok, c_tok, "ok", None)
        return LLMResult(text, p_tok, c_tok, latency, attempts)

    def _log(self, attempts, waited, start, p_tok, c_tok, status, error) -> float:
        latency = (time.perf_counter() - start) * 1000
        price_in, price_out = PRICES.get(self.s.model, (0.0, 0.0))
        log_call(self.engine, provider=self.s.provider, model=self.s.model, domain=self.domain.name,
                 prompt_tokens=p_tok, completion_tokens=c_tok, latency_ms=round(latency, 1),
                 attempts=attempts, waited_ms=round(waited * 1000, 1), status=status, error=error,
                 cost_usd=(p_tok * price_in + c_tok * price_out) / 1e6)
        return latency
