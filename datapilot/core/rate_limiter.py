"""Client-side limiter for requests per minute (RPM) and tokens per minute (TPM).

Sliding 60-second window: before each call we check how many requests and tokens
were used in the last minute, and wait if this call would go over either limit.
"""
from __future__ import annotations

import threading
import time
from collections import deque


class RateLimiter:
    def __init__(self, rpm: int, tpm: int, window_s: float = 60.0, clock=time.monotonic, sleep=time.sleep):
        self.rpm, self.tpm, self.window = rpm, tpm, window_s
        self._clock, self._sleep = clock, sleep
        self._events: deque[tuple[float, int]] = deque()  # (timestamp, tokens)
        self._lock = threading.Lock()

    def _trim(self, now: float) -> None:
        while self._events and now - self._events[0][0] >= self.window:
            self._events.popleft()

    def acquire(self, tokens: int) -> float:
        """Block until a call using `tokens` fits. Returns seconds waited."""
        if tokens > self.tpm:
            raise ValueError(f"One call needs {tokens} tokens, above the TPM limit {self.tpm}")
        waited = 0.0
        while True:
            with self._lock:
                now = self._clock()
                self._trim(now)
                used = sum(t for _, t in self._events)
                if len(self._events) < self.rpm and used + tokens <= self.tpm:
                    self._events.append((now, tokens))
                    return waited
                # Wait until the oldest call leaves the window.
                pause = max(self.window - (now - self._events[0][0]), 0.01)
            self._sleep(pause)
            waited += pause

    def check(self, tokens: int) -> tuple[bool, str, float]:
        """Would a call fit right now? Returns (ok, reason, retry_after_s). Records nothing."""
        with self._lock:
            now = self._clock()
            self._trim(now)
            used = sum(t for _, t in self._events)
            retry = round(self.window - (now - self._events[0][0]), 1) if self._events else 0.0
            if len(self._events) >= self.rpm:
                return False, f"RPM limit reached ({len(self._events)}/{self.rpm} requests this minute)", retry
            if used + tokens > self.tpm:
                return False, f"TPM limit reached ({used}+{tokens} > {self.tpm} tokens this minute)", retry
            return True, "ok", 0.0

    def status(self) -> dict:
        """What is used and left in the current 60-second window (no waiting, no recording)."""
        with self._lock:
            now = self._clock()
            self._trim(now)
            used_req = len(self._events)
            used_tok = sum(t for _, t in self._events)
            reset_in = self.window - (now - self._events[0][0]) if self._events else 0.0
        return {
            "rpm_limit": self.rpm, "requests_used": used_req, "requests_left": self.rpm - used_req,
            "tpm_limit": self.tpm, "tokens_used": used_tok, "tokens_left": self.tpm - used_tok,
            "resets_in_s": round(reset_in, 1),
        }

    def correct(self, estimated: int, actual: int) -> None:
        """After a call, replace our token estimate with the real count."""
        with self._lock:
            for i in range(len(self._events) - 1, -1, -1):
                ts, t = self._events[i]
                if t == estimated:
                    self._events[i] = (ts, actual)
                    return
