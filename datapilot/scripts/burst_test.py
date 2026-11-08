"""Thursday's test: 50 calls at once must all succeed with no rate-limit errors.

Usage: python -m scripts.burst_test            (uses LLM_PROVIDER from .env)
       python -m scripts.burst_test --fake     (no API key, free)
"""
import sys
from concurrent.futures import ThreadPoolExecutor

from sqlalchemy import func, select

from core.config import Settings
from core.db import llm_calls
from core.llm_client import FakeProvider, LLMClient

N = 50

if __name__ == "__main__":
    s = Settings()
    client = LLMClient(s, provider=FakeProvider() if "--fake" in sys.argv else None)
    with ThreadPoolExecutor(max_workers=10) as pool:
        results = list(pool.map(lambda i: client.ask(f"Reply with the number {i}.", max_tokens=20), range(N)))
    print(f"{len(results)}/{N} calls succeeded")
    with client.engine.connect() as c:
        ok = c.execute(select(func.count()).where(llm_calls.c.status == "ok")).scalar()
        err = c.execute(select(func.count()).where(llm_calls.c.status == "error")).scalar()
    print(f"llm_calls table: {ok} ok rows, {err} error rows")
