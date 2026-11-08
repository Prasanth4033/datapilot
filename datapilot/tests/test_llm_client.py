from sqlalchemy import select

from core.config import Settings, load_domain
from core.db import get_engine, llm_calls
from core.llm_client import FakeProvider, LLMClient, RetryableError
from core.rate_limiter import RateLimiter


def make_client(tmp_path, provider, domain="lending", retries=4):
    s = Settings(provider="fake", domain=domain, max_retries=retries,
                 database_url=f"sqlite:///{tmp_path / 't.db'}")
    return LLMClient(s, provider=provider, limiter=RateLimiter(1000, 1_000_000),
                     engine=get_engine(s.database_url), wait=lambda rs: 0)  # no real waiting in tests


def rows(client):
    with client.engine.connect() as c:
        return c.execute(select(llm_calls)).mappings().all()


def test_call_is_logged(tmp_path):
    client = make_client(tmp_path, FakeProvider())
    r = client.ask("hello")
    assert r.text == "echo: hello"
    [row] = rows(client)
    assert row["status"] == "ok" and row["domain"] == "lending" and row["prompt_tokens"] > 0


def test_retries_then_succeeds(tmp_path):
    client = make_client(tmp_path, FakeProvider(fail_first=2))
    assert client.ask("hi").attempts == 3


def test_gives_up_and_logs_error(tmp_path):
    client = make_client(tmp_path, FakeProvider(fail_first=99), retries=3)
    try:
        client.ask("hi")
        raise AssertionError("should have failed")
    except RuntimeError as e:
        assert "3 attempt" in str(e)
    assert rows(client)[-1]["status"] == "error"


def test_domain_switch_needs_no_code_change():
    assert load_domain("lending").name == "lending"
    assert "retail" in load_domain("retail").system_prompt


class Boom(FakeProvider):
    def complete(self, *a):
        raise ValueError("bad request")  # not retryable


def test_non_retryable_error_fails_fast(tmp_path):
    client = make_client(tmp_path, Boom())
    try:
        client.ask("hi")
    except RuntimeError as e:
        assert "1 attempt" in str(e)
