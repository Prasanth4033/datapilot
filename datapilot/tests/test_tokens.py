from core import tokens
from core.rate_limiter import RateLimiter


def test_tokenize_rebuilds_the_text():
    text = "What does DPD mean?"
    assert "".join(p for _, p in tokens.tokenize(text)) == text


def test_chat_breakdown_adds_overhead():
    b = tokens.chat_breakdown("You are helpful.", "Hi")
    assert b["total"] == b["system"] + b["question"] + b["overhead"]
    assert b["overhead"] == 9


def test_check_rejects_without_recording():
    rl = RateLimiter(rpm=1, tpm=1000)
    assert rl.check(10)[0] is True
    rl.acquire(10)
    ok, reason, retry = rl.check(10)
    assert not ok and "RPM" in reason and retry > 0
    assert rl.status()["requests_used"] == 1
