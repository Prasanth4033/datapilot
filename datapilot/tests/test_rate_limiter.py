import pytest

from core.rate_limiter import RateLimiter


class FakeClock:
    def __init__(self): self.t = 0.0
    def __call__(self): return self.t
    def sleep(self, s): self.t += s


def test_waits_when_rpm_is_reached():
    clock = FakeClock()
    rl = RateLimiter(rpm=3, tpm=10_000, clock=clock, sleep=clock.sleep)
    for _ in range(3):
        assert rl.acquire(10) == 0
    assert rl.acquire(10) == pytest.approx(60.0)   # 4th call waits for the window


def test_waits_when_tpm_is_reached():
    clock = FakeClock()
    rl = RateLimiter(rpm=100, tpm=1_000, clock=clock, sleep=clock.sleep)
    rl.acquire(800)
    assert rl.acquire(300) > 0                    # 800 + 300 > 1000, must wait


def test_rejects_call_bigger_than_tpm():
    with pytest.raises(ValueError):
        RateLimiter(rpm=10, tpm=100).acquire(500)
