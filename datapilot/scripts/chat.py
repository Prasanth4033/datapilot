"""Interactive mode: type a question, get the answer plus the live rate-limit status.

Usage:  python -m scripts.chat               (limits from .env)
        python -m scripts.chat --rpm 3       (tiny limit, to watch the limiter make you wait)
Commands while running:  status  = show limits without asking   |   exit = quit
"""
import sys
from dataclasses import replace

from core.config import Settings
from core.llm_client import LLMClient


def show_status(st: dict) -> None:
    print(f"  Requests this minute: {st['requests_used']}/{st['rpm_limit']} used, {st['requests_left']} left")
    print(f"  Tokens this minute:   {st['tokens_used']}/{st['tpm_limit']} used, {st['tokens_left']} left")
    if st["requests_used"]:
        print(f"  Window resets in:     {st['resets_in_s']} s")


def main() -> None:
    s = Settings()
    if "--rpm" in sys.argv:
        s = replace(s, rpm_limit=int(sys.argv[sys.argv.index("--rpm") + 1]))
    client = LLMClient(s)
    print(f"DataPilot chat | provider={s.provider} model={s.model} domain={client.domain.name}")
    print(f"Limits: {s.rpm_limit} requests/min, {s.tpm_limit} tokens/min. Type 'status' or 'exit'.\n")

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            break
        if question.lower() == "status":
            show_status(client.limiter.status())
            print()
            continue

        before = client.limiter.status()
        if before["requests_left"] <= 0:
            print(f"  Rate limit reached: waiting about {before['resets_in_s']} s for the window to free up...")
        try:
            r = client.ask(question)
        except RuntimeError as e:
            print(f"  Error: {e}\n")
            continue
        print(f"DataPilot: {r.text}")
        print(f"  This call: {r.prompt_tokens}+{r.completion_tokens} tokens, {r.latency_ms:.0f} ms, {r.attempts} attempt(s)")
        show_status(client.limiter.status())
        print()


if __name__ == "__main__":
    main()
