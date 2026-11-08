"""Rate-limit playground: several users share one model, and requests get rejected (429)
or queued when the limits are exceeded. Also shows how each question becomes tokens.

Examples
  python -m scripts.playground                                   # alice and bob, interactive
  python -m scripts.playground --users alice,bob,carol --rpm 4 --user-rpm 2
  python -m scripts.playground --auto 4 --rpm 5 --user-rpm 3     # automatic demo, no typing
  python -m scripts.playground --mode wait                       # queue instead of reject
  python -m scripts.playground --tpm 300                         # watch the TOKEN limit trigger
  python -m scripts.playground --file questions.txt --rpm 3      # send many questions from a file

Interactive input:  alice: What does DPD mean?   |  status  |  exit
"""
from __future__ import annotations

import argparse
import time
from dataclasses import replace

from core import tokens
from core.config import Settings
from core.db import log_call
from core.llm_client import PRICES, LLMClient
from core.rate_limiter import RateLimiter

SAMPLE_QUESTIONS = ["What does DPD mean?", "Explain APR in one line.",
                    "How is a late fee calculated?", "What is a general ledger posting?"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--users", default="alice,bob", help="comma-separated names (default alice,bob)")
    p.add_argument("--rpm", type=int, default=5, help="shared provider limit: requests per minute")
    p.add_argument("--tpm", type=int, default=2000, help="shared provider limit: tokens per minute")
    p.add_argument("--user-rpm", type=int, default=3, help="each user's own requests per minute")
    p.add_argument("--mode", choices=["reject", "wait"], default="reject",
                   help="reject = return 429 at once; wait = queue until there is room")
    p.add_argument("--file", help="text file with one 'name: question' per line, sent in order")
    p.add_argument("--auto", type=int, default=0, help="run N automatic rounds (every user asks once per round)")
    p.add_argument("--max-tokens", type=int, default=100, help="answer budget reserved per call")
    p.add_argument("--hide-tokens", action="store_true", help="do not print the token breakdown")
    return p.parse_args()


def show_tokens(question: str, system: str, model: str, max_tokens: int) -> int:
    pieces = tokens.tokenize(question, model)
    b = tokens.chat_breakdown(system, question, model)
    print(f"   Tokenizer: {tokens.tokenizer_name(model)}")
    print(f"   Question split into {len(pieces)} tokens:")
    print("     " + " ".join(f"[{p!r}{'' if i is None else ':' + str(i)}]" for i, p in pieces))
    print(f"   Input tokens = system prompt {b['system']} + question {b['question']} "
          f"+ chat overhead {b['overhead']} = {b['total']}")
    print(f"   Reserved for the limiter = input {b['total']} + answer budget {max_tokens} = {b['total'] + max_tokens}")
    return b["total"] + max_tokens


def show_status(name: str, rl: RateLimiter) -> str:
    st = rl.status()
    return f"{name}: {st['requests_used']}/{st['rpm_limit']} req, {st['tokens_used']}/{st['tpm_limit']} tok"


def main():
    a = parse_args()
    users = [u.strip().lower() for u in a.users.split(",") if u.strip()]
    s = Settings()
    client = LLMClient(s, limiter=RateLimiter(10**6, 10**9))      # gating happens here, not inside the client
    provider = RateLimiter(a.rpm, a.tpm)                           # the shared limit, like your API account
    per_user = {u: RateLimiter(a.user_rpm, a.tpm) for u in users}  # each user's own quota
    price_in, price_out = PRICES.get(s.model, (0.0, 0.0))

    print(f"Playground | users={', '.join(users)} | provider limit {a.rpm} req/min, {a.tpm} tok/min "
          f"| per user {a.user_rpm} req/min | mode={a.mode} | domain={client.domain.name}\n")

    def handle(user: str, question: str) -> None:
        print(f"{user} > {question}")
        needed = tokens.chat_breakdown(client.domain.system_prompt, question, s.model)["total"] + a.max_tokens
        if not a.hide_tokens:
            show_tokens(question, client.domain.system_prompt, s.model, a.max_tokens)
        while True:
            ok_u, why_u, retry_u = per_user[user].check(needed)
            ok_p, why_p, retry_p = provider.check(needed)
            if ok_u and ok_p:
                break
            who, why, retry = (f"{user}'s quota", why_u, retry_u) if not ok_u else ("provider", why_p, retry_p)
            if a.mode == "reject":
                print(f"   429 REJECTED by {who}: {why}. Retry after {retry} s")
                log_call(client.engine, provider=s.provider, model=s.model, domain=client.domain.name,
                         prompt_tokens=0, completion_tokens=0, latency_ms=0, attempts=0, waited_ms=0,
                         status="rejected", error=f"{user}: {who}: {why}", cost_usd=0)
                print(f"   [{show_status(user, per_user[user])} | {show_status('provider', provider)}]\n")
                return
            print(f"   QUEUED by {who}: {why}. Waiting {retry} s...")
            time.sleep(max(retry, 0.1))
        per_user[user].acquire(needed)
        provider.acquire(needed)
        r = client.ask(question, max_tokens=a.max_tokens)
        per_user[user].correct(needed, r.prompt_tokens + r.completion_tokens)
        provider.correct(needed, r.prompt_tokens + r.completion_tokens)
        cost = (r.prompt_tokens * price_in + r.completion_tokens * price_out) / 1e6
        print(f"   200 OK  DataPilot: {r.text}")
        print(f"   Billed: {r.prompt_tokens} input + {r.completion_tokens} output tokens  ->  "
              f"{r.prompt_tokens} x ${price_in}/1M + {r.completion_tokens} x ${price_out}/1M = ${cost:.8f}")
        print(f"   [{show_status(user, per_user[user])} | {show_status('provider', provider)}]\n")

    if a.file:
        from pathlib import Path
        for n, line in enumerate(Path(a.file).read_text(encoding="utf-8").splitlines(), 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            name, sep, question = line.partition(":")
            name = name.strip().lower()
            if not sep or name not in per_user or not question.strip():
                print(f"   Line {n} skipped (use  name: question  with one of: {', '.join(users)}): {line}\n")
                continue
            handle(name, question.strip())
        return

    if a.auto:
        for rnd in range(a.auto):
            for i, u in enumerate(users):
                handle(u, SAMPLE_QUESTIONS[(rnd + i) % len(SAMPLE_QUESTIONS)])
        return

    print("Type  name: question   (e.g.  bob: What is APR?)   |  status  |  exit\n")
    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line.lower() in {"exit", "quit"}:
            break
        if line.lower() == "status":
            for u in users:
                print("   " + show_status(u, per_user[u]))
            print("   " + show_status("provider", provider) + "\n")
            continue
        name, sep, question = line.partition(":")
        name = name.strip().lower()
        if not sep or name not in per_user or not question.strip():
            print(f"   Use  name: question   with one of: {', '.join(users)}\n")
            continue
        handle(name, question.strip())


if __name__ == "__main__":
    main()
