"""Usage: python -m scripts.ask "What does DPD mean?" """
import sys

from core.llm_client import LLMClient

if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "In one sentence, what is DataPilot for?"
    r = LLMClient().ask(question)
    print(r.text)
    print(f"\n[{r.prompt_tokens}+{r.completion_tokens} tokens, {r.latency_ms:.0f} ms, {r.attempts} attempt(s)]")
