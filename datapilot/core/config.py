"""Settings from .env and the active domain pack."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    provider: str = os.getenv("LLM_PROVIDER", "openai")
    model: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
    timeout_s: float = float(os.getenv("LLM_TIMEOUT_SECONDS", "30"))
    max_retries: int = int(os.getenv("LLM_MAX_RETRIES", "4"))
    rpm_limit: int = int(os.getenv("LLM_RPM_LIMIT", "60"))
    tpm_limit: int = int(os.getenv("LLM_TPM_LIMIT", "60000"))
    domain: str = os.getenv("DOMAIN", "lending")
    database_url: str = os.getenv("DATABASE_URL", f"sqlite:///{ROOT / 'datapilot.db'}")


@dataclass(frozen=True)
class DomainPack:
    name: str
    description: str
    system_prompt: str
    glossary: dict = field(default_factory=dict)


def load_domain(name: str | None = None) -> DomainPack:
    """Read domains/<name>/domain.yaml. The code never hard-codes a domain."""
    name = name or Settings().domain
    path = ROOT / "domains" / name / "domain.yaml"
    if not path.exists():
        available = sorted(p.name for p in (ROOT / "domains").iterdir() if p.is_dir())
        raise FileNotFoundError(f"No domain pack '{name}'. Available: {available}")
    data = yaml.safe_load(path.read_text()) or {}
    missing = {"name", "description", "system_prompt"} - data.keys()
    if missing:
        raise ValueError(f"{path} is missing keys: {sorted(missing)}")
    return DomainPack(
        name=data["name"],
        description=data["description"],
        system_prompt=data["system_prompt"].strip(),
        glossary=data.get("glossary", {}),
    )
