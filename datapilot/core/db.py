"""The llm_calls table: one row per LLM call (tokens, latency, status, cost)."""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, Float, Integer, MetaData, String, Table, Text, create_engine

metadata = MetaData()

llm_calls = Table(
    "llm_calls", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("provider", String(40)), Column("model", String(80)), Column("domain", String(40)),
    Column("prompt_tokens", Integer), Column("completion_tokens", Integer),
    Column("latency_ms", Float), Column("attempts", Integer), Column("waited_ms", Float),
    Column("status", String(20)),          # ok | error
    Column("error", Text), Column("cost_usd", Float),
)


def get_engine(url: str):
    engine = create_engine(url, future=True)
    metadata.create_all(engine)
    return engine


def log_call(engine, **row) -> None:
    row.setdefault("created_at", datetime.now(timezone.utc))
    with engine.begin() as conn:
        conn.execute(llm_calls.insert().values(**row))
