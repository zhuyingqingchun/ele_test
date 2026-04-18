from __future__ import annotations

from .retriever import RetrievalResult


def build_stage2_context(condition_name: str, retrieval: RetrievalResult, event_summary: str) -> str:
    scenario_lines = "\n".join(
        f"{item.rank}. {item.label} | score={item.score:.4f} | {item.text}"
        for item in retrieval.scenario
    )
    family_lines = "\n".join(
        f"{item.rank}. {item.label} | score={item.score:.4f}"
        for item in retrieval.family
    )
    location_lines = "\n".join(
        f"{item.rank}. {item.label} | score={item.score:.4f}"
        for item in retrieval.location
    )
    mechanism_lines = "\n".join(
        f"{item.rank}. {item.label} | score={item.score:.4f} | {item.text}"
        for item in retrieval.mechanism
    )
    return (
        "Servo window diagnostic context\n"
        f"Condition: {condition_name}\n"
        f"Event evidence: {event_summary}\n\n"
        f"Top retrieved scenarios:\n{scenario_lines}\n\n"
        f"Top retrieved families:\n{family_lines}\n\n"
        f"Top retrieved locations:\n{location_lines}\n\n"
        f"Top retrieved mechanisms:\n{mechanism_lines}"
    )
