"""Structured result in, controller English out.

The narrator receives no tool definitions and no dataset - only what the
executor produced. It therefore cannot call a tool, cannot look anything up,
and has nothing to compute from. Its failure modes reduce to phrasing, which
the grounding validator then checks.
"""
from __future__ import annotations

import json

from backend.client import Usage, complete
from backend.executor.executor import ExecutionContext
from backend.planner.prompts import ABSTAIN, narrator_prompt

# gpt-oss spends completion tokens on hidden reasoning before writing a word.
# A station-closure result is a 13-row table, and at 900 tokens the reasoning
# consumed the whole budget and the answer came back empty. Narration is
# formatting, not deliberation - so the effort goes down and the ceiling up.
NARRATE_MAX_TOKENS = 1600





def narrate(question: str, ctx: ExecutionContext, usage: Usage | None = None,
            history: list | None = None) -> str:
    payload = json.dumps(ctx.for_model(), default=str)
    messages = [{"role": "system", "content": narrator_prompt()}]
    if history:
        messages.append({"role": "user",
                         "content": "Conversation so far:\n" + "\n".join(history[-4:])})
    messages.append({"role": "user",
                     "content": f"QUESTION: {question}\n\nRESULT:\n{payload}"})
    return complete(messages, role="narrator", max_tokens=NARRATE_MAX_TOKENS,
                    reasoning_effort="low", usage=usage)


def renarrate(question: str, ctx: ExecutionContext, correction: str,
              usage: Usage | None = None) -> str:
    """Second attempt after the grounding validator rejected the first."""
    payload = json.dumps(ctx.for_model(), default=str)
    messages = [
        {"role": "system", "content": narrator_prompt()},
        {"role": "user", "content": f"QUESTION: {question}\n\nRESULT:\n{payload}"},
        {"role": "user", "content": correction},
    ]
    return complete(messages, role="narrator", max_tokens=NARRATE_MAX_TOKENS,
                    reasoning_effort="low", usage=usage)


def narrate_stream(question: str, ctx: ExecutionContext, usage: Usage | None = None,
                   history: list | None = None):
    """Same prompt as `narrate`, yielded token by token."""
    from backend.client import complete_stream

    payload = json.dumps(ctx.for_model(), default=str)
    messages = [{"role": "system", "content": narrator_prompt()}]
    if history:
        messages.append({"role": "user",
                         "content": "Conversation so far:\n" + "\n".join(history[-4:])})
    messages.append({"role": "user",
                     "content": f"QUESTION: {question}\n\nRESULT:\n{payload}"})
    yield from complete_stream(messages, role="narrator",
                               max_tokens=NARRATE_MAX_TOKENS,
                               reasoning_effort="low", usage=usage)
