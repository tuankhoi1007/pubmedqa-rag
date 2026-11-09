"""Prompt templates shipped inside the package.

``exp10`` is the notebook ``RULES + QUERY`` text with few-shot left off.
``{question}`` and ``{context}`` stay unsubstituted until a call fills them.
"""

from __future__ import annotations

import re
from importlib.resources import files

PROMPT_ID = "exp10"

_FILES = {
    PROMPT_ID: "exp10.txt",
}

_SOURCES_LINE = re.compile(r"(?im)^[ \t]*Sources:[ \t]*(.*)$")


def load_prompt(prompt_id: str = PROMPT_ID) -> str:
    """Return the unsubstituted template. The path does not depend on cwd."""
    filename = _FILES.get(prompt_id)
    if filename is None:
        known = ", ".join(sorted(_FILES))
        raise KeyError(f"unknown prompt_id {prompt_id!r}; known: {known}")
    return files("rag.prompts").joinpath(filename).read_text(encoding="utf-8")


def render_prompt(question: str, context: str, prompt_id: str = PROMPT_ID) -> str:
    """Insert this call's question and passages into the template."""
    template = load_prompt(prompt_id)
    if template.count("{question}") != 1 or template.count("{context}") != 1:
        raise RuntimeError(f"prompt {prompt_id} is missing its placeholders")
    return template.replace("{context}", context).replace("{question}", question)


def accepted_sources(raw: str, passage_count: int) -> list[int]:
    """Passage numbers from the first Sources line that exist in this context.

    Numbers are 1-based passage positions, not pubids. Out-of-range numbers
    are dropped. Nothing is added when the line is missing or invalid.
    """
    if not isinstance(raw, str) or passage_count < 1:
        return []
    match = _SOURCES_LINE.search(raw)
    if match is None:
        return []
    accepted: list[int] = []
    seen: set[int] = set()
    for token in re.findall(r"\d+", match.group(1)):
        number = int(token)
        if number < 1 or number > passage_count or number in seen:
            continue
        seen.add(number)
        accepted.append(number)
    return accepted
