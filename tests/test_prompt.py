import ast
import json
from pathlib import Path

import pytest

from rag.prompts import PROMPT_ID, accepted_sources, load_prompt, render_prompt

NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "Thực_hành_clean.ipynb"
LACE_QUESTION = (
    "Do mitochondria play a role in remodelling lace plant leaves "
    "during programmed cell death?"
)


def _notebook_exp10() -> str:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    rules = query = None
    for cell in notebook["cells"]:
        source = "".join(cell.get("source", []))
        if "PROMPT = RULES" not in source:
            continue
        for node in ast.parse(source).body:
            if not isinstance(node, ast.Assign) or not isinstance(node.targets[0], ast.Name):
                continue
            name = node.targets[0].id
            if name in {"RULES", "QUERY"} and isinstance(node.value, ast.Constant):
                if name == "RULES":
                    rules = node.value.value
                else:
                    query = node.value.value
    assert isinstance(rules, str) and isinstance(query, str)
    return rules + query


def test_exp10_resource_matches_notebook_rules_and_query(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    template = load_prompt(PROMPT_ID)

    assert template == _notebook_exp10()
    assert template.count("{question}") == 1
    assert template.count("{context}") == 1
    assert "EXAMPLES:" not in template
    assert LACE_QUESTION not in template
    assert "lace plant" not in template


def test_render_inserts_the_call_question():
    first = render_prompt("Is metformin useful here?", "[1] (pubid 1) alpha")
    second = render_prompt("Does aspirin reduce stroke?", "[1] (pubid 2) beta")

    assert "Question: Is metformin useful here?" in first
    assert "[1] (pubid 1) alpha" in first
    assert "Does aspirin reduce stroke?" not in first
    assert "Question: Does aspirin reduce stroke?" in second
    assert "[1] (pubid 2) beta" in second
    assert "Is metformin useful here?" not in second
    assert LACE_QUESTION not in first
    assert LACE_QUESTION not in second


def test_unknown_prompt_id_is_rejected():
    with pytest.raises(KeyError, match="unknown prompt_id"):
        load_prompt("exp9")


def test_sources_keep_only_passage_numbers_in_range():
    raw = (
        "Conclusion: The pubid 21645374 is not a source.\n"
        "Answer: yes\n"
        "Sources: 2, 9, 21645374, 1, 2"
    )
    assert accepted_sources(raw, passage_count=2) == [2, 1]
    assert accepted_sources("Answer: yes\nSources: pubid 111", passage_count=2) == []
    assert accepted_sources("", passage_count=2) == []
