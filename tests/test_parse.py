from rag.core import parse_label


def test_parse_yes_with_conclusion():
    raw = "Conclusion: Authors found an effect.\nAnswer: yes\nSources: 1, 2"
    assert parse_label(raw) == "yes"


def test_parse_case_insensitive():
    assert parse_label("Answer: NO") == "no"
    assert parse_label("Answer: Maybe") == "maybe"


def test_parse_fail_garbage():
    assert parse_label("I am not sure") == "parse_fail"
    assert parse_label("") == "parse_fail"
