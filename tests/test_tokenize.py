from rag.core import clean_and_tokenize


def test_lower_and_punct():
    assert clean_and_tokenize("Hello, World!") == ["hello", "world"]


def test_stopwords():
    stops = {"the", "a"}
    assert clean_and_tokenize("The a cat", stops) == ["cat"]
