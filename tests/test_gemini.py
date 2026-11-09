import pytest

from rag.gemini import (
    SERVING_MAX_ATTEMPTS,
    SERVING_TIMEOUT_MS,
    GeminiClient,
    GeminiRequestError,
    require_api_key,
)


def test_missing_api_key_names_the_variable_only():
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY is not set"):
        require_api_key({})
    with pytest.raises(RuntimeError, match="^GEMINI_API_KEY is not set$"):
        require_api_key({"OTHER_KEY": "present"})


def test_generate_sends_model_and_prompt_without_generation_config():
    class Models:
        def __init__(self):
            self.kwargs = None
            self.calls = 0

        def generate_content(self, **kwargs):
            self.calls += 1
            self.kwargs = kwargs
            return type("Response", (), {"text": "Answer: yes"})()

    class SDK:
        def __init__(self):
            self.models = Models()

    sdk = SDK()
    client = GeminiClient(api_key="local-test-value", sdk_client=sdk)
    result = client.generate("Passages:\nalpha", "gemini-3.5-flash-lite")
    assert result.text == "Answer: yes"
    assert sdk.models.calls == SERVING_MAX_ATTEMPTS == 1
    assert sdk.models.kwargs == {
        "model": "gemini-3.5-flash-lite",
        "contents": "Passages:\nalpha",
    }
    assert SERVING_TIMEOUT_MS == 30_000


def test_empty_sdk_text_is_an_empty_string():
    class Models:
        def generate_content(self, **kwargs):
            return type("Response", (), {"text": None})()

    class SDK:
        def __init__(self):
            self.models = Models()

    text = GeminiClient(api_key="local-test-value", sdk_client=SDK()).generate("p", "m").text
    assert text == ""


def test_sdk_timeout_is_a_request_error_and_is_not_retried():
    class Models:
        def __init__(self):
            self.calls = 0

        def generate_content(self, **kwargs):
            self.calls += 1
            raise TimeoutError("deadline exceeded")

    class SDK:
        def __init__(self):
            self.models = Models()

    sdk = SDK()
    client = GeminiClient(api_key="local-test-value", sdk_client=sdk)
    with pytest.raises(GeminiRequestError, match="^Gemini request failed$") as caught:
        client.generate("prompt", "gemini-3.5-flash-lite")
    assert isinstance(caught.value.__cause__, TimeoutError)
    assert caught.value.kind == "timeout"
    assert sdk.models.calls == 1
    assert "maybe" not in str(caught.value)


def _failed_request(exc):
    class Models:
        def __init__(self):
            self.calls = 0

        def generate_content(self, **kwargs):
            self.calls += 1
            raise exc

    class SDK:
        def __init__(self):
            self.models = Models()

    sdk = SDK()
    client = GeminiClient(api_key="local-test-value", sdk_client=sdk)
    with pytest.raises(GeminiRequestError, match="^Gemini request failed$") as caught:
        client.generate("prompt", "gemini-3.5-flash-lite")
    assert caught.value.__cause__ is exc
    assert sdk.models.calls == 1
    return caught.value


def test_timeout_kind_wins_over_a_429_in_the_message():
    import httpx
    from google.genai._gaos.lib.compat_errors import APITimeoutError

    assert _failed_request(TimeoutError("429 RESOURCE_EXHAUSTED")).kind == "timeout"
    assert _failed_request(httpx.TimeoutException("timed out")).kind == "timeout"
    request = httpx.Request("POST", "https://example.test/generate")
    assert _failed_request(APITimeoutError(request)).kind == "timeout"


def test_structured_429_is_rate_limit_and_other_codes_are_not():
    from google.genai.errors import APIError

    quota = APIError(429, {"status": "RESOURCE_EXHAUSTED", "message": "quota"})
    assert quota.code == 429
    assert "status_code" not in quota.__dict__
    assert _failed_request(quota).kind == "rate_limit"

    class StatusCode(Exception):
        def __init__(self):
            super().__init__("slow down")
            self.status_code = 429

    assert _failed_request(StatusCode()).kind == "rate_limit"

    other = APIError(500, {"message": "429 RESOURCE_EXHAUSTED"})
    assert other.code == 500
    assert _failed_request(other).kind == "other"


def test_message_fallback_runs_only_without_a_structured_code():
    assert _failed_request(RuntimeError("429 RESOURCE_EXHAUSTED")).kind == "rate_limit"
    assert _failed_request(RuntimeError("RESOURCE_EXHAUSTED")).kind == "rate_limit"

    class EmptyCode(Exception):
        code = None
        status_code = None

        def __str__(self):
            return "429"

    assert _failed_request(EmptyCode()).kind == "rate_limit"
    assert _failed_request(RuntimeError("14290")).kind == "other"
    assert _failed_request(RuntimeError("HTTP 429")).kind == "rate_limit"


def test_other_errors_stay_other():
    assert _failed_request(RuntimeError("backend unavailable")).kind == "other"
