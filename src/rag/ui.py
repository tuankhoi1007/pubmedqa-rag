"""Gradio page for the local research demo. Not medical advice.

The ask button calls :func:`rag.ui.answer_question`, which validates with
:class:`rag.api.AskRequest` and then :func:`rag.api.ask_service`. Styling uses
a Gradio theme plus the ``css`` string on ``gr.Blocks``. Gradio 6 applies that
theme and CSS when the app is mounted.
"""

from __future__ import annotations

import html
import re
import warnings
from typing import TYPE_CHECKING

import gradio as gr

if TYPE_CHECKING:
    from fastapi import FastAPI

VALIDATION_MESSAGE = "Please enter a question (1–1000 characters)."
FONT_STACK = "Figtree, 'Noto Sans', system-ui, sans-serif"
# First three questions of PubMedQA pqa_labeled train, in dataset order.
EXAMPLE_QUESTIONS = (
    "Do mitochondria play a role in remodelling lace plant leaves during programmed cell death?",
    "Landolt C and snellen e acuity: differences in strabismus amblyopia?",
    "Syncope during bathing in infants, a pediatric form of water-induced urticaria?",
)
_SERVICE_MESSAGES = {
    "not_ready": "The system is still starting. Please wait a moment and try again.",
    "queue_full": "The server is busy. Please try again in a few seconds.",
    "invalid_question": "That question cannot be answered. Please rephrase it as a yes/no question.",
    "rate_limit": "The service is receiving too many requests. Please wait a moment and try again.",
    "timeout": "The model took too long to respond. Please try again.",
    "upstream": "The answering service failed. Please try again.",
    "internal": "Something went wrong while answering. Please try again.",
}
_GENERIC_MESSAGE = "The request could not be completed. Please try again."
_PUBMED_ID = re.compile(r"\d+")


def _icon_check() -> str:
    return (
        '<svg class="label-icon" viewBox="0 0 24 24" width="28" height="28" '
        'aria-hidden="true"><path fill="currentColor" '
        'd="M9.2 16.6 4.8 12.2l1.4-1.4 3 3 8.2-8.2 1.4 1.4z"/></svg>'
    )


def _icon_cross() -> str:
    return (
        '<svg class="label-icon" viewBox="0 0 24 24" width="28" height="28" '
        'aria-hidden="true"><path fill="currentColor" '
        'd="M6.4 6.4 12 12l5.6-5.6 1.4 1.4L13.4 13.4l5.6 5.6-1.4 1.4L12 14.8 '
        'l-5.6 5.6-1.4-1.4 5.6-5.6-5.6-5.6z"/></svg>'
    )


def _icon_maybe() -> str:
    return (
        '<svg class="label-icon" viewBox="0 0 24 24" width="28" height="28" '
        'aria-hidden="true"><path fill="currentColor" '
        'd="M12 4a8 8 0 1 0 0 16 8 8 0 0 0 0-16zm-1.2 11.2h2.4v2.2h-2.4v-2.2zm.2-1.4 '
        'l-.3-6.2h2.6l-.3 6.2h-2z"/></svg>'
    )


def _icon_scope() -> str:
    return (
        '<svg class="label-icon" viewBox="0 0 24 24" width="28" height="28" '
        'aria-hidden="true"><path fill="currentColor" d="M4 11h16v2H4z"/></svg>'
    )


def _icon_fail() -> str:
    return (
        '<svg class="label-icon" viewBox="0 0 24 24" width="28" height="28" '
        'aria-hidden="true"><path fill="currentColor" '
        'd="M12 3.5 2.8 19.5h18.4L12 3.5zm-1.1 6.2h2.2l-.3 5.2h-1.6l-.3-5.2zm0 '
        '6.6h2.2V18h-2.2v-1.7z"/></svg>'
    )


_LABELS = {
    "yes": ("YES", "answer-yes", _icon_check()),
    "no": ("NO", "answer-no", _icon_cross()),
    "maybe": ("MAYBE", "answer-maybe", _icon_maybe()),
    "parse_fail": ("PARSE FAIL", "answer-parse-fail", _icon_fail()),
    "out_of_scope": ("OUT OF SCOPE", "answer-out-of-scope", _icon_scope()),
}
OUT_OF_SCOPE_MESSAGE = (
    "This question doesn't match the PubMedQA corpus. "
    "Try a biomedical yes/no question."
)

CSS = f"""
.gradio-container {{
  font-family: {FONT_STACK} !important;
  max-width: 880px !important;
  margin: 0 auto !important;
  padding: 24px 16px 48px !important;
}}
.page-header h1,
.answer-label,
.section-title {{
  font-family: {FONT_STACK};
}}
.page-header h1 {{
  margin: 0 0 8px;
  color: #164E63;
  font-size: 2rem;
  line-height: 1.2;
  font-weight: 700;
}}
.eyebrow {{
  margin: 0 0 8px;
  color: #475569;
  font-size: 0.85rem;
  font-weight: 600;
  letter-spacing: 0.04em;
  text-transform: uppercase;
}}
.lede {{
  margin: 0 0 12px;
  color: #164E63;
  font-size: 1.05rem;
  line-height: 1.5;
}}
.disclaimer {{
  margin: 0;
  padding: 10px 12px;
  border-left: 4px solid #0891B2;
  background: #FFFFFF;
  color: #164E63;
}}
.section-title {{
  margin: 8px 0;
  color: #164E63;
  font-size: 1.15rem;
}}
.answer-label {{
  display: inline-flex;
  align-items: center;
  gap: 10px;
  min-height: 48px;
  margin: 8px 0;
  padding: 12px 18px;
  border-radius: 12px;
  font-size: 1.75rem;
  font-weight: 700;
  letter-spacing: 0.04em;
  line-height: 1;
}}
.label-icon {{
  flex: 0 0 auto;
}}
.answer-yes {{ background: #059669; color: #000000; }}
.answer-no {{ background: #FEE2E2; color: #7F1D1D; }}
.answer-maybe {{ background: #FEF3C7; color: #78350F; }}
.answer-parse-fail {{ background: #E2E8F0; color: #0F172A; }}
.answer-out-of-scope {{ background: #E2E8F0; color: #1E293B; }}
.scope-note {{
  margin: 8px 0 0;
  color: #164E63;
  font-size: 1rem;
  font-weight: 500;
  letter-spacing: 0;
  line-height: 1.45;
}}
.latency {{
  margin: 4px 0 12px;
  color: #475569;
}}
.source-card {{
  margin: 0 0 12px;
  padding: 16px;
  border: 1px solid #A5F3FC;
  border-radius: 12px;
  background: #FFFFFF;
  color: #164E63;
}}
.source-card h3 {{
  margin: 0 0 8px;
  font-size: 1rem;
}}
.source-card a {{
  color: #155E75;
  font-weight: 600;
}}
.source-card a:hover {{
  color: #164E63;
}}
.source-card p {{
  margin: 0;
  white-space: pre-wrap;
}}
.score {{
  color: #475569;
  font-weight: 500;
}}
.notice-error {{
  margin: 8px 0;
  padding: 12px 16px;
  border: 1px solid #DC2626;
  border-radius: 12px;
  background: #FFFFFF;
  color: #DC2626;
}}
.empty-sources {{
  color: #475569;
}}
button {{
  cursor: pointer;
  transition: background-color 200ms ease, border-color 200ms ease, color 200ms ease;
}}
button:focus-visible,
textarea:focus-visible,
a:focus-visible {{
  outline: 2px solid #0891B2;
  outline-offset: 2px;
}}
.example-button {{
  height: auto !important;
  text-align: left !important;
  white-space: normal !important;
}}
@media (max-width: 720px) {{
  .gradio-container {{
    padding: 16px 12px 32px !important;
  }}
  .page-header h1 {{
    font-size: 1.5rem;
  }}
  .answer-label {{
    font-size: 1.35rem;
  }}
  .example-row {{
    flex-direction: column !important;
  }}
}}
@media (prefers-reduced-motion: reduce) {{
  *, *::before, *::after {{
    transition: none !important;
    animation: none !important;
  }}
}}
"""


def build_theme() -> gr.themes.Base:
    """Light clinical theme: cyan primary, black button text, Figtree then Noto Sans."""
    theme = gr.themes.Base(
        primary_hue="cyan",
        secondary_hue="emerald",
        neutral_hue="slate",
        text_size="md",
        spacing_size="md",
        radius_size="md",
        font=[
            gr.themes.GoogleFont("Figtree", weights=(400, 600, 700)),
            gr.themes.GoogleFont("Noto Sans", weights=(400, 500, 700)),
            "system-ui",
            "sans-serif",
        ],
        font_mono=["ui-monospace", "Consolas", "monospace"],
    )
    return theme.set(
        body_background_fill="#ECFEFF",
        body_background_fill_dark="#ECFEFF",
        body_text_color="#164E63",
        body_text_color_dark="#164E63",
        body_text_color_subdued="#475569",
        body_text_color_subdued_dark="#475569",
        background_fill_primary="#ECFEFF",
        background_fill_primary_dark="#ECFEFF",
        background_fill_secondary="#FFFFFF",
        background_fill_secondary_dark="#FFFFFF",
        block_background_fill="#FFFFFF",
        block_background_fill_dark="#FFFFFF",
        block_border_color="#A5F3FC",
        block_border_color_dark="#A5F3FC",
        border_color_primary="#A5F3FC",
        border_color_primary_dark="#A5F3FC",
        input_background_fill="#FFFFFF",
        input_background_fill_dark="#FFFFFF",
        input_border_color_focus="#0891B2",
        input_border_color_focus_dark="#0891B2",
        link_text_color="#155E75",
        link_text_color_dark="#155E75",
        link_text_color_hover="#164E63",
        link_text_color_hover_dark="#164E63",
        link_text_color_active="#164E63",
        link_text_color_active_dark="#164E63",
        link_text_color_visited="#155E75",
        link_text_color_visited_dark="#155E75",
        button_primary_background_fill="#0891B2",
        button_primary_background_fill_dark="#0891B2",
        button_primary_background_fill_hover="#22D3EE",
        button_primary_background_fill_hover_dark="#22D3EE",
        button_primary_text_color="#000000",
        button_primary_text_color_dark="#000000",
        button_primary_text_color_hover="#0F172A",
        button_primary_text_color_hover_dark="#0F172A",
        button_primary_border_color="#0891B2",
        button_primary_border_color_dark="#0891B2",
        button_primary_border_color_hover="#22D3EE",
        button_primary_border_color_hover_dark="#22D3EE",
        button_secondary_background_fill="#FFFFFF",
        button_secondary_background_fill_dark="#FFFFFF",
        button_secondary_text_color="#164E63",
        button_secondary_text_color_dark="#164E63",
        button_secondary_border_color="#A5F3FC",
        button_secondary_border_color_dark="#A5F3FC",
    )


def answer_question(app: object, text: str | None) -> tuple[str, str, str, str, str]:
    """Validate, then ask. Errors become a notice and never leave this function."""
    from pydantic import ValidationError

    from rag.api import AskRequest, AskServiceError, ask_service

    try:
        parsed = AskRequest(question=text if isinstance(text, str) else "")
    except ValidationError:
        return _error_result(VALIDATION_MESSAGE)
    try:
        response = ask_service(app.state, parsed.question, True)
    except AskServiceError as exc:
        return _error_result(_SERVICE_MESSAGES.get(exc.code, _GENERIC_MESSAGE))
    except Exception:
        return _error_result(_SERVICE_MESSAGES["internal"])
    raw = response.raw_response if isinstance(response.raw_response, str) else ""
    return (
        "",
        _label_html(response.label),
        _latency_html(response.latency_seconds),
        _sources_html(response.sources),
        raw,
    )


def build_ui(app: FastAPI) -> gr.Blocks:
    """Build the page. The ask event runs one call at a time."""
    theme = build_theme()
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="The parameters have been moved from the Blocks constructor.*",
            category=UserWarning,
        )
        demo = gr.Blocks(
            theme=theme,
            css=CSS,
            analytics_enabled=False,
            title="PubMedQA RAG",
            fill_width=False,
        )
    with demo:
        gr.HTML(
            "<header class='page-header'>"
            "<p class='eyebrow'>PubMedQA research demo</p>"
            "<h1>Biomedical yes/no answers</h1>"
            "<p class='lede'>This system answers yes, no, or maybe for biomedical "
            "questions using 1,000 PubMedQA abstracts (pqa_labeled).</p>"
            "<p class='disclaimer'>Not medical advice. Please ask a yes/no question.</p>"
            "</header>"
        )
        question = gr.Textbox(
            label="Question",
            lines=4,
            placeholder="Ask a yes/no question",
        )
        ask = gr.Button("Ask", variant="primary")
        with gr.Row(elem_classes=["example-row"]):
            for example in EXAMPLE_QUESTIONS:
                gr.Button(example, size="sm", elem_classes=["example-button"]).click(
                    _keep(example),
                    outputs=question,
                    concurrency_limit=1,
                )
        notice = gr.HTML()
        label = gr.HTML()
        latency = gr.HTML()
        sources = gr.HTML()
        with gr.Accordion("Debug", open=False):
            raw = gr.Textbox(label="raw_response", lines=8, interactive=False)
        ask.click(
            lambda text: answer_question(app, text),
            inputs=question,
            outputs=[notice, label, latency, sources, raw],
            concurrency_limit=1,
        )
    demo.pubmed_theme = theme
    demo.pubmed_css = CSS
    demo.queue(default_concurrency_limit=1)
    return demo


def _keep(text: str):
    def fill() -> str:
        return text

    return fill


def _error_result(message: str) -> tuple[str, str, str, str, str]:
    return (
        f"<div class='notice-error' role='alert'>{html.escape(message)}</div>",
        "",
        "",
        "",
        "",
    )


def _label_html(label: str) -> str:
    word, css_class, icon = _LABELS.get(
        label,
        (html.escape(label), "answer-parse-fail", _icon_fail()),
    )
    note = ""
    if label == "out_of_scope":
        note = f"<p class='scope-note'>{html.escape(OUT_OF_SCOPE_MESSAGE)}</p>"
    return (
        "<div class='answer-label "
        + css_class
        + "' role='status'>"
        + icon
        + "<span>"
        + word
        + "</span></div>"
        + note
    )


def _latency_html(seconds: float) -> str:
    return f"<p class='latency'>Latency: {seconds:.3f} seconds</p>"


def _sources_html(sources: list) -> str:
    if not sources:
        return "<p class='empty-sources'>No cited sources.</p>"
    cards = ["<h2 class='section-title'>Cited sources</h2>"]
    for source in sources:
        pubid = str(source.pubid)
        safe_pubid = html.escape(pubid)
        if _PUBMED_ID.fullmatch(pubid):
            href = f"https://pubmed.ncbi.nlm.nih.gov/{pubid}/"
            pubid_html = (
                f"<a href='{href}' target='_blank' rel='noopener noreferrer'>"
                f"PubMed {safe_pubid}</a>"
            )
        else:
            pubid_html = f"PubMed {safe_pubid}"
        cards.append(
            "<article class='source-card'>"
            f"<h3>Rank {html.escape(str(source.rank))} · {pubid_html} "
            f"<span class='score'>score {_format_score(source.score)}</span></h3>"
            f"<p>{html.escape(source.text)}</p>"
            "</article>"
        )
    return "".join(cards)


def _format_score(score: float) -> str:
    return html.escape(f"{score:.6f}".rstrip("0").rstrip("."))
