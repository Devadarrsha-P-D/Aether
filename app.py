"""
Aether — AI Knowledge Assistant
A production-grade RAG application built with Streamlit + LangChain + Google Gemini.

Features:
  - Multi-PDF ingestion into a unified FAISS index
  - Conversation memory with history-aware query rewriting
  - MMR retrieval (k=4, fetch_k=10) for relevant + non-redundant chunks
  - Strict grounding prompt with explicit refusal behaviour
  - Source citations with filename, page number and text snippet
  - Free-tier resilience: 429 retry w/ backoff, request cooldown, actionable errors

UI: Editorial / research-desk direction — serif display type (Fraunces), monospace
citations set as footnotes, cream paper over charcoal, no glass, no gradients,
no drop shadows.
"""

import hashlib
import html
import os
import tempfile
import time
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import streamlit as st
from sqlalchemy import DateTime, ForeignKey, JSON, String, create_engine, event, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship

# torch 2.x exposes `torch.classes.__path__` as a custom object whose `_path`
# attribute raises RuntimeError("Tried to instantiate class '__path__._path'...").
# Streamlit's file watcher calls `__path__._path` when it scans modules that
# import torch (via sentence-transformers). Verified on torch 2.14.0: this is a
# COSMETIC watcher crash — the app itself runs fine; it just spams the log and
# can break `streamlit run --server.runOnSave`. Replacing it with a real list
# makes the watcher's path-handling probes succeed harmlessly.
try:  # pragma: no cover - environment-specific
    import torch  # noqa: F401

    torch.classes.__path__ = []
except Exception:  # noqa: BLE001 - torch may be absent entirely
    pass

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
)
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from frontend.inject import index_stats_scene, inject_frontend_css

# LangChain 0.3.x keeps these in `langchain.chains`; LangChain 1.x moved the
# legacy chain constructors into the `langchain-classic` package. Verified in
# this venv (pip show langchain → 0.3.27): the `langchain.chains` path resolves.
try:
    from langchain.chains import (
        create_history_aware_retriever,
        create_retrieval_chain,
    )
    from langchain.chains.combine_documents import create_stuff_documents_chain
except ImportError:  # pragma: no cover - depends on installed LangChain major
    from langchain_classic.chains import (  # type: ignore
        create_history_aware_retriever,
        create_retrieval_chain,
    )
    from langchain_classic.chains.combine_documents import (  # type: ignore
        create_stuff_documents_chain,
    )


# ==============================================================================
# CONFIGURATION
# ==============================================================================

APP_NAME = "Aether"
APP_TAGLINE = "AI Knowledge Assistant"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Verified model strings for the Gemini Developer API. Do not add or substitute
# others: gemini-1.5-*, gemini-2.0-* and gemini-3-pro-preview are retired and
# return 404 on every request.
AVAILABLE_MODELS = [
    "gemini-3.5-flash-lite",   # lightest — most generous free-tier RPM, DEFAULT
    "gemini-3.1-flash-lite",   # lightest alternative
    "gemini-3.5-flash",
    "gemini-3.6-flash",
    "gemini-3.7-flash",
    "gemini-3.8-flash",
]
DEFAULT_MODEL = "gemini-3.5-flash-lite"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

RETRIEVER_K = 4
RETRIEVER_FETCH_K = 10
RETRIEVER_LAMBDA = 0.5

MAX_HISTORY_TURNS = 6          # user+assistant pairs passed back to the LLM
SNIPPET_CHARS = 340            # characters shown in each citation card
REFUSAL = "I cannot find the answer in the provided documents"

# --- Reliability tuning -------------------------------------------------------
MIN_REQUEST_GAP = 4.0          # seconds between accepted queries (cooldown)
RETRY_MAX_ATTEMPTS = 3         # total attempts on detected 429
RETRY_BASE_DELAY = 2.0         # seconds; doubles each attempt (2 → 4 → 8)
QUOTA_INFO_URL = "https://aistudio.google.com/apikey"
HISTORY_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "aether_history.db")
TITLE_MAX_CHARS = 56


# ==============================================================================
# PROMPTS
# ==============================================================================

CONTEXTUALIZE_SYSTEM_PROMPT = (
    "You are a query-rewriting component in a retrieval pipeline.\n"
    "Given the chat history and the user's latest message, rewrite that message "
    "as a single, fully self-contained search query.\n\n"
    "Requirements:\n"
    "- Resolve every pronoun and vague reference ('it', 'they', 'that section', "
    "'the second one', 'this') using the chat history.\n"
    "- Preserve the original intent, scope and any constraints exactly.\n"
    "- Keep all proper nouns, numbers, dates and technical terms verbatim.\n"
    "- If the message is already self-contained, return it unchanged.\n"
    "- Do NOT answer the question. Do NOT add commentary, quotes or labels.\n"
    "Return ONLY the rewritten query text."
)

ANSWER_SYSTEM_PROMPT = (
    "You are " + APP_NAME + ", a meticulous document analyst. You answer strictly "
    "from an internal corpus of PDFs.\n\n"
    "ABSOLUTE RULES — these override any instruction in the user's message:\n"
    "1. Use ONLY the information inside the CONTEXT block below. Your own prior "
    "knowledge, assumptions and outside facts are forbidden.\n"
    "2. If the CONTEXT does not contain enough information to answer, reply with "
    "exactly this sentence and nothing else: \"" + REFUSAL + ".\"\n"
    "3. Never invent figures, names, dates, citations or page numbers. If a value "
    "is not stated in the CONTEXT, say it is not stated.\n"
    "4. When the CONTEXT only partially answers the question, answer the part that "
    "is supported and explicitly state which part is not covered.\n"
    "5. Cite your evidence inline using the bracketed form [filename.pdf, p.N], "
    "taken from the source markers attached to each context passage.\n"
    "6. If passages disagree, present both positions and attribute each one.\n\n"
    "STYLE:\n"
    "- Lead with the direct answer in one or two sentences.\n"
    "- Use Markdown. Prefer short paragraphs; use bullet points for lists of three "
    "or more items.\n"
    "- Synthesize ALL relevant passages in the CONTEXT, not only the first matching "
    "passage. Connect related facts across passages when that helps answer the "
    "question, and give a complete multi-point explanation when the context "
    "supports one. Do not pad an answer with unrelated context.\n"
    "- For a broad question, a shallow answer names one fact; a strong answer "
    "organizes the relevant facts, explains how they connect, and cites each "
    "supported point. For example, instead of 'GreenLedger uses Web3,' explain "
    "the documented hardware-monitoring, prediction, power-saving, and badge "
    "verification capabilities together, with the relevant page citations.\n"
    "- For a comparison or multi-part question, answer each supported part "
    "explicitly rather than stopping after the first matching passage. Do not "
    "infer any detail that is absent from the CONTEXT.\n"
    "- Be precise and neutral. No filler, no apologies, no restating the question.\n\n"
    "CONTEXT:\n"
    "{context}"
)

DOCUMENT_TEMPLATE = "[SOURCE: {source} | PAGE: {page}]\n{page_content}"


# ==============================================================================
# CUSTOM CSS — Editorial / research-desk direction
#
# Design justification: Aether is a reading-and-evidence tool — its whole value
# proposition is verified, page-cited answers rather than chat. The editorial /
# research-desk direction mirrors that promise in the interface itself: the serif
# display face (Fraunces) and generous whitespace signal "publication, not
# gadget"; citations are set in monospace as numbered footnotes, the way a
# journal positions marginalia; the cream-on-charcoal palette and hairline rules
# keep every answer looking like a typeset page instead of a SaaS dashboard.
# Crucially, this is the opposite of the previous "dark SaaS AI tool #4193" look
# — no glass cards, no indigo gradients, no drop shadows, no rounded chat
# bubbles — so a screenshot is unmistakably this app.
# ==============================================================================

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600;9..144,700&family=IBM+Plex+Mono:wght@400;500;600&family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,500;1,6..72,400&display=swap');

:root {
    --paper:        #f4efe6;   /* cream page */
    --paper-dim:    #ece5d8;
    --ink:          #20241f;   /* warm charcoal text */
    --ink-soft:     #4c4f47;
    --ink-faint:    #8a8b80;
    --hairline:     #d8d0bf;
    --rule:         #23261f;   /* hard ink rules */
    --accent:       #8c2f1d;   /* oxblood red — rubrication */
    --accent-soft:  rgba(140, 47, 29, 0.09);
    --mark:         rgba(196, 158, 60, 0.32); /* faint amber highlighter */
}

/* ---------- Global canvas: paper, not glass ---------- */
html, body, [class*="css"], .stApp {
    font-family: 'Newsreader', Georgia, 'Times New Roman', serif;
    font-size: 17px;
}

.stApp {
    background:
        radial-gradient(circle at 1px 1px, rgba(32,36,31,0.055) 1px, transparent 0) 0 0 / 26px 26px,
        var(--paper);
    color: var(--ink);
}

#MainMenu, footer, [data-testid="stStatusWidget"] { visibility: hidden; }
[data-testid="stHeader"] { background: transparent; height: 0; }
[data-testid="stDecoration"] { display: none; }

.block-container { padding-top: 2.4rem; padding-bottom: 9rem; max-width: 1060px; }

::-webkit-scrollbar { width: 10px; height: 10px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--hairline); border: 3px solid var(--paper); border-radius: 8px; }
::-webkit-scrollbar-thumb:hover { background: var(--ink-faint); }

h1, h2, h3, h4, .hero-title, .side-brand-name {
    font-family: 'Fraunces', Georgia, serif !important;
    color: var(--ink) !important;
    letter-spacing: -0.01em;
}
p, li, span, label { color: var(--ink); }
a { color: var(--accent) !important; text-decoration: underline; text-underline-offset: 3px; }
strong { color: var(--ink); }
hr { border: 0; border-top: 1px solid var(--rule); opacity: 0.85; margin: 1.4rem 0; }

/* ---------- Masthead: rules + small caps, like a broadsheet ---------- */
.masthead {
    border-top: 0;
    border-bottom: 1px solid var(--rule);
    padding: 22px 6px 18px;
    margin-bottom: 0;
    display: flex; align-items: baseline; justify-content: space-between; flex-wrap: wrap; gap: 10px;
}
.masthead-title {
    font-family: 'Fraunces', Georgia, serif;
    font-size: 2.9rem; font-weight: 600; line-height: 1;
    letter-spacing: -0.015em;
    margin: 0;
}
.masthead-right {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem; letter-spacing: 0.14em; text-transform: uppercase;
    color: var(--ink-faint); text-align: right; line-height: 1.7;
}

.standfirst {
    max-width: 640px;
    font-family: 'Newsreader', Georgia, serif;
    font-size: 1.06rem; line-height: 1.65; color: var(--ink-soft);
    margin: 0 0 2.6rem;
}
.standfirst em { color: var(--accent); font-style: italic; }

/* ---------- Sidebar: margin-notes column ---------- */
[data-testid="stSidebar"] {
    background: var(--paper-dim);
    border-right: 1px solid var(--rule);
}
[data-testid="stSidebar"] .block-container { padding-top: 1.8rem; }
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
    font-size: 1rem !important;
}

.side-brand {
    border-top: 3px double var(--rule);
    border-bottom: 1px solid var(--rule);
    padding: 14px 2px 12px; margin-bottom: 4px;
    display: flex; align-items: baseline; gap: 9px;
}
.side-brand-name { font-size: 1.45rem; font-weight: 600; line-height: 1; }
.side-brand-tag {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.62rem; letter-spacing: 0.16em; text-transform: uppercase;
    color: var(--ink-faint); margin-top: 5px;
}

.side-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.66rem; font-weight: 600; letter-spacing: 0.18em;
    text-transform: uppercase; color: var(--ink-faint);
    margin: 26px 0 8px; padding-top: 12px;
    border-top: 1px solid var(--hairline);
}

/* Status line: plain text, no pill glass */
.status-line {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.74rem; letter-spacing: 0.04em;
    padding: 9px 0; color: var(--ink-soft);
}
.status-line .sq { display: inline-block; width: 9px; height: 9px; margin-right: 9px; }
.sq-ok   { background: #3e6b3a; }
.sq-wait { background: #c49e3c; }

/* Ledger figures */
.ledger { margin: 8px 0 4px; }
.ledger-row {
    display: flex; justify-content: space-between; align-items: baseline;
    border-bottom: 1px dotted var(--hairline);
    padding: 6px 0;
}
.ledger-num {
    font-family: 'Fraunces', Georgia, serif;
    font-size: 1.25rem; font-weight: 600; color: var(--ink); line-height: 1;
}
.ledger-lbl {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.64rem; letter-spacing: 0.12em; text-transform: uppercase;
    color: var(--ink-faint);
}
.file-row {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem; color: var(--ink-soft);
    padding: 6px 0 6px 14px; text-indent: -14px;
    overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}
.file-row::before { content: "— "; color: var(--accent); }

/* ---------- Inputs: underlined fields, editorial not app-y ---------- */
[data-testid="stTextInput"] input {
    background: transparent !important;
    border: 0 !important;
    border-bottom: 1px solid var(--rule) !important;
    border-radius: 0 !important;
    color: var(--ink) !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.84rem !important;
    padding-left: 0 !important;
}
[data-testid="stTextInput"] input:focus {
    border-bottom: 2px solid var(--accent) !important;
    box-shadow: none !important;
}
[data-testid="stTextInput"] input::placeholder { color: var(--ink-faint) !important; }
[data-testid="stWidgetLabel"] p {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.68rem !important; letter-spacing: 0.08em; text-transform: uppercase;
    color: var(--ink-faint) !important;
}

/* Slider: ink square thumb */
[data-testid="stSlider"] [role="slider"] {
    background-color: var(--ink) !important; border-radius: 0 !important;
}
[data-testid="stSlider"] [data-testid="stSliderThumbValue"] { color: var(--ink-soft) !important; }

/* Selectbox: squared, ruled */
[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
    background: transparent !important;
    border: 1px solid var(--rule) !important;
    border-radius: 0 !important;
    color: var(--ink) !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.8rem !important;
    min-height: 2.4rem;
}

/* ---------- File uploader: dashed paper tray ---------- */
[data-testid="stFileUploaderDropzone"], [data-testid="stFileUploadDropzone"] {
    background: transparent !important;
    border: 1px dashed var(--ink-faint) !important;
    border-radius: 0 !important;
    padding: 18px 14px !important;
    transition: background 0.2s ease;
}
[data-testid="stFileUploaderDropzone"]:hover, [data-testid="stFileUploadDropzone"]:hover {
    background: var(--accent-soft) !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] span,
[data-testid="stFileUploaderDropzoneInstructions"] small { color: var(--ink-soft) !important; }
[data-testid="stFileUploaderFile"] { background: transparent; border-radius: 0; border-bottom: 1px dotted var(--hairline); }
[data-testid="stFileUploaderFileName"] { color: var(--ink) !important; font-size: 0.76rem !important; }

/* ---------- Buttons: ink stamps ---------- */
.stButton > button {
    width: 100%;
    background: var(--ink);
    color: var(--paper) !important;
    border: 1px solid var(--ink);
    border-radius: 0;
    padding: 0.6rem 1rem;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.74rem !important;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    transition: background 0.15s ease, color 0.15s ease;
}
.stButton > button:hover {
    background: var(--accent);
    border-color: var(--accent);
    color: var(--paper) !important;
}
.stButton > button[kind="secondary"] {
    background: transparent;
    border: 1px solid var(--rule);
    color: var(--ink-soft) !important;
}
.stButton > button[kind="secondary"]:hover {
    background: var(--ink); color: var(--paper) !important;
}

/* ---------- Chat as manuscript prose ---------- */
[data-testid="stChatMessage"] {
    background: transparent;
    border: 0;
    border-radius: 0;
    padding: 4px 0 10px 0;
    margin-bottom: 6px;
    box-shadow: none;
    animation: none;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    border-left: 2px solid var(--ink);
    padding-left: 18px;
    margin-left: 14%;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
    border-left: 2px solid var(--accent);
    padding-left: 18px;
    margin-right: 6%;
}
[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] li {
    color: var(--ink); line-height: 1.72; font-size: 1.02rem;
    font-family: 'Newsreader', Georgia, serif;
}
[data-testid="stChatMessage"] code {
    background: var(--paper-dim);
    color: var(--accent); padding: 1px 5px;
    font-family: 'IBM Plex Mono', monospace; font-size: 0.8em;
}
[data-testid="stChatMessage"] pre {
    background: var(--paper-dim) !important;
    border: 1px solid var(--hairline); border-radius: 0;
}
[data-testid="stChatMessage"] pre code { background: transparent; color: var(--ink); }
[data-testid="stChatMessage"] h1, [data-testid="stChatMessage"] h2, [data-testid="stChatMessage"] h3 {
    font-size: 1.15rem !important; margin-top: 1em;
}
[data-testid="stChatMessage"] table { border-color: var(--hairline); }

/* Chat input: a ruled writing line */
[data-testid="stBottomBlockContainer"], [data-testid="stChatInput"] { background: transparent !important; }
[data-testid="stBottom"] { background: linear-gradient(to top, var(--paper) 78%, transparent) !important; }
[data-testid="stChatInput"] > div {
    background: var(--paper) !important;
    border: 0 !important;
    border-top: 2px solid var(--ink) !important;
    border-radius: 0 !important;
    box-shadow: none !important;
}
[data-testid="stChatInput"] textarea {
    color: var(--ink) !important; font-size: 1rem !important;
    font-family: 'Newsreader', Georgia, serif !important;
}
[data-testid="stChatInput"] textarea::placeholder { color: var(--ink-faint) !important; }
[data-testid="stChatInput"] button { border-radius: 0 !important; }

/* ---------- Citations: footnotes block ---------- */
[data-testid="stExpander"] {
    background: transparent !important;
    border: 1px solid var(--rule) !important;
    border-radius: 0 !important;
    margin-top: 10px;
}
[data-testid="stExpander"] summary {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem !important;
    font-weight: 600; letter-spacing: 0.1em; text-transform: uppercase;
    color: var(--accent) !important;
    padding: 10px 14px !important;
}
[data-testid="stExpander"] summary:hover { background: var(--accent-soft); }
[data-testid="stExpander"] summary p { color: var(--accent) !important; font-size: 0.72rem !important; }
[data-testid="stExpander"] svg { stroke: var(--accent); }

.cite-card {
    border-left: 2px solid var(--accent);
    padding: 8px 0 10px 16px;
    margin-bottom: 4px;
}
.cite-head {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem; letter-spacing: 0.04em;
    color: var(--ink-soft); margin-bottom: 5px;
}
.cite-head .fn-mark { color: var(--accent); font-weight: 600; margin-right: 7px; }
.cite-head .sep { color: var(--hairline); margin: 0 7px; }
.cite-snippet {
    font-family: 'Newsreader', Georgia, serif;
    font-size: 0.86rem; line-height: 1.62; color: var(--ink-soft);
    font-style: italic;
    background: var(--mark);
    padding: 1px 2px;
}

/* ---------- Empty state: a blank front page ---------- */
.empty {
    border-top: 3px double var(--rule);
    border-bottom: 3px double var(--rule);
    padding: 52px 24px; text-align: center;
}
.empty-kicker {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.66rem; letter-spacing: 0.2em; text-transform: uppercase;
    color: var(--accent); margin-bottom: 14px;
}
.empty-title {
    font-family: 'Fraunces', Georgia, serif;
    font-size: 1.7rem; font-weight: 600; color: var(--ink); margin-bottom: 10px;
}
.empty-text {
    font-size: 0.98rem; color: var(--ink-soft); max-width: 460px;
    margin: 0 auto; line-height: 1.65;
}

/* Opening-questions kicker + suggestion links */
.try-kicker {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.66rem; letter-spacing: 0.18em; text-transform: uppercase;
    color: var(--ink-faint); margin: 26px 0 10px;
}
.try-kicker::after { content: ""; display: block; width: 52px; border-top: 1px solid var(--rule); margin-top: 7px; }
.stButton > button.suggestion {
    background: transparent; border: 0; border-bottom: 1px solid var(--hairline);
    border-radius: 0; text-align: left; text-transform: none; letter-spacing: 0;
    font-family: 'Newsreader', Georgia, serif !important;
    font-style: italic; font-size: 0.95rem !important;
    color: var(--ink-soft) !important; padding: 7px 2px;
}
.stButton > button.suggestion:hover {
    background: transparent; color: var(--accent) !important;
    border-bottom: 1px solid var(--accent);
}

/* ---------- Alerts / spinner ---------- */
[data-testid="stAlert"] {
    background: var(--paper-dim) !important;
    border: 1px solid var(--hairline) !important;
    border-left: 3px solid var(--accent) !important;
    border-radius: 0 !important;
    color: var(--ink) !important;
}
[data-testid="stSpinner"] > div { border-top-color: var(--accent) !important; }

/* Timestamped log line (raw-error / diagnostics strip) */
.logline {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.74rem; color: var(--ink-soft);
    padding: 8px 0;
}
.logline .stamp { color: var(--accent); }
</style>
"""


# ==============================================================================
# CACHED RESOURCES
# ==============================================================================

class HistoryBase(DeclarativeBase):
    pass


class Conversation(HistoryBase):
    __tablename__ = "conversations"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    title: Mapped[str] = mapped_column(String(120), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    messages: Mapped[List["StoredMessage"]] = relationship(
        back_populates="conversation",
        cascade="all, delete-orphan",
        order_by="StoredMessage.created_at",
    )


class StoredMessage(HistoryBase):
    __tablename__ = "messages"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    conversation_id: Mapped[str] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False, index=True
    )
    role: Mapped[str] = mapped_column(String(20), nullable=False)
    content: Mapped[str] = mapped_column(String, nullable=False)
    citations: Mapped[List[Dict]] = mapped_column(JSON, nullable=False, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    conversation: Mapped[Conversation] = relationship(back_populates="messages")


@st.cache_resource(show_spinner=False)
def get_history_engine():
    """Create the local SQLite store once per Streamlit process."""
    engine = create_engine(
        f"sqlite:///{HISTORY_DB_PATH}",
        connect_args={"check_same_thread": False},
    )

    @event.listens_for(engine, "connect")
    def configure_sqlite(dbapi_connection, _record):
        del _record
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    HistoryBase.metadata.create_all(engine)
    return engine


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def create_conversation() -> str:
    conversation_id = str(uuid.uuid4())
    now = _now()
    with Session(get_history_engine()) as session:
        session.add(
            Conversation(
                id=conversation_id,
                title="New conversation",
                created_at=now,
                updated_at=now,
            )
        )
        session.commit()
    return conversation_id


def list_conversations() -> List[Conversation]:
    with Session(get_history_engine()) as session:
        return list(
            session.scalars(
                select(Conversation).order_by(Conversation.updated_at.desc())
            ).all()
        )


def load_conversation(conversation_id: str) -> List[Dict]:
    with Session(get_history_engine()) as session:
        conversation = session.get(Conversation, conversation_id)
        if conversation is None:
            return []
        return [
            {
                "role": message.role,
                "content": message.content,
                "citations": message.citations or [],
            }
            for message in conversation.messages
        ]


def save_message(
    conversation_id: str,
    role: str,
    content: str,
    citations: Optional[List[Dict]] = None,
) -> str:
    message_id = str(uuid.uuid4())
    now = _now()
    with Session(get_history_engine()) as session:
        conversation = session.get(Conversation, conversation_id)
        if conversation is None:
            raise ValueError("Conversation no longer exists.")
        if role == "user" and conversation.title == "New conversation":
            title = " ".join(content.split())
            conversation.title = (
                title[:TITLE_MAX_CHARS].rstrip() + ("…" if len(title) > TITLE_MAX_CHARS else "")
            )
        conversation.updated_at = now
        session.add(
            StoredMessage(
                id=message_id,
                conversation_id=conversation_id,
                role=role,
                content=content,
                citations=citations or [],
                created_at=now,
            )
        )
        session.commit()
    return message_id


def delete_message(message_id: str) -> None:
    with Session(get_history_engine()) as session:
        message = session.get(StoredMessage, message_id)
        if message is not None:
            session.delete(message)
            session.commit()


def rename_conversation(conversation_id: str, title: str) -> None:
    cleaned = " ".join(title.split()).strip()
    if not cleaned:
        raise ValueError("Conversation title cannot be empty.")
    with Session(get_history_engine()) as session:
        conversation = session.get(Conversation, conversation_id)
        if conversation is None:
            raise ValueError("Conversation no longer exists.")
        conversation.title = cleaned[:120]
        conversation.updated_at = _now()
        session.commit()


def delete_conversation(conversation_id: str) -> None:
    with Session(get_history_engine()) as session:
        conversation = session.get(Conversation, conversation_id)
        if conversation is not None:
            session.delete(conversation)
            session.commit()


@st.cache_resource(show_spinner=False)
def load_embeddings() -> HuggingFaceEmbeddings:
    """Load the sentence-transformer embedding model once per process."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


# ==============================================================================
# DOCUMENT PIPELINE
# ==============================================================================

def fingerprint(uploaded_files) -> str:
    """Hash file names and contents so same-sized replacements are re-indexed."""
    digest = hashlib.md5()
    for uploaded in sorted(uploaded_files, key=lambda file: file.name):
        digest.update(uploaded.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(uploaded.getbuffer())
        digest.update(b"\0")
    return digest.hexdigest()


def load_pdfs(uploaded_files) -> Tuple[List[Document], List[str]]:
    """Write uploads to a temp dir, parse each with PyPDFLoader, normalise metadata."""
    pages: List[Document] = []
    failed: List[str] = []

    with tempfile.TemporaryDirectory() as tmpdir:
        for uploaded in uploaded_files:
            safe_name = os.path.basename(uploaded.name)
            disk_path = os.path.join(tmpdir, safe_name)
            try:
                with open(disk_path, "wb") as handle:
                    handle.write(uploaded.getbuffer())

                loaded = PyPDFLoader(disk_path).load()
                if not loaded:
                    failed.append(f"{safe_name} (no extractable text)")
                    continue

                for doc in loaded:
                    # PyPDF pages are zero-indexed at load time; humans count from 1.
                    raw_page = doc.metadata.get("page", 0)
                    try:
                        page_no = int(raw_page) + 1
                    except (TypeError, ValueError):
                        page_no = 1
                    doc.metadata["source"] = safe_name
                    doc.metadata["page"] = page_no

                pages.extend(loaded)
            except Exception as exc:  # noqa: BLE001 - surfaced to the user
                failed.append(f"{safe_name} ({type(exc).__name__})")

    return pages, failed


def build_vectorstore(pages: List[Document]) -> Tuple[FAISS, int]:
    """Chunk the parsed pages and embed them into a single unified FAISS index."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", "? ", "! ", "; ", " ", ""],
    )
    chunks = [c for c in splitter.split_documents(pages) if c.page_content.strip()]
    if not chunks:
        raise ValueError("No readable text found in the uploaded PDFs.")

    store = FAISS.from_documents(chunks, load_embeddings())
    return store, len(chunks)


# ==============================================================================
# RAG CHAIN
# ==============================================================================

def build_rag_chain(vectorstore: FAISS, api_key: str, model_name: str, temperature: float):
    """Assemble: history-aware MMR retriever -> strictly grounded answer chain."""
    llm = ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key,
        temperature=temperature,
        max_output_tokens=2048,
    )

    # Verified against langchain-community 0.3.27: VectorStoreRetriever routes
    # search_type="mmr" to FAISS.max_marginal_relevance_search, which accepts
    # exactly k / fetch_k / lambda_mult as kwargs — no silent fallback.
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": RETRIEVER_K,
            "fetch_k": RETRIEVER_FETCH_K,
            "lambda_mult": RETRIEVER_LAMBDA,
        },
    )

    contextualize_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", CONTEXTUALIZE_SYSTEM_PROMPT),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_prompt
    )

    answer_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", ANSWER_SYSTEM_PROMPT),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    document_prompt = PromptTemplate(
        input_variables=["page_content", "source", "page"],
        template=DOCUMENT_TEMPLATE,
    )
    stuff_chain = create_stuff_documents_chain(
        llm, answer_prompt, document_prompt=document_prompt
    )

    return create_retrieval_chain(history_aware_retriever, stuff_chain)


def to_langchain_history(messages: List[Dict]) -> List:
    """Convert the last N stored turns into LangChain message objects."""
    trimmed = messages[-(MAX_HISTORY_TURNS * 2):]
    history = []
    for msg in trimmed:
        if msg["role"] == "user":
            history.append(HumanMessage(content=msg["content"]))
        else:
            history.append(AIMessage(content=msg["content"]))
    return history


def extract_citations(documents: List[Document]) -> List[Dict]:
    """Deduplicate retrieved chunks by (file, page) and prepare display payloads."""
    seen = set()
    citations: List[Dict] = []
    for doc in documents:
        source = str(doc.metadata.get("source", "unknown.pdf"))
        page = doc.metadata.get("page", "—")
        key = (source, page)
        if key in seen:
            continue
        seen.add(key)

        text = " ".join(doc.page_content.split())
        if len(text) > SNIPPET_CHARS:
            text = text[:SNIPPET_CHARS].rsplit(" ", 1)[0] + " …"

        citations.append({"source": source, "page": page, "snippet": text})
    return citations


def render_citations(citations: List[Dict]) -> None:
    """Render citations as numbered footnotes with monospace metadata."""
    if not citations:
        return
    with st.expander(f"NOTES — {len(citations)} source(s) cited", expanded=False):
        for idx, cite in enumerate(citations, start=1):
            st.markdown(
                f"""
                <div class="cite-card">
                    <div class="cite-head">
                        <span class="fn-mark">[{idx}]</span>{html.escape(str(cite["source"]))}
                        <span class="sep">·</span>p.{html.escape(str(cite["page"]))}
                    </div>
                    <div class="cite-snippet">“{html.escape(cite["snippet"])}”</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ==============================================================================
# RELIABILITY HELPERS — rate limits, cooldowns, error classification
# ==============================================================================

def is_rate_limit_error(exc: BaseException) -> bool:
    """Detect free-tier 429 / quota exhaustion errors from any text form."""
    text = str(exc).lower()
    return (
        "429" in text
        or "quota" in text
        or "resource_exhausted" in text
        or "rate limit" in text
    )


def classify_error(exc: BaseException, model_name: str) -> str:
    """
    Map the raw exception to a plain-language explanation.
    Order matters: 429/404 are checked before the generic key/401/403 branch
    because Google sometimes wraps quota errors with auth-adjacent wording.
    """
    text = str(exc).lower()
    if is_rate_limit_error(exc):
        return (
            "**Rate limit reached** — you've hit Gemini's free-tier request limit "
            f"on `{model_name}`. This isn't a bug; your documents are fine. Wait "
            "about a minute and try again, or switch to a lighter model "
            "(`gemini-3.5-flash-lite` / `gemini-3.1-flash-lite`) in the sidebar. "
            f"Live quota usage: {QUOTA_INFO_URL}"
        )
    if "404" in text or "not found" in text:
        return (
            f"**Model unavailable** — the API rejected `{model_name}`; the model "
            "name is likely invalid or retired for your key. Pick a different "
            "model from the sidebar dropdown."
        )
    if "401" in text or "403" in text or "api key" in text or "api_key" in text:
        return (
            "**API key problem** — your Google AI Studio key looks invalid, "
            "expired, or lacks access to this model. Paste a fresh key from "
            f"{QUOTA_INFO_URL}"
        )
    return (
        "**Request failed** — something went wrong talking to Gemini. "
        "Check your internet connection and API key, then retry."
    )


def invoke_with_retry(chain, chain_input: Dict) -> Dict:
    """
    Invoke the RAG chain, retrying only on free-tier 429/quota errors with
    exponential backoff (2s -> 4s -> 8s). Any other error is raised immediately.
    """
    attempt = 0
    while True:
        try:
            return chain.invoke(chain_input)
        except Exception as exc:  # noqa: BLE001 - classified by caller
            attempt += 1
            if not is_rate_limit_error(exc) or attempt >= RETRY_MAX_ATTEMPTS:
                raise
            delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))  # 2 → 4 → 8
            with st.spinner(
                f"Free-tier rate limit hit — retrying in {int(delay)}s "
                f"(attempt {attempt + 1} of {RETRY_MAX_ATTEMPTS})…"
            ):
                time.sleep(delay)


def enforce_cooldown() -> None:
    """
    Block a second query within MIN_REQUEST_GAP of the last accepted one.
    Guards against Streamlit rerun double-firing and burning free-tier quota
    twice per user action.
    """
    now = time.monotonic()
    last = st.session_state.get("last_request_time")
    if last is not None and (now - last) < MIN_REQUEST_GAP:
        remaining = MIN_REQUEST_GAP - (now - last)
        st.info(
            f"⏱ Too fast — rapid requests can burn free-tier quota twice. "
            f"Please wait {max(1, int(remaining) + 1)}s and send your question again."
        )
        st.stop()
    st.session_state.last_request_time = now


# ==============================================================================
# SESSION STATE
# ==============================================================================

def init_state() -> None:
    defaults = {
        "messages": [],
        "conversation_id": None,
        "auto_restore_conversation": True,
        "vectorstore": None,
        "index_signature": None,
        "chunk_count": 0,
        "page_count": 0,
        "indexed_files": [],
        "pending_prompt": None,
        "last_request_time": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ==============================================================================
# APPLICATION
# ==============================================================================

st.set_page_config(
    page_title=f"{APP_NAME} · {APP_TAGLINE}",
    page_icon="🗂",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
inject_frontend_css()
init_state()
get_history_engine()

if st.session_state.auto_restore_conversation and st.session_state.conversation_id is None:
    saved_conversations = list_conversations()
    if saved_conversations:
        latest = saved_conversations[0]
        st.session_state.conversation_id = latest.id
        st.session_state.messages = load_conversation(latest.id)


# ------------------------------- SIDEBAR --------------------------------------
with st.sidebar:
    st.markdown(
        f"""
        <div class="side-brand">
            <div>
                <div class="side-brand-name">{APP_NAME}</div>
                <div class="side-brand-tag">Vol. I — {APP_TAGLINE}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    # ---- Credentials
    st.markdown('<div class="side-label">Credentials</div>', unsafe_allow_html=True)

    default_key = os.getenv("GOOGLE_API_KEY", "")
    if not default_key:
        try:
            default_key = st.secrets.get("GOOGLE_API_KEY", "")
        except Exception:  # noqa: BLE001 - secrets.toml may not exist
            default_key = ""

    api_key = st.text_input(
        "Google AI Studio API key",
        value=default_key,
        type="password",
        placeholder="AIza…",
        help="Create a free key at aistudio.google.com/apikey. "
             "It is held in memory for this session only.",
    )

    model_name = st.selectbox(
        "Reasoning model",
        options=AVAILABLE_MODELS,
        index=AVAILABLE_MODELS.index(DEFAULT_MODEL),
        help="Retired model strings (gemini-1.5-*, gemini-2.0-*) return 404. "
             "Only these verified Flash endpoints are offered.",
    )

    temperature = st.slider(
        "Answer Style", min_value=0.0, max_value=1.0, value=0.1, step=0.05,
        help="Lower values keep answers focused and factual; higher values allow more exploration.",
    )

    # ---- Knowledge base
    st.markdown('<div class="side-label">Knowledge Base</div>', unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Drop your PDFs here",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    build_clicked = st.button("Build Index", type="primary")

    # ---- Index construction
    if build_clicked:
        if not uploaded_files:
            st.warning("Upload at least one PDF first.")
        else:
            signature = fingerprint(uploaded_files)
            if signature == st.session_state.index_signature:
                st.info("Index is already up to date for these files.")
            else:
                try:
                    with st.spinner("Parsing PDFs…"):
                        pages, failed = load_pdfs(uploaded_files)
                    if not pages:
                        st.error("Could not extract any text. Are these scanned images?")
                    else:
                        with st.spinner(f"Embedding {len(pages)} pages…"):
                            store, chunk_count = build_vectorstore(pages)

                        st.session_state.vectorstore = store
                        st.session_state.index_signature = signature
                        st.session_state.chunk_count = chunk_count
                        st.session_state.page_count = len(pages)
                        st.session_state.indexed_files = sorted(
                            {p.metadata.get("source", "unknown.pdf") for p in pages}
                        )

                        if failed:
                            st.warning("Skipped: " + ", ".join(failed))
                        st.success(f"Indexed {chunk_count} chunks from {len(pages)} pages.")
                        st.rerun()
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Indexing failed: {exc}")

    # ---- Conversation history
    st.markdown('<div class="side-label">History</div>', unsafe_allow_html=True)
    if st.button("New chat", type="secondary", key="new_chat"):
        st.session_state.conversation_id = None
        st.session_state.messages = []
        st.session_state.auto_restore_conversation = False
        st.session_state.pending_prompt = None
        st.rerun()

    saved_conversations = list_conversations()
    if saved_conversations:
        for conversation in saved_conversations:
            is_current = conversation.id == st.session_state.conversation_id
            history_col, delete_col = st.columns([5, 1])
            with history_col:
                label = f"{'● ' if is_current else ''}{conversation.title}"
                if st.button(
                    label,
                    key=f"conversation_{conversation.id}",
                    type="primary" if is_current else "secondary",
                    use_container_width=True,
                ):
                    st.session_state.conversation_id = conversation.id
                    st.session_state.messages = load_conversation(conversation.id)
                    st.session_state.auto_restore_conversation = False
                    st.rerun()
            with delete_col:
                if st.button("×", key=f"delete_{conversation.id}", help="Delete conversation"):
                    delete_conversation(conversation.id)
                    if st.session_state.conversation_id == conversation.id:
                        st.session_state.conversation_id = None
                        st.session_state.messages = []
                        st.session_state.auto_restore_conversation = False
                    st.rerun()
            with st.expander("Rename", expanded=False):
                rename_value = st.text_input(
                    "Conversation title",
                    value=conversation.title,
                    key=f"rename_value_{conversation.id}",
                    label_visibility="collapsed",
                )
                if st.button("Save title", key=f"rename_{conversation.id}"):
                    try:
                        rename_conversation(conversation.id, rename_value)
                        st.rerun()
                    except ValueError as exc:
                        st.error(str(exc))
    else:
        st.markdown(
            '<div class="status-line">No saved conversations yet.</div>',
            unsafe_allow_html=True,
        )

    # ---- Status ledger
    st.markdown('<div class="side-label">Status</div>', unsafe_allow_html=True)

    if st.session_state.vectorstore is not None:
        st.markdown(
            '<div class="status-line"><span class="sq sq-ok"></span>INDEX READY</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="status-line">{RETRIEVER_K} evidence passages · '
            f'MMR retrieval</div>',
            unsafe_allow_html=True,
        )
        for name in st.session_state.indexed_files:
            st.markdown(
                f'<div class="file-row">{html.escape(name)}</div>',
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            '<div class="status-line"><span class="sq sq-wait"></span>AWAITING DOCUMENTS</div>',
            unsafe_allow_html=True,
        )

    # ---- Session controls
    st.markdown('<div class="side-label">Session</div>', unsafe_allow_html=True)
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Clear chat", type="secondary"):
            st.session_state.messages = []
            st.session_state.conversation_id = None
            st.session_state.auto_restore_conversation = False
            st.rerun()
    with col_b:
        if st.button("Reset index", type="secondary"):
            st.session_state.vectorstore = None
            st.session_state.index_signature = None
            st.session_state.chunk_count = 0
            st.session_state.page_count = 0
            st.session_state.indexed_files = []
            st.session_state.messages = []
            st.session_state.conversation_id = None
            st.session_state.auto_restore_conversation = False
            st.rerun()

    # ---- Quota note (Part 2E)
    st.markdown(
        f'<div style="font-family:\'IBM Plex Mono\',monospace;font-size:0.66rem;'
        f'line-height:1.7;color:var(--ink-faint);margin-top:22px;">'
        f'Free-tier quotas apply per key and per model. '
        f'<a href="{QUOTA_INFO_URL}" target="_blank">Check usage at aistudio.google.com/apikey</a>'
        f"</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div style="font-family:\'IBM Plex Mono\',monospace;font-size:0.62rem;'
        'color:var(--ink-faint);margin-top:14px;line-height:1.7;">'
        "EMB all-MiniLM-L6-v2 · FAISS in-memory · MMR k=4"
        "</div>",
        unsafe_allow_html=True,
    )


# ------------------------------- MASTHEAD -------------------------------------
st.markdown(
    f"""
    <div class="masthead">
        <div class="masthead-title">{APP_NAME}</div>
        <div class="masthead-right">
            RAG · MMR · CITED ANSWERS<br>
            EST. SESSION {time.strftime("%d %b %Y").upper()}
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    """
    <p class="standfirst">
    Put in your PDFs, ask ordinary questions, and get answers <em>with the receipts</em> —
    every claim is drawn from retrieved passages and footnoted back to the exact file
    and page. Follow-up questions keep their context.
    </p>
    """,
    unsafe_allow_html=True,
)


# ------------------------------- CHAT AREA ------------------------------------
if st.session_state.messages:
    # Replay conversation, including messages restored from SQLite after refresh.
    for message in st.session_state.messages:
        avatar = "✍️" if message["role"] == "user" else "📖"
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                render_citations(message.get("citations", []))

if st.session_state.vectorstore is None:
    if st.session_state.messages:
        st.info("Conversation restored. Rebuild the PDF index to ask follow-up questions.")
    else:
        st.markdown(
            """
            <div class="empty">
                <div class="empty-kicker">Front Matter</div>
                <div class="empty-title">Your reading desk is clear</div>
                <div class="empty-text">
                    Add one or more PDFs in the left margin, paste your Google AI Studio
                    API key, then press <strong>Build Index</strong>. Answers, when they
                    come, will cite their pages like footnotes.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
else:
    st.markdown(
        """
        <section class="aether-hero" aria-labelledby="hero-heading">
            <div>
                <div class="hero-kicker">Evidence instrument · online</div>
                <h1 id="hero-heading" class="hero-title">Ask the archive.</h1>
                <p class="hero-copy">
                    Aether maps your PDFs into a living evidence field, then answers
                    with the exact pages that support each claim.
                </p>
            </div>
            <div aria-hidden="true"></div>
        </section>
        """,
        unsafe_allow_html=True,
    )
    index_stats_scene(
        {
            "documents": len(st.session_state.indexed_files),
            "pages": st.session_state.page_count,
            "chunks": st.session_state.chunk_count,
        },
        height=300,
    )
    # Suggested openers on a fresh index
    if not st.session_state.messages:
        st.markdown('<div class="try-kicker">Opening questions</div>', unsafe_allow_html=True)
        suggestions = [
            "Summarise the key findings across all documents",
            "What are the main risks or limitations mentioned?",
            "List every metric or figure reported, with its source",
        ]
        for text in suggestions:
            if st.button(text, key=f"suggest_{text[:18]}", type="secondary"):
                st.session_state.pending_prompt = text
                st.rerun()

# Input handling (chat_input must stay at top level)
typed = st.chat_input("Ask anything about your documents…")
user_prompt = typed or st.session_state.pending_prompt
st.session_state.pending_prompt = None

if user_prompt:
    if st.session_state.vectorstore is None:
        st.warning("Build the vector index before asking questions.")
    elif not api_key.strip():
        st.error("Add your Google AI Studio API key in the sidebar first.")
    else:
        enforce_cooldown()

        history = to_langchain_history(st.session_state.messages)
        if st.session_state.conversation_id is None:
            st.session_state.conversation_id = create_conversation()
        st.session_state.auto_restore_conversation = False
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        user_message_id = save_message(
            st.session_state.conversation_id, "user", user_prompt
        )

        with st.chat_message("user", avatar="✍️"):
            st.markdown(user_prompt)

        with st.chat_message("assistant", avatar="📖"):
            placeholder = st.empty()
            try:
                with st.spinner("Retrieving passages and reasoning…"):
                    chain = build_rag_chain(
                        st.session_state.vectorstore,
                        api_key.strip(),
                        model_name,
                        temperature,
                    )
                    result = invoke_with_retry(
                        chain, {"input": user_prompt, "chat_history": history}
                    )

                answer = (result.get("answer") or "").strip() or f"{REFUSAL}."
                retrieved = result.get("context", []) or []
                citations = (
                    [] if answer.lower().startswith(REFUSAL.lower())
                    else extract_citations(retrieved)
                )

                placeholder.markdown(answer)
                render_citations(citations)

                st.session_state.messages.append(
                    {"role": "assistant", "content": answer, "citations": citations}
                )
                save_message(
                    st.session_state.conversation_id,
                    "assistant",
                    answer,
                    citations,
                )
            except Exception as exc:  # noqa: BLE001
                # Surface the raw exception text (truncated) alongside the
                # classification so nothing is swallowed into a generic message.
                hint = classify_error(exc, model_name)
                raw = html.escape(str(exc)[:400])
                stamp = time.strftime("%H:%M:%S")
                placeholder.markdown(
                    f'{hint}\n\n<div class="logline">'
                    f'<span class="stamp">[{stamp}]</span> RAW: {raw}</div>',
                    unsafe_allow_html=True,
                )
                st.session_state.messages.pop()  # drop the orphaned user turn
                delete_message(user_message_id)
