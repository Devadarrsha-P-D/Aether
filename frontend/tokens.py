"""Design tokens shared by the Streamlit shell and iframe scenes."""

from __future__ import annotations

from functools import lru_cache


TOKENS = {
    "ink": "#0a1624",
    "ink-soft": "#14283b",
    "navy": "#07111e",
    "cream": "#f4ecdc",
    "cream-muted": "#b7b7a5",
    "copper": "#c47b52",
    "copper-bright": "#e09a69",
    "line": "rgba(244, 236, 220, .18    )",
    "display": "'Fraunces', Georgia, serif",
    "sans": "'DM Sans', system-ui, sans-serif",
    "mono": "'IBM Plex Mono', ui-monospace, monospace",
    "text-xs": "0.6875rem",
    "text-sm": "0.8125rem",
    "text-md": "1rem",
    "text-lg": "clamp(1.5rem, 3vw, 2.5rem)",
    "space-1": "0.5rem",
    "space-2": "1rem",
    "space-3": "1.5rem",
    "space-4": "2.5rem",
    "ease-out": "cubic-bezier(.16, 1, .3, 1)",
    "ease-smooth": "cubic-bezier(.22, .61, .36, 1)",
}


@lru_cache(maxsize=1)
def css() -> str:
    """Return the one canonical custom-property block used by the frontend."""
    variables = "\n".join(f"  --{name}: {value};" for name, value in TOKENS.items())
    return (
        ":root {\n"
        f"{variables}\n"
        "  --content-max: 1180px;\n"
        "  --motion-fast: 160ms;\n"
        "  --paper: #07111e;\n"
        "  --paper-dim: #0d1b2a;\n"
        "  --ink: #f4ecdc;\n"
        "  --ink-soft: #d6d2c6;\n"
        "  --ink-faint: #b7b7a5;\n"
        "  --hairline: rgba(244,236,220,.18);\n"
        "  --rule: rgba(244,236,220,.42);\n"
        "  --accent: #c47b52;\n"
        "  --accent-soft: rgba(196,123,82,.16);\n"
        "}\n"
        "*, *::before, *::after { box-sizing: border-box; }\n"
        "html, body, .stApp { background: var(--paper); color: var(--ink); }\n"
        "body, [class*='css'] { font-family: var(--sans); }\n"
        "h1, h2, h3, h4, .masthead-title, .hero-title, .side-brand-name { font-family: var(--display) !important; color: var(--ink) !important; }\n"
        "p, li, span, label { color: var(--ink-soft); }\n"
        ".block-container { max-width: var(--content-max); padding-top: 2rem; }\n"
        "[data-testid='stSidebar'] { background: var(--paper-dim); border-right: 1px solid var(--hairline); }\n"
        ".masthead { border-top: 0; border-bottom: 1px solid var(--hairline); }\n"
        ".masthead-title { color: var(--ink) !important; }\n"
        ".standfirst em, a { color: var(--copper-bright) !important; }\n"
        ".side-label, .masthead-right, .try-kicker, .empty-kicker { color: var(--copper-bright) !important; }\n"
        ".side-label { border-color: var(--hairline); }\n"
        ".status-line, .file-row, .ledger-lbl { color: var(--ink-faint); }\n"
        "[data-testid='stChatMessage'] { border-radius: 2px; padding: 1.25rem 1.5rem; animation: none !important; opacity: 1 !important; transform: none !important; }\n"
        "[data-testid='stChatMessage']:has([data-testid='stChatMessageAvatarUser']) { background: rgba(196,123,82,.11); border-left: 3px solid var(--copper); margin-left: 9%; }\n"
        "[data-testid='stChatMessage']:has([data-testid='stChatMessageAvatarAssistant']) { background: rgba(244,236,220,.06); border-left: 3px solid var(--copper-bright); margin-right: 9%; }\n"
        "[data-testid='stChatMessage'] p, [data-testid='stChatMessage'] li { color: var(--ink); font-family: var(--sans); line-height: 1.7; }\n"
        ".cite-card { border-color: var(--hairline) !important; background: rgba(244,236,220,.06) !important; }\n"
        ".cite-head, .fn-mark { color: var(--copper-bright) !important; }\n"
        "button, input, textarea, [role='slider'], [role='combobox'] { transition: border-color var(--motion-fast) var(--ease-out), background var(--motion-fast) var(--ease-out), transform var(--motion-fast) var(--ease-out); }\n"
        "button:focus-visible, input:focus-visible, textarea:focus-visible, [role='slider']:focus-visible, [role='combobox']:focus-visible { outline: 3px solid var(--copper-bright) !important; outline-offset: 3px; }\n"
        ".stButton > button, [data-testid='stDownloadButton'] button { background: var(--ink-soft) !important; color: var(--paper) !important; border: 1px solid var(--ink-soft); border-radius: 2px; }\n"
        ".stButton > button *, [data-testid='stDownloadButton'] button * { color: inherit !important; }\n"
        ".stButton > button:hover, [data-testid='stDownloadButton'] button:hover { background: var(--copper) !important; border-color: var(--copper); color: var(--paper) !important; transform: translateY(-1px); }\n"
        ".stButton > button[kind='secondary'] { background: transparent !important; color: var(--ink-soft) !important; border-color: var(--hairline) !important; }\n"
        "[data-testid='stBaseButton-secondary'] { background: transparent !important; color: var(--ink-soft) !important; border-color: var(--hairline) !important; }\n"
        "[data-testid='stBaseButton-secondary'] * { color: inherit !important; }\n"
        ".stButton > button[kind='secondary']:hover { background: var(--ink) !important; color: var(--paper) !important; border-color: var(--ink) !important; }\n"
        "[data-testid='stBaseButton-secondary']:hover { background: var(--ink) !important; color: var(--paper) !important; border-color: var(--ink) !important; }\n"
        "[data-testid='stBaseButton-primary'], [data-testid='stBaseButton-primary'] * { color: var(--paper) !important; }\n"
        "[data-testid='stWidgetLabel'], [data-testid='stWidgetLabel'] *, [data-testid='stFileUploaderDropzoneInstructions'] * { color: var(--ink-soft) !important; }\n"
        "[data-testid='stSelectbox'] [data-baseweb='select'], [data-testid='stSelectbox'] [data-baseweb='select'] * { color: var(--ink) !important; }\n"
        "[data-testid='stSlider'] [data-testid='stSliderThumbValue'], [data-testid='stSlider'] [data-testid='stSliderThumbValue'] * { color: var(--ink-soft) !important; }\n"
        "[data-testid='stTextInput'] input, [data-testid='stChatInput'] textarea { color: var(--ink) !important; background: rgba(244,236,220,.06) !important; border-color: var(--hairline) !important; }\n"
        ".aether-hero { display: grid; grid-template-columns: 1.3fr .7fr; gap: 1.5rem; align-items: center; padding: 1.25rem 0 2rem; }\n"
        ".hero-kicker { color: var(--copper-bright); font: 600 var(--text-xs) var(--mono); letter-spacing: .18em; text-transform: uppercase; }\n"
        ".hero-title { font-size: clamp(2.5rem, 7vw, 5.6rem); line-height: .95; margin: .45rem 0 1rem; }\n"
        ".hero-copy { max-width: 46rem; color: var(--ink-soft); font-size: 1.05rem; line-height: 1.6; }\n"
        "@keyframes aether-in { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }\n"
        "@media (max-width: 720px) { .aether-hero { grid-template-columns: 1fr; } [data-testid='stChatMessage']:has([data-testid='stChatMessageAvatarUser']), [data-testid='stChatMessage']:has([data-testid='stChatMessageAvatarAssistant']) { margin-left: 0; margin-right: 0; } }\n"
        "@media (prefers-reduced-motion: reduce) { *, *::before, *::after { animation: none !important; transition: none !important; scroll-behavior: auto !important; } }\n"
    )


# A descriptive alias keeps call sites readable while preserving one emitter.
emit_css = css
