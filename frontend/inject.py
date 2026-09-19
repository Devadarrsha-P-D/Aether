"""Small, rerun-safe adapters between Streamlit and the frontend scenes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import streamlit as st

from . import tokens

_ROOT = Path(__file__).parent
def _bridge_script() -> str:
    # Streamlit's iframe parent is same-origin in current deployments, but this
    # is undocumented behavior; always retain a static fallback for other hosts.
    return """<script>
(function () {
  try {
    var parentDoc = window.parent && window.parent.document;
    if (!parentDoc || !parentDoc.documentElement) throw new Error("no same-origin parent");
    var root = document.documentElement;
    root.dataset.bridge = "parent";
    root.dataset.theme = parentDoc.documentElement.dataset.theme || "";
  } catch (_) {
    document.documentElement.dataset.bridge = "static";
  }
}());
</script>"""


def inject_frontend_css() -> None:
    """Emit the complete token stylesheet on every Streamlit rerun."""
    # Streamlit remounts markdown elements during widget interactions. A
    # session-state guard would prevent the style from being recreated even
    # though its previous DOM node no longer exists.
    st.markdown(f"<style data-aether-tokens>{tokens.emit_css()}</style>", unsafe_allow_html=True)
    st.markdown(_bridge_script(), unsafe_allow_html=True)


def _scene(
    name: str,
    *,
    height: int = 420,
    width: int = 1,
    data: Mapping[str, Any] | None = None,
) -> Any:
    markup = (_ROOT / "scenes" / f"{name}.html").read_text(encoding="utf-8")
    if data:
        payload = json.dumps(dict(data), separators=(",", ":")).replace("</", "<\\/")
        markup = markup.replace(
            "<script>",
            f"<script>window.__AETHER_STATS__={payload};",
            1,
        )
    return st.components.v1.html(markup, height=height, width=width, scrolling=False)


def ambient_scene(*, height: int = 420, width: int = 1) -> Any:
    """Render the ambient background scene."""
    return _scene("ambient", height=height, width=width)


def index_stats_scene(
    stats: Mapping[str, Any] | None = None, *, height: int = 300, width: int = 1
) -> Any:
    """Render the index statistics scene."""
    return _scene("index_stats", height=height, width=width, data=stats)


# Explicit render aliases make migration from component-oriented call sites easy.
render_ambient = ambient_scene
render_index_stats = index_stats_scene
