"""Minimal translation layer.

Source strings in the code are English. Each language is a JSON file in
openkick/locale/ mapping English -> translation. Missing entries fall back to
English, so a partial translation never breaks the app. To add a language,
copy locale/es.json to locale/<code>.json and translate the values.
"""

from __future__ import annotations

import json
from pathlib import Path

LOCALE_DIR = Path(__file__).parent / "locale"
LANGUAGES = {"es": "Español", "en": "English", "de": "Deutsch", "fr": "Français", "it": "Italiano",
             "pt_BR": "Português (Brasil)", "ru": "Русский", "ja": "日本語", "ko": "한국어", "zh_CN": "简体中文"}
DECIMAL_COMMA = {"es", "de", "fr", "it", "pt_BR", "ru"}     # languages that write 1,5 rather than 1.5

_lang = "es"
_catalog: dict[str, str] = {}


def set_language(code: str):
    global _lang, _catalog
    _lang = code
    _catalog = {}
    f = LOCALE_DIR / f"{code}.json"
    if code != "en" and f.exists():
        _catalog = json.loads(f.read_text(encoding="utf-8"))


def language() -> str:
    return _lang


def tr(text: str, **fmt) -> str:
    out = _catalog.get(text, text)
    return out.format(**fmt) if fmt else out


def pick(value) -> str:
    """Localized value from a {lang: text} dict (templates, plugins) or a plain string. A dict
    without the current language gives its English text, translated by the catalog if it can."""
    if isinstance(value, dict):
        if value.get(_lang):
            return value[_lang]
        en = value.get("en")
        return _catalog.get(en, en) if en else next(iter(value.values()), "")
    return str(value)


def system_language() -> str:
    """The language of the computer if OpenKick has it (pt_BR for pt_PT too), else Spanish."""
    from PySide6.QtCore import QLocale
    name = QLocale.system().name()                 # "de_DE", "pt_BR", "zh_CN"...
    if name in LANGUAGES:
        return name
    base = name.split("_")[0]
    return base if base in LANGUAGES else next((c for c in LANGUAGES if c.split("_")[0] == base), "es")


set_language("es")
