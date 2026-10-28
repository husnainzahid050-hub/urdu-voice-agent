"""Urdu text cleaning for TTS and sentence splitting.

TTS models read characters, not markup: asterisks, emojis, URLs or Latin text make
them mispronounce or skip words. `clean_for_tts` turns an LLM reply into plain Urdu.
"""

from __future__ import annotations

import re
import unicodedata

# Arabic code points that look like Urdu letters but are different characters, and
# rare Urdu letters missing from the MMS voice's vocabulary, mapped to what it can say.
_ARABIC_TO_URDU = str.maketrans(
    {
        "ي": "ی",  # Arabic yeh -> Urdu yeh (ی)
        "ى": "ی",  # alef maksura -> ی
        "ك": "ک",  # Arabic kaf -> keheh (ک)
        "ه": "ہ",  # Arabic heh -> heh goal (ہ)
        "ة": "ت",  # teh marbuta -> teh (ت), as in Urdu pronunciation
        "ۃ": "ت",  # Urdu teh marbuta goal (ۃ) -> ت
        "ۀ": "ۂ",  # heh with yeh above -> heh goal with hamza (ۂ)
        "ۓ": "ئے",  # yeh barree with hamza (ۓ) -> ئے
        **{chr(0x06F0 + d): str(d) for d in range(10)},  # Urdu digits ۰-۹ -> 0-9
        **{chr(0x0660 + d): str(d) for d in range(10)},  # Arabic-Indic digits -> 0-9
    }
)

# Urdu sentence enders: full stop (۔), question mark (؟), plus Latin ones the LLM may use.
_SENTENCE_END = "۔؟!?."

_URL = re.compile(r"https?://\S+|www\.\S+")
_MD_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_MARKUP = re.compile(r"[*_`#>|~\[\]{}<>=+^\\/@$%&]")
_LIST_MARKER = re.compile(r"^\s*(?:[-•–]|\d+[.)])\s+", re.MULTILINE)
_SPACES = re.compile(r"(?:[^\S\n]|\u200b)+")  # spaces, tabs, nbsp, zero-width space
_LATIN_WORD = re.compile(r"[A-Za-z]+")
_PUNCT_RUN = re.compile(r"[۔،؟](?:\s*[۔،؟])+")


def _is_emoji_or_symbol(char: str) -> bool:
    category = unicodedata.category(char)
    # So = other symbols (emoji, ©, ™ ...), Sk/Sm = modifier and math symbols, Cs = surrogates.
    return category in {"So", "Sk", "Sm", "Cs"} or "\U0001f000" <= char <= "\U0001faff"


def normalize_urdu(text: str) -> str:
    """Unicode NFC plus Arabic -> Urdu letter forms."""
    return unicodedata.normalize("NFC", text).translate(_ARABIC_TO_URDU)


def clean_for_tts(text: str, keep_latin: bool = False) -> str:
    """Return plain spoken Urdu: no markdown, emojis, URLs or symbols.

    Latin words are removed unless `keep_latin` is True, because the Urdu TTS
    voices cannot pronounce them. Sentence punctuation is kept for pauses.
    """
    text = normalize_urdu(text)
    text = _URL.sub(" ", text)
    text = _MD_LINK.sub(r"\1", text)
    text = _LIST_MARKER.sub("", text)
    text = _MARKUP.sub(" ", text)
    text = "".join(" " if _is_emoji_or_symbol(c) else c for c in text)
    if not keep_latin:
        text = _LATIN_WORD.sub(" ", text)
    # Line breaks become sentence breaks so list items are not run together.
    text = re.sub(r"\s*\n+\s*", "۔ ", text)
    text = re.sub(r"(?<!\d)\.|\.(?!\d)", "۔", text)  # keep decimal points
    text = text.replace(",", "،").replace("?", "؟").replace("!", "۔")
    text = re.sub(r"[:;؛]", "،", text)
    text = _SPACES.sub(" ", text)
    # A run of marks left after stripping (e.g. "؟ ۔") collapses to the strongest one.
    text = _PUNCT_RUN.sub(lambda m: _strongest_mark(m.group(0)), text)
    text = re.sub(r"\s+([۔،؟])", r"\1", text)
    text = text.strip().lstrip("۔،؟ ")
    if text and text[-1] not in _SENTENCE_END:
        text += "۔"
    return text


def _strongest_mark(run: str) -> str:
    for mark in "؟۔،":
        if mark in run:
            return mark
    return run[0]


def split_sentences(text: str) -> list[str]:
    """Split Urdu text into sentences, keeping each sentence's end mark."""
    parts = re.split(rf"(?<=[{re.escape(_SENTENCE_END)}])\s+|\n+", text.strip())
    return [part.strip() for part in parts if part and part.strip(" " + _SENTENCE_END)]
