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


# --- numbers as Urdu words ------------------------------------------------------------

# 1-99 are irregular in Urdu, so they are listed in full (index = value).
_UNDER_100 = (
    "صفر ایک دو تین چار پانچ چھ سات آٹھ نو دس گیارہ بارہ تیرہ چودہ پندرہ سولہ سترہ اٹھارہ "
    "انیس بیس اکیس بائیس تئیس چوبیس پچیس چھبیس ستائیس اٹھائیس انتیس تیس اکتیس بتیس تینتیس "
    "چونتیس پینتیس چھتیس سینتیس اڑتیس انتالیس چالیس اکتالیس بیالیس تینتالیس چوالیس پینتالیس "
    "چھیالیس سینتالیس اڑتالیس انچاس پچاس اکاون باون ترپن چون پچپن چھپن ستاون اٹھاون انسٹھ "
    "ساٹھ اکسٹھ باسٹھ ترسٹھ چونسٹھ پینسٹھ چھیاسٹھ سڑسٹھ اڑسٹھ انہتر ستر اکہتر بہتر تہتر چوہتر "
    "پچھتر چھہتر ستتر اٹھتر اناسی اسی اکاسی بیاسی تراسی چوراسی پچاسی چھیاسی ستاسی اٹھاسی "
    "نواسی نوے اکانوے بانوے ترانوے چورانوے پچانوے چھیانوے ستانوے اٹھانوے ننانوے"
).split()
assert len(_UNDER_100) == 100

# South Asian grouping: کروڑ (10^7), لاکھ (10^5), ہزار (10^3), سو (10^2).
_SCALES = ((10_000_000, "کروڑ"), (100_000, "لاکھ"), (1_000, "ہزار"), (100, "سو"))

_RUPEES_BEFORE = re.compile(r"\b(?:Rs\.?|PKR)\s*(\d[\d,]*)", re.IGNORECASE)
_PERCENT = re.compile(r"(\d+(?:\.\d+)?)\s*%")
_THOUSANDS_COMMA = re.compile(r"(?<=\d),(?=\d{2,3}\b)")
# Phone numbers: start with 0 or +, at least 7 digits, may contain spaces or dashes.
_PHONE = re.compile(r"(?<!\d)(?:\+|0)\d(?:[\d\- ]{5,}\d)")
_DECIMAL = re.compile(r"(?<!\d)(\d+)\.(\d+)(?!\d)")
_INTEGER = re.compile(r"\d+")


def number_to_urdu(n: int) -> str:
    """Spell a non-negative integer in Urdu words, e.g. 1234 -> ایک ہزار دو سو چونتیس."""
    if n < 0:
        return "منفی " + number_to_urdu(-n)
    if n < 100:
        return _UNDER_100[n]
    words: list[str] = []
    for value, name in _SCALES:
        count, n = divmod(n, value)
        if count:
            words += [number_to_urdu(count), name]
    if n:
        words.append(_UNDER_100[n])
    return " ".join(words)


def digits_to_urdu(digits: str) -> str:
    """Read a digit string one digit at a time (phone numbers, codes)."""
    return " ".join(_UNDER_100[int(d)] for d in digits if d.isdigit())


def numbers_to_words(text: str) -> str:
    """Replace digits in `text` with Urdu words: amounts, decimals, percents, phones."""
    text = _RUPEES_BEFORE.sub(r"\1 روپے", text)
    text = _THOUSANDS_COMMA.sub("", text)
    text = _PHONE.sub(lambda m: " " + digits_to_urdu(m.group(0)) + " ", text)
    text = _PERCENT.sub(r"\1 فیصد", text)
    text = _DECIMAL.sub(
        lambda m: f"{number_to_urdu(int(m.group(1)))} اعشاریہ {digits_to_urdu(m.group(2))}", text
    )

    def integer(match: re.Match[str]) -> str:
        digits = match.group(0)
        # Very long digit runs are codes or IDs, not amounts: read them digit by digit.
        return digits_to_urdu(digits) if len(digits) > 9 else number_to_urdu(int(digits))

    return _INTEGER.sub(integer, text)


def _is_emoji_or_symbol(char: str) -> bool:
    category = unicodedata.category(char)
    # So = other symbols (emoji, ©, ™ ...), Sk/Sm = modifier and math symbols, Cs = surrogates.
    return category in {"So", "Sk", "Sm", "Cs"} or "\U0001f000" <= char <= "\U0001faff"


def normalize_urdu(text: str) -> str:
    """Unicode NFC plus Arabic -> Urdu letter forms."""
    return unicodedata.normalize("NFC", text).translate(_ARABIC_TO_URDU)


def clean_for_tts(text: str, keep_latin: bool = False) -> str:
    """Return plain spoken Urdu: no markdown, emojis, URLs, symbols or digits.

    Latin words are removed unless `keep_latin` is True, because the Urdu TTS
    voices cannot pronounce them. Numbers become Urdu words. Sentence punctuation
    is kept for pauses.
    """
    text = normalize_urdu(text)
    text = _URL.sub(" ", text)
    text = _MD_LINK.sub(r"\1", text)
    text = _LIST_MARKER.sub("", text)
    text = numbers_to_words(text)
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
