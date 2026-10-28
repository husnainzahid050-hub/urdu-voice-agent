from src.text_utils import clean_for_tts, normalize_urdu, split_sentences


def test_removes_markdown_emojis_and_bullets() -> None:
    text = "**جی ہاں!** آپ کا آرڈر 😊 تیار ہے۔\n- پہلا قدم: پانی ابالیں"
    assert clean_for_tts(text) == "جی ہاں۔ آپ کا آرڈر تیار ہے۔ پہلا قدم، پانی ابالیں۔"


def test_removes_urls_and_latin_but_keeps_decimals() -> None:
    assert clean_for_tts("دیکھیں https://x.com یہاں OK قیمت 3.5 ہے.") == "دیکھیں یہاں قیمت 3.5 ہے۔"


def test_keep_latin_option() -> None:
    assert "OK" in clean_for_tts("ٹھیک ہے OK", keep_latin=True)


def test_arabic_letters_become_urdu() -> None:
    assert normalize_urdu("كيا") == "کیا"


def test_adds_final_full_stop_and_collapses_marks() -> None:
    assert clean_for_tts("سلام") == "سلام۔"
    assert clean_for_tts("کیا حال ہے؟ ۔") == "کیا حال ہے؟"


def test_split_sentences() -> None:
    assert split_sentences("جی ہاں۔ آپ کیسے ہیں؟ ٹھیک ہوں۔") == [
        "جی ہاں۔",
        "آپ کیسے ہیں؟",
        "ٹھیک ہوں۔",
    ]


def test_empty_text() -> None:
    assert clean_for_tts("😊 **") == ""
    assert split_sentences("") == []
