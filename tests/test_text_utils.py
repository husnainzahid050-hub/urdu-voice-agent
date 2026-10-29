import pytest

from src.text_utils import (
    clean_for_tts,
    digits_to_urdu,
    normalize_urdu,
    number_to_urdu,
    numbers_to_words,
    split_sentences,
)


def test_removes_markdown_emojis_and_bullets() -> None:
    text = "**جی ہاں!** آپ کا آرڈر 😊 تیار ہے۔\n- پہلا قدم: پانی ابالیں"
    assert clean_for_tts(text) == "جی ہاں۔ آپ کا آرڈر تیار ہے۔ پہلا قدم، پانی ابالیں۔"


def test_removes_urls_and_latin_and_speaks_decimals() -> None:
    assert (
        clean_for_tts("دیکھیں https://x.com یہاں OK قیمت 3.5 ہے.")
        == "دیکھیں یہاں قیمت تین اعشاریہ پانچ ہے۔"
    )


@pytest.mark.parametrize(
    ("n", "words"),
    [
        (0, "صفر"),
        (7, "سات"),
        (19, "انیس"),
        (34, "چونتیس"),
        (79, "اناسی"),
        (99, "ننانوے"),
        (100, "ایک سو"),
        (250, "دو سو پچاس"),
        (1234, "ایک ہزار دو سو چونتیس"),
        (15000, "پندرہ ہزار"),
        (250000, "دو لاکھ پچاس ہزار"),
        (12345678, "ایک کروڑ تئیس لاکھ پینتالیس ہزار چھ سو اٹھتر"),
    ],
)
def test_number_to_urdu(n: int, words: str) -> None:
    assert number_to_urdu(n) == words


def test_phone_numbers_are_read_digit_by_digit() -> None:
    assert digits_to_urdu("0300") == "صفر تین صفر صفر"
    assert "صفر تین صفر صفر ایک دو تین چار پانچ چھ سات" in numbers_to_words("فون 0300-1234567")


def test_amounts_percent_and_thousands_commas() -> None:
    assert numbers_to_words("500 روپے") == "پانچ سو روپے"
    assert numbers_to_words("Rs. 1,500") == "ایک ہزار پانچ سو روپے"
    assert numbers_to_words("20% رعایت") == "بیس فیصد رعایت"


def test_clean_for_tts_has_no_digits_left() -> None:
    out = clean_for_tts("آرڈر نمبر 1234 کی قیمت 2,500 روپے ہے، فون 03001234567 پر کال کریں۔")
    assert not any(ch.isdigit() for ch in out)
    assert "ایک ہزار دو سو چونتیس" in out and "دو ہزار پانچ سو روپے" in out


def test_keep_latin_option() -> None:
    assert "OK" in clean_for_tts("ٹھیک ہے OK", keep_latin=True)


def test_arabic_letters_become_urdu() -> None:
    assert normalize_urdu("كيا") == "کیا"


def test_letters_and_digits_the_voice_cannot_say_are_mapped() -> None:
    assert normalize_urdu("زکوٰۃ") == "زکوٰت"
    assert normalize_urdu("گئۓ") == "گئئے"
    assert normalize_urdu("۳۵۰") == "350"


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
