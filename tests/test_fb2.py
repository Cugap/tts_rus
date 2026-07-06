from __future__ import annotations

import pytest
from pathlib import Path

from app.text_processing import (
    Fb2ValidationError,
    load_chapters,
    load_fb2_chapters,
    validate_fb2,
)


def test_validate_fb2_accepts_sample() -> None:
    path = Path(__file__).resolve().parent.parent / "book_example" / "sample.fb2"
    result = validate_fb2(path)

    assert result.ok is True
    assert result.issues == []
    assert result.stats["chapters_with_text"] == 2


def test_load_fb2_chapters_from_sample() -> None:
    path = Path(__file__).resolve().parent.parent / "book_example" / "sample.fb2"
    chapters = load_fb2_chapters(path)

    assert len(chapters) == 2
    assert chapters[0].title == "Глава 1"
    assert "первая глава" in chapters[0].text
    assert "Вторая строка первой главы" in chapters[0].text
    assert chapters[1].title == "Глава 2"
    assert "вторая глава" in chapters[1].text


def test_validate_fb2_rejects_broken_xml(tmp_path: Path) -> None:
    broken = tmp_path / "broken.fb2"
    broken.write_text("<FictionBook><body>", encoding="utf-8")

    result = validate_fb2(broken)

    assert result.ok is False
    assert result.issues[0].code == "xml_parse"


def test_validate_fb2_rejects_not_fb2(tmp_path: Path) -> None:
    not_fb2 = tmp_path / "book.fb2"
    not_fb2.write_text("<html><body>text</body></html>", encoding="utf-8")

    result = validate_fb2(not_fb2)

    assert result.ok is False
    assert result.issues[0].code == "not_fb2"


def test_validate_fb2_rejects_image_only(tmp_path: Path) -> None:
    image_only = tmp_path / "images.fb2"
    image_only.write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
<FictionBook xmlns="http://www.gribuser.ru/xml/fictionbook/2.0"
             xmlns:l="http://www.w3.org/1999/xlink">
  <body>
    <section>
      <image l:href="#img1"/>
    </section>
  </body>
  <binary id="img1" content-type="image/jpeg">AAAA</binary>
</FictionBook>
""",
        encoding="utf-8",
    )

    result = validate_fb2(image_only)

    assert result.ok is False
    assert any(issue.code == "image_only" for issue in result.issues)


def test_validate_fb2_rejects_too_short(tmp_path: Path) -> None:
    short = tmp_path / "short.fb2"
    short.write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
<FictionBook xmlns="http://www.gribuser.ru/xml/fictionbook/2.0">
  <body>
    <section>
      <title><p>Глава 1</p></title>
      <p>Коротко.</p>
    </section>
  </body>
</FictionBook>
""",
        encoding="utf-8",
    )

    result = validate_fb2(short)

    assert result.ok is False
    assert any(issue.code in {"too_short", "short_chapters"} for issue in result.issues)


def test_validate_fb2_warns_low_cyrillic_ratio(tmp_path: Path) -> None:
    low = tmp_path / "low_cyrillic.fb2"
    low.write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
<FictionBook xmlns="http://www.gribuser.ru/xml/fictionbook/2.0">
  <body>
    <section>
      <title><p>Chapter 1</p></title>
      <p>Hello world! This is English text. No Russian here at all.</p>
      <p>More English words and sentences for testing purposes only.</p>
      <p>12345 67890 special chars and symbols test text.</p>
    </section>
  </body>
</FictionBook>
""",
        encoding="utf-8",
    )

    result = validate_fb2(low)

    codes = {issue.code for issue in result.issues}
    assert "low_cyrillic_ratio" in codes


def test_validate_fb2_warns_long_paragraphs(tmp_path: Path) -> None:
    long_p = tmp_path / "long_paragraphs.fb2"
    long_text = "Привет. " * 2000  # ~16000 chars, well over limit
    long_p.write_text(
        f"""<?xml version="1.0" encoding="UTF-8"?>
<FictionBook xmlns="http://www.gribuser.ru/xml/fictionbook/2.0">
  <body>
    <section>
      <title><p>Глава с длинным абзацем</p></title>
      <p>{long_text}</p>
    </section>
  </body>
</FictionBook>
""",
        encoding="utf-8",
    )

    result = validate_fb2(long_p)

    codes = {issue.code for issue in result.issues}
    assert "long_paragraphs" in codes
    assert result.stats["max_paragraph_len"] > 3000


def test_validate_fb2_warns_long_chapter_title(tmp_path: Path) -> None:
    long_title = tmp_path / "long_title.fb2"
    long_title.write_text(
        f"""<?xml version="1.0" encoding="UTF-8"?>
<FictionBook xmlns="http://www.gribuser.ru/xml/fictionbook/2.0">
  <body>
    <section>
      <title><p>{'Очень длинное название главы которое явно превышает лимит ' * 5}</p></title>
      <p>Немного текста для проверки предупреждения о длинном заголовке.</p>
    </section>
  </body>
</FictionBook>
""",
        encoding="utf-8",
    )

    result = validate_fb2(long_title)

    codes = {issue.code for issue in result.issues}
    assert "long_chapter_title" in codes


def test_validate_fb2_stats_have_new_fields(tmp_path: Path) -> None:
    path = tmp_path / "stats_test.fb2"
    path.write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
<FictionBook xmlns="http://www.gribuser.ru/xml/fictionbook/2.0">
  <body>
    <section>
      <title><p>Глава 1</p></title>
      <p>Это первая глава. Тут немного текста для синтеза.</p>
      <p>Вторая строка первой главы.</p>
    </section>
    <section>
      <title><p>Глава 2</p></title>
      <p>Это вторая глава. Еще немного текста.</p>
    </section>
  </body>
</FictionBook>
""",
        encoding="utf-8",
    )

    result = validate_fb2(path)

    assert result.ok is True
    assert "max_paragraph_len" in result.stats
    assert "max_chapter_title_len" in result.stats
    assert "letter_chars" in result.stats
    assert "replacement_chars" in result.stats
    assert result.stats["chapters_with_text"] == 2


def test_load_fb2_chapters_raises_on_invalid(tmp_path: Path) -> None:
    broken = tmp_path / "broken.fb2"
    broken.write_text("<FictionBook>", encoding="utf-8")

    with pytest.raises(Fb2ValidationError) as exc_info:
        load_fb2_chapters(broken)

    assert exc_info.value.issues


def test_load_chapters_dispatches_by_extension(tmp_path: Path) -> None:
    fb2_path = Path(__file__).resolve().parent.parent / "book_example" / "sample.fb2"
    txt_path = tmp_path / "book.txt"
    txt_path.write_text(
        "Глава 1\nТекст первой главы.\n\nГлава 2\nТекст второй главы.\n",
        encoding="utf-8",
    )

    fb2_chapters = load_chapters(fb2_path)
    txt_chapters = load_chapters(txt_path)

    assert len(fb2_chapters) == 2
    assert len(txt_chapters) == 2
