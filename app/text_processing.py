from __future__ import annotations

import re
import threading
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

try:
    from razdel import sentenize
except ImportError:
    sentenize = None

try:
    from num2words import num2words
except ImportError:
    num2words = None

_accentizer = None
_accentizer_lock = threading.Lock()


def get_accentizer():
    global _accentizer
    if _accentizer is None:
        with _accentizer_lock:
            if _accentizer is None:
                try:
                    from ruaccent import RUAccent
                    import numpy as np

                    _accentizer = RUAccent()
                    _accentizer.load(omograph_model_size="turbo", use_dictionary=True)

                    # Monkeypatch accent_model to provide token_type_ids for newer transformers
                    def put_accent_patched(word):
                        lower_word = word.lower()
                        inputs = _accentizer.accent_model.tokenizer(
                            lower_word, return_tensors="np"
                        )
                        if "token_type_ids" not in inputs:
                            inputs["token_type_ids"] = np.zeros_like(inputs["input_ids"])
                        inputs = {k: v.astype(np.int64) for k, v in inputs.items()}
                        outputs = _accentizer.accent_model.session.run(None, inputs)
                        output_names = {
                            output_key.name: idx
                            for idx, output_key in enumerate(
                                _accentizer.accent_model.session.get_outputs()
                            )
                        }
                        logits = outputs[output_names["logits"]]

                        e_x = np.exp(logits - np.max(logits))
                        probabilities = e_x / e_x.sum(axis=-1, keepdims=True)

                        scores = np.max(probabilities, axis=-1)[0]
                        labels = np.argmax(logits, axis=-1)[0]
                        pred_with_scores = [
                            {
                                "label": _accentizer.accent_model.id2label[str(label)],
                                "score": float(score),
                            }
                            for label, score in zip(labels, scores)
                        ]

                        stressed_word = _accentizer.accent_model.render_stress(
                            word, pred_with_scores
                        )
                        return stressed_word

                    _accentizer.accent_model.put_accent = put_accent_patched

                    # Monkeypatch stress_usage_predictor as well
                    def predict_stress_usage_patched(text):
                        inputs = _accentizer.stress_usage_predictor.tokenizer(
                            text,
                            return_offsets_mapping=True,
                            return_special_tokens_mask=True,
                            return_tensors="np",
                        )
                        offset_mapping, special_tokens_mask, input_ids = (
                            inputs.pop("offset_mapping")[0],
                            inputs.pop("special_tokens_mask")[0],
                            inputs["input_ids"][0],
                        )
                        if "token_type_ids" not in inputs:
                            inputs["token_type_ids"] = np.zeros_like(inputs["input_ids"])
                        inputs = {k: v.astype(np.int64) for k, v in inputs.items()}
                        outputs = _accentizer.stress_usage_predictor.session.run(None, inputs)
                        logits = outputs[0]
                        maxes = np.max(logits, axis=-1, keepdims=True)
                        shifted_exp = np.exp(logits - maxes)
                        scores = shifted_exp / shifted_exp.sum(axis=-1, keepdims=True)
                        pre_entities = _accentizer.stress_usage_predictor.collect_pre_entities(
                            text, input_ids, scores[0], offset_mapping, special_tokens_mask
                        )
                        grouped_entities = _accentizer.stress_usage_predictor.aggregate_words(
                            pre_entities, "AVERAGE"
                        )
                        return grouped_entities

                    _accentizer.stress_usage_predictor.predict_stress_usage = (
                        predict_stress_usage_patched
                    )

                except ImportError:
                    _accentizer = False
    return _accentizer


def normalize_numbers(text: str) -> str:
    if num2words is None:
        return text

    def replace_num(match):
        num_str = match.group(0)
        try:
            return num2words(int(num_str), lang="ru")
        except Exception:
            return num_str

    return re.sub(r"\b\d+\b", replace_num, text)


def normalize_text(text: str) -> str:
    text = normalize_numbers(text)
    acc = get_accentizer()
    if acc:
        text = acc.process_all(text)
    return text


SUPPORTED_BOOK_EXTENSIONS = {".txt", ".fb2"}
FB2_NS = "http://www.gribuser.ru/xml/fictionbook/2.0"
FB2_SKIP_BODY_NAMES = frozenset({"notes", "comments"})
FB2_MIN_TOTAL_CHARS = 50
FB2_MIN_CHAPTER_CHARS = 20
FB2_MIN_LETTER_RATIO = 0.25
FB2_MAX_REPLACEMENT_CHAR_RATIO = 0.02
FB2_TEXT_BLOCK_TAGS = frozenset(
    {"p", "v", "subtitle", "text-author", "cite", "date"}
)
FB2_CONTAINER_TAGS = frozenset({"poem", "stanza", "epigraph", "section", "title"})
MAX_CHAPTER_TITLE_LENGTH = 60

# Additional validation thresholds
FB2_MAX_PARAGRAPH_LENGTH = 3000
FB2_MIN_CYRILLIC_RATIO = 0.50
FB2_MAX_NON_ALPHA_RATIO = 0.80
FB2_MAX_CHAPTER_TITLE_LENGTH = 120
CYRILLIC_PATTERN = re.compile(r"[а-яА-ЯёЁ]")

CHAPTER_PATTERN = re.compile(
    rf"(?im)^\s*(?:глава|chapter|часть)\s+([0-9ivxlcdm]+|[^\n]{{1,{MAX_CHAPTER_TITLE_LENGTH}}})\s*$"
)


@dataclass(slots=True)
class Fb2ValidationIssue:
    code: str
    message: str


@dataclass(slots=True)
class Fb2ValidationResult:
    ok: bool
    issues: list[Fb2ValidationIssue]
    stats: dict[str, int | str | None]


class Fb2ValidationError(ValueError):
    def __init__(self, issues: list[Fb2ValidationIssue]):
        self.issues = issues
        super().__init__(format_fb2_validation_issues(issues))


@dataclass(slots=True)
class Fb2Document:
    root: ET.Element
    namespace: str

    def tag(self, local: str) -> str:
        return f"{{{self.namespace}}}{local}"


@dataclass(slots=True)
class _Fb2Analysis:
    total_chars: int = 0
    letter_chars: int = 0
    cyrillic_chars: int = 0
    replacement_chars: int = 0
    non_alpha_chars: int = 0
    paragraphs: int = 0
    long_paragraphs: int = 0
    max_paragraph_len: int = 0
    max_chapter_title_len: int = 0
    images: int = 0
    tables: int = 0
    binaries: int = 0
    sections: int = 0
    chapters_with_text: int = 0
    empty_sections: int = 0
    short_chapters: list[str] | None = None

    def __post_init__(self) -> None:
        if self.short_chapters is None:
            self.short_chapters = []


def format_fb2_validation_issues(issues: list[Fb2ValidationIssue]) -> str:
    return "\n".join(issue.message for issue in issues)


@dataclass(slots=True)
class Chapter:
    number: int
    title: str
    text: str


def load_text_file(path: Path) -> str:
    for encoding in ("utf-8", "utf-8-sig", "cp1251", "koi8-r"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="ignore")


def _fb2_tag(local: str, namespace: str = FB2_NS) -> str:
    return f"{{{namespace}}}{local}"


def _fb2_local_tag(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def _fb2_detect_namespace(root: ET.Element) -> str | None:
    if _fb2_local_tag(root.tag) != "FictionBook":
        return None
    if "}" in root.tag:
        return root.tag.rsplit("}", 1)[0][1:]
    return FB2_NS


def _parse_fb2(source: Path | bytes) -> Fb2Document:
    try:
        if isinstance(source, Path):
            root = ET.parse(source).getroot()
        else:
            root = ET.fromstring(source)
    except ET.ParseError as err:
        raise Fb2ValidationError(
            [
                Fb2ValidationIssue(
                    code="xml_parse",
                    message=f"Файл повреждён или не является корректным XML: {err}",
                )
            ]
        ) from err

    namespace = _fb2_detect_namespace(root)
    if namespace is None:
        raise Fb2ValidationError(
            [
                Fb2ValidationIssue(
                    code="not_fb2",
                    message="Файл не является FB2: отсутствует корневой элемент FictionBook.",
                )
            ]
        )
    return Fb2Document(root=root, namespace=namespace)


def _fb2_main_bodies(document: Fb2Document) -> list[ET.Element]:
    bodies = document.root.findall(f".//{document.tag('body')}")
    main_bodies = [
        body
        for body in bodies
        if (body.get("name") or "").strip().lower() not in FB2_SKIP_BODY_NAMES
    ]
    if not main_bodies and bodies:
        main_bodies = [bodies[0]]
    return main_bodies


def _fb2_inline_text(element: ET.Element) -> str:
    return " ".join(part.strip() for part in element.itertext() if part and part.strip())


def _fb2_section_title(section: ET.Element) -> str:
    title_el = section.find(_fb2_tag("title", _fb2_namespace_from_element(section)))
    if title_el is None:
        for child in section:
            if _fb2_local_tag(child.tag) == "title":
                title_el = child
                break
    if title_el is None:
        return ""
    return _fb2_inline_text(title_el)


def _fb2_namespace_from_element(element: ET.Element) -> str:
    if "}" in element.tag:
        return element.tag.rsplit("}", 1)[0][1:]
    return FB2_NS


def _fb2_collect_blocks(section: ET.Element) -> list[str]:
    blocks: list[str] = []
    for child in section:
        local = _fb2_local_tag(child.tag)
        if local == "title":
            continue
        if local == "section":
            blocks.extend(_fb2_collect_blocks(child))
        elif local in {"p", "v", "subtitle", "text-author", "cite", "date"}:
            text = _fb2_inline_text(child)
            if text:
                blocks.append(text)
        elif local in {"poem", "stanza", "epigraph"}:
            blocks.extend(_fb2_collect_blocks(child))
        elif local == "empty-line":
            blocks.append("")
    return blocks


def _fb2_section_text(section: ET.Element) -> str:
    parts: list[str] = []
    for block in _fb2_collect_blocks(section):
        if block:
            parts.append(block)
        elif parts and parts[-1]:
            parts.append("")
    return "\n\n".join(parts).strip()


def _fb2_analyze(document: Fb2Document) -> _Fb2Analysis:
    analysis = _Fb2Analysis()
    main_bodies = _fb2_main_bodies(document)
    if not main_bodies:
        return analysis

    for body in main_bodies:
        for elem in body.iter():
            local = _fb2_local_tag(elem.tag)
            if local in FB2_TEXT_BLOCK_TAGS:
                text = _fb2_inline_text(elem)
                if text:
                    text_len = len(text)
                    analysis.paragraphs += 1
                    analysis.total_chars += text_len
                    analysis.letter_chars += sum(1 for char in text if char.isalpha())
                    analysis.cyrillic_chars += len(CYRILLIC_PATTERN.findall(text))
                    analysis.replacement_chars += text.count("\ufffd")
                    analysis.non_alpha_chars += sum(1 for char in text if not char.isalpha())
                    if text_len > FB2_MAX_PARAGRAPH_LENGTH:
                        analysis.long_paragraphs += 1
                    if text_len > analysis.max_paragraph_len:
                        analysis.max_paragraph_len = text_len
            elif local == "image":
                analysis.images += 1
            elif local == "table":
                analysis.tables += 1
            elif local == "binary":
                analysis.binaries += 1
            elif local == "section":
                analysis.sections += 1

    for body in main_bodies:
        sections = body.findall(f"./{document.tag('section')}")
        if sections:
            for section in sections:
                text = _fb2_section_text(section)
                title = _fb2_section_title(section) or f"Глава {analysis.chapters_with_text + 1}"
                title_len = len(title)
                if title_len > analysis.max_chapter_title_len:
                    analysis.max_chapter_title_len = title_len
                if not text:
                    analysis.empty_sections += 1
                    continue
                if len(text) < FB2_MIN_CHAPTER_CHARS:
                    analysis.short_chapters.append(title)
                analysis.chapters_with_text += 1
            continue

        text = _fb2_section_text(body)
        if text:
            if len(text) < FB2_MIN_CHAPTER_CHARS:
                analysis.short_chapters.append("Глава 1")
            analysis.chapters_with_text = 1

    return analysis


def validate_fb2(source: Path | bytes) -> Fb2ValidationResult:
    try:
        document = _parse_fb2(source)
    except Fb2ValidationError as err:
        return Fb2ValidationResult(ok=False, issues=err.issues, stats={})
    return validate_fb2_document(document)


def validate_fb2_document(document: Fb2Document) -> Fb2ValidationResult:
    issues: list[Fb2ValidationIssue] = []
    main_bodies = _fb2_main_bodies(document)
    if not main_bodies:
        issues.append(
            Fb2ValidationIssue(
                code="no_body",
                message="В файле нет основного текста: отсутствует элемент body.",
            )
        )
        return Fb2ValidationResult(ok=False, issues=issues, stats={})

    analysis = _fb2_analyze(document)
    stats: dict[str, int | str | None] = {
        "total_chars": analysis.total_chars,
        "paragraphs": analysis.paragraphs,
        "chapters_with_text": analysis.chapters_with_text,
        "images": analysis.images,
        "tables": analysis.tables,
        "sections": analysis.sections,
        "empty_sections": analysis.empty_sections,
        "letter_chars": analysis.letter_chars,
        "replacement_chars": analysis.replacement_chars,
        "max_paragraph_len": analysis.max_paragraph_len,
        "max_chapter_title_len": analysis.max_chapter_title_len,
    }

    if analysis.total_chars == 0:
        if analysis.images > 0:
            issues.append(
                Fb2ValidationIssue(
                    code="image_only",
                    message=(
                        "Книга состоит только из изображений: текст для озвучки не найден. "
                        "Сканированные FB2 без текстового слоя озвучить нельзя."
                    ),
                )
            )
        elif analysis.tables > 0:
            issues.append(
                Fb2ValidationIssue(
                    code="table_only",
                    message="В файле есть таблицы, но нет текста для озвучки.",
                )
            )
        else:
            issues.append(
                Fb2ValidationIssue(
                    code="no_text",
                    message="В FB2 нет текста для озвучки.",
                )
            )
    elif analysis.total_chars < FB2_MIN_TOTAL_CHARS:
        issues.append(
            Fb2ValidationIssue(
                code="too_short",
                message=(
                    f"Слишком мало текста для озвучки: {analysis.total_chars} символов "
                    f"(минимум {FB2_MIN_TOTAL_CHARS})."
                ),
            )
        )

    if analysis.chapters_with_text == 0 and analysis.total_chars > 0:
        issues.append(
            Fb2ValidationIssue(
                code="empty_chapters",
                message="Все главы пустые: текст не удалось извлечь из секций.",
            )
        )

    if analysis.total_chars > 0:
        letter_ratio = analysis.letter_chars / analysis.total_chars
        if letter_ratio < FB2_MIN_LETTER_RATIO:
            issues.append(
                Fb2ValidationIssue(
                    code="low_letter_ratio",
                    message=(
                        "Текст содержит слишком мало букв — возможно, файл повреждён "
                        "или сохранён в неподдерживаемой кодировке."
                    ),
                )
            )

        replacement_ratio = analysis.replacement_chars / analysis.total_chars
        if replacement_ratio > FB2_MAX_REPLACEMENT_CHAR_RATIO:
            issues.append(
                Fb2ValidationIssue(
                    code="encoding",
                    message=(
                        "В тексте много символов замены (�): вероятна проблема с кодировкой FB2."
                    ),
                )
            )

    if (
        analysis.chapters_with_text > 0
        and analysis.short_chapters
        and len(analysis.short_chapters) == analysis.chapters_with_text
    ):
        preview = ", ".join(analysis.short_chapters[:5])
        suffix = "..." if len(analysis.short_chapters) > 5 else ""
        issues.append(
            Fb2ValidationIssue(
                code="short_chapters",
                message=(
                    f"Все главы слишком короткие для озвучки: {preview}{suffix} "
                    f"(минимум {FB2_MIN_CHAPTER_CHARS} символов в главе)."
                ),
            )
        )

    if analysis.images > 0 and analysis.paragraphs == 0:
        issues.append(
            Fb2ValidationIssue(
                code="image_only",
                message=(
                    "В файле есть изображения, но нет текстовых абзацев для озвучки."
                ),
            )
        )

    # ── Дополнительные проверки качества ──────────────────────────────────

    if analysis.total_chars > 0:
        cyrillic_ratio = analysis.cyrillic_chars / analysis.total_chars
        if cyrillic_ratio < FB2_MIN_CYRILLIC_RATIO:
            issues.append(
                Fb2ValidationIssue(
                    code="low_cyrillic_ratio",
                    message=(
                        f"Мало кириллических символов ({cyrillic_ratio:.1%}): "
                        f"текст может содержать много не-Russian символов, "
                        f"что может ухудшить качество синтеза русской речи."
                    ),
                )
            )

        non_alpha_ratio = analysis.non_alpha_chars / analysis.total_chars
        if non_alpha_ratio > FB2_MAX_NON_ALPHA_RATIO:
            issues.append(
                Fb2ValidationIssue(
                    code="high_non_alpha_ratio",
                    message=(
                        f"Много небуквенных символов ({non_alpha_ratio:.1%}): "
                        f"возможно, файл содержит много цифр,符号 или спецсимволов, "
                        f"что может привести к артефактам при синтезе."
                    ),
                )
            )

    if analysis.long_paragraphs > 0:
        issues.append(
            Fb2ValidationIssue(
                code="long_paragraphs",
                message=(
                    f"Обнаружено {analysis.long_paragraphs} абзацев длиннее "
                    f"{FB2_MAX_PARAGRAPH_LENGTH} символов (макс. {analysis.max_paragraph_len}). "
                    f"Очень длинные абзацы могут быть обрезаны при синтезе."
                ),
            )
        )

    if analysis.max_chapter_title_len > FB2_MAX_CHAPTER_TITLE_LENGTH:
        issues.append(
            Fb2ValidationIssue(
                code="long_chapter_title",
                message=(
                    f"Самое длинное название главы — {analysis.max_chapter_title_len} символов "
                    f"(рекомендуется не более {FB2_MAX_CHAPTER_TITLE_LENGTH}). "
                    f"Длинные заголовки могут отображаться некорректно."
                ),
            )
        )

    return Fb2ValidationResult(ok=not issues, issues=issues, stats=stats)


def _fb2_chapters_from_document(document: Fb2Document) -> list[Chapter]:
    main_bodies = _fb2_main_bodies(document)
    chapters: list[Chapter] = []
    for body in main_bodies:
        sections = body.findall(f"./{document.tag('section')}")
        if sections:
            for section in sections:
                text = _fb2_section_text(section)
                if not text:
                    continue
                title = _fb2_section_title(section) or f"Глава {len(chapters) + 1}"
                chapters.append(
                    Chapter(number=len(chapters) + 1, title=title, text=text)
                )
            continue

        text = _fb2_section_text(body)
        if text:
            chapters.append(
                Chapter(number=len(chapters) + 1, title="Глава 1", text=text)
            )

    if not chapters:
        raise Fb2ValidationError(
            [
                Fb2ValidationIssue(
                    code="no_text",
                    message="В FB2 нет текста для озвучки.",
                )
            ]
        )
    return chapters


def load_fb2_chapters(path: Path) -> list[Chapter]:
    document = _parse_fb2(path)
    validation = validate_fb2_document(document)
    if not validation.ok:
        raise Fb2ValidationError(validation.issues)
    return _fb2_chapters_from_document(document)


def load_fb2_chapters_from_bytes(raw: bytes) -> list[Chapter]:
    document = _parse_fb2(raw)
    validation = validate_fb2_document(document)
    if not validation.ok:
        raise Fb2ValidationError(validation.issues)
    return _fb2_chapters_from_document(document)


def load_chapters(path: Path) -> list[Chapter]:
    ext = path.suffix.lower()
    if ext == ".fb2":
        return load_fb2_chapters(path)
    if ext == ".txt":
        return split_into_chapters(load_text_file(path))
    raise ValueError(f"Unsupported book format: {ext}")


def split_into_chapters(text: str) -> list[Chapter]:
    matches = list(CHAPTER_PATTERN.finditer(text))
    if not matches:
        stripped = text.strip()
        return [Chapter(number=1, title="Глава 1", text=stripped)] if stripped else []

    chapters: list[Chapter] = []
    for idx, m in enumerate(matches):
        start = m.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        chunk = text[start:end].strip()
        if not chunk:
            continue
        title = m.group(0).strip()
        chapters.append(Chapter(number=len(chapters) + 1, title=title, text=chunk))
    return chapters


def split_text_safely(text: str, max_chars: int) -> list[str]:
    clean = re.sub(r"\s+", " ", text).strip()
    if not clean:
        return []
    if len(clean) <= max_chars:
        return [clean]

    def _split_long_text(long_text: str, limit: int) -> list[str]:
        PUNCTUATION_FALLBACK_RATIO = 0.6
        res = []
        cursor = 0
        while cursor < len(long_text):
            end = min(cursor + limit, len(long_text))
            piece = long_text[cursor:end]
            if end < len(long_text):
                rfind = max(
                    piece.rfind(". "),
                    piece.rfind("! "),
                    piece.rfind("? "),
                    piece.rfind(", "),
                )
                if rfind > int(limit * PUNCTUATION_FALLBACK_RATIO):
                    end = cursor + rfind + 1
                    piece = long_text[cursor:end]
            res.append(piece.strip())
            cursor = end
        return [x for x in res if x]

    if sentenize is not None:
        result = []
        current_chunk = []
        current_len = 0
        for sentence in sentenize(clean):
            s_text = sentence.text

            if len(s_text) > max_chars:
                if current_chunk:
                    result.append(" ".join(current_chunk))
                    current_chunk = []
                    current_len = 0
                result.extend(_split_long_text(s_text, max_chars))
                continue

            if current_len + len(s_text) + 1 > max_chars and current_chunk:
                result.append(" ".join(current_chunk))
                current_chunk = []
                current_len = 0

            current_chunk.append(s_text)
            current_len += len(s_text) + 1

        if current_chunk:
            result.append(" ".join(current_chunk))
        return result

    return _split_long_text(clean, max_chars)
