from __future__ import annotations

import re
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


def get_accentizer():
    global _accentizer
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


MAX_CHAPTER_TITLE_LENGTH = 60
CHAPTER_PATTERN = re.compile(
    rf"(?im)^\s*(?:глава|chapter|часть)\s+([0-9ivxlcdm]+|[^\n]{{1,{MAX_CHAPTER_TITLE_LENGTH}}})\s*$"
)


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
