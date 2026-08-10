from __future__ import annotations

import re
import string

_ARTICLES = re.compile(r"\b(a|an|the)\b")


def _word_to_digit(word: str) -> str | None:
    mapping = {
        "zero": "0",
        "one": "1",
        "two": "2",
        "three": "3",
        "four": "4",
        "five": "5",
        "six": "6",
        "seven": "7",
        "eight": "8",
        "nine": "9",
        "ten": "10",
    }
    return mapping.get(word)


def normalize_answer(s: str) -> str:
    s = s.lower().strip()
    s = s.translate(str.maketrans("", "", string.punctuation))
    s = _ARTICLES.sub(" ", s)
    s = " ".join(s.split())
    tokens = s.split()
    out: list[str] = []
    for t in tokens:
        d = _word_to_digit(t)
        out.append(d if d is not None else t)
    return " ".join(out)


def normalized_set(answers: list[str]) -> set[str]:
    return {normalize_answer(a) for a in answers if a}
