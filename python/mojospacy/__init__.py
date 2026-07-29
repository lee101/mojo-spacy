"""A standalone Mojo port of spaCy's tokenizer, Matcher, and dense Vectors."""

from __future__ import annotations

from .language import English, Language
from .matcher import Matcher
from .strings import StringStore, hash_string
from .tokenizer import Tokenizer
from .tokens import Doc, Span, Token
from .vectors import Vectors, cosine_similarity, normalize_vectors
from .vocab import Vocab

__version__ = "0.1.0"


def blank(name: str, *, vocab=True, config=None, meta=None):
    del config
    if name == "en":
        return English(vocab=vocab, meta=meta)
    language = type(f"{name.title()}Language", (Language,), {"lang": name})
    return language(vocab=vocab, meta=meta)


__all__ = [
    "Doc",
    "English",
    "Language",
    "Matcher",
    "Span",
    "StringStore",
    "Token",
    "Tokenizer",
    "Vectors",
    "Vocab",
    "blank",
    "cosine_similarity",
    "hash_string",
    "normalize_vectors",
]
