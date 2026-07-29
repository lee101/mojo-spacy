from __future__ import annotations

from .tokenizer import Tokenizer
from .vocab import Vocab


class Language:
    lang = "xx"

    def __init__(self, vocab=True, *, max_length=1_000_000, meta=None, create_tokenizer=None, batch_size=1000, **kwargs):
        del kwargs
        self.vocab = Vocab(lang=self.lang) if vocab is True else vocab
        self.max_length = max_length
        self.meta = dict(meta or {})
        self.batch_size = batch_size
        self.tokenizer = create_tokenizer(self) if create_tokenizer else Tokenizer(self.vocab)
        self.pipeline = []

    def __call__(self, text, *, disable=(), component_cfg=None):
        del disable, component_cfg
        if len(text) > self.max_length:
            raise ValueError(f"text length {len(text)} exceeds max_length {self.max_length}")
        return self.tokenizer(text)

    def make_doc(self, text):
        return self.tokenizer(text)

    def pipe(self, texts, *, as_tuples=False, batch_size=None, disable=(), component_cfg=None, n_process=1):
        del batch_size, disable, component_cfg, n_process
        if as_tuples:
            for text, context in texts:
                yield self(text), context
        else:
            yield from self.tokenizer.pipe(texts)


class English(Language):
    lang = "en"
