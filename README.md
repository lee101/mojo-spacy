# mojo-spacy

`mojo-spacy` is a standalone Mojo implementation of the compute-heavy parts of
spaCy's tokenizer, rule matcher, and dense vector table. Its Python package is
named `mojospacy`; for the covered subset, importing it as `spacy` keeps the
familiar API:

```python
import numpy as np
import mojospacy as spacy
from mojospacy.matcher import Matcher

nlp = spacy.blank("en")
doc = nlp("I can't visit New York today.")

matcher = Matcher(nlp.vocab)
matcher.add("PLACE", [[{"LOWER": "new"}, {"LOWER": "york"}]])
print([(doc[start:end].text, doc.vocab.strings[match_id])
       for match_id, start, end in matcher(doc)])

vectors = spacy.Vectors(shape=(2, 3))
vectors.add("north", vector=np.array([1, 0, 0], dtype="f"))
vectors.add("east", vector=np.array([0, 1, 0], dtype="f"))
print(vectors.most_similar(np.array([[0.9, 0.1, 0]], dtype="f"), n=1)[1])
```

This prints:

```text
[('New York', 'PLACE')]
[[0]]
```

## Covered subset

| area | implemented API |
| --- | --- |
| Tokenization | `blank`, `Language`, `English`, `Tokenizer`, `Doc`, `Token`, `Span`; English contractions, punctuation, abbreviations, decimals, fractions, email/URL tokens, Unicode word text, exact character offsets and whitespace reconstruction |
| Matching | `Matcher`, `add`, `remove`, `get`, callbacks, `as_spans`, `ORTH`/`TEXT`, `LOWER`, `SHAPE`, `LENGTH`, `IS_ALPHA`, `IS_DIGIT`, `IS_PUNCT`, `IS_SPACE`, `IS_ASCII`, `LIKE_NUM`, `IN`/`NOT_IN`, and `! ? * + {n} {n,m} {n,} {,m}` quantifiers |
| Vectors | `Vectors` construction, `data`, `key2row`, `add`, lookup, assignment, `find`, `resize`, `most_similar`, `cosine_similarity`, `normalize_vectors`, and token/doc/span mean vectors |
| Vocabulary | `Vocab`, `StringStore`, and spaCy-compatible 64-bit string hashes |

The parity suite compares this subset directly with spaCy 3.8.14. It asserts
complete token sequences including offsets and whitespace, matcher IDs and
spans including quantified matches, and vector keys, rows, scores, resizing,
and cosine values.

This is not a full NLP pipeline. It does not include trained models, tagging,
parsing, NER, lemmatization, serialization, language packages other than the
English-oriented default rules, custom tokenizer regex hooks, `REGEX`/`FUZZY`
matcher predicates, annotation-dependent matcher attributes, floret vectors,
or GPU vector storage. The tokenizer intentionally covers productive general
rules and common English exceptions, not spaCy's entire language-specific
exception inventory.

## Install

```bash
pixi install
pixi run build
pixi run test
```

Pixi installs the pinned Mojo nightly, Python, NumPy, pytest, and spaCy used by
the parity tests. `pixi run build` produces
`dist/libmojo-spacy.so`. Importing `mojospacy` also rebuilds a missing or stale
library when a compiler or Pixi is available.

## Performance

Measured with `pixi run bench`, which holds the repository's machine-wide
benchmark lock. These are best-of-three wall-clock measurements from the
final publication gate on an
Intel(R) Xeon(R) CPU E5-2697 v4 @ 2.30GHz, Linux 6.8.0-136-generic, Python
3.13.14:

| case | mojo-spacy | reference | ratio | result |
| --- | ---: | ---: | ---: | --- |
| Tokenizer (1.2M chars) | 419.48 ms | 1940.43 ms (spaCy) | 4.63x | faster |
| Matcher (64k tokens, 3 rules) | 78.46 ms | 153.65 ms (spaCy) | 1.96x | faster |
| Cosine similarity (4M dims) | 1.39 ms | 2.48 ms (NumPy) | 1.78x | faster |
| Normalize (100k x 64) | 6.22 ms | 20.02 ms (NumPy) | 3.22x | faster |
| Vectors.most_similar (20k x 64, q=16) | 7.22 ms | 15.83 ms (spaCy) | 2.19x | faster |

Matcher calls lazily cache the contiguous ten-column lexical attribute matrix
on each `Doc`, avoiding repeated Python property evaluation and allocation.
Cosine computes both squared norms and the cross product in one SIMD pass,
uses a scalar remainder loop, and switches to a four-way CPU reduction only
for vectors of at least eight million elements.
`Vectors.most_similar` fuses each candidate's norm and query dot product into
one SIMD pass with a scalar remainder, and splits independent query batches
across four CPU workers only when at least eight million elements will be
examined.

No GPU path is included. The available hot loops are branch-heavy or stream
vector memory at less than two arithmetic operations per byte, so they do not
have enough arithmetic intensity to justify device transfer and launch costs.

## How it works

All kernels live in one Mojo compilation unit to avoid paying the compiler's
fixed startup cost repeatedly. Python calls its exported C ABI with `ctypes`.
Arrays cross the boundary as 64-bit integer addresses; each export rebuilds
an `UnsafePointer[..., AnyOrigin[mut=True]]` inside Mojo.

The tokenizer encodes input once as contiguous UTF-32, so Mojo emits character
offsets rather than byte offsets. It scans whitespace, contextual punctuation,
URLs, email addresses, abbreviations, decimals, and numeric fractions in one
pass. Python then applies the small contraction-exception layer and constructs
the standalone token objects.

The matcher represents each token as a row-major `int64` attribute matrix.
Patterns become flat criterion/value arrays. Quantifiers are evaluated with
two caller-owned frontier buffers, which enumerate every legal end position
without allocating in Mojo. The lexical matrix is cached after its first use.
Dense vectors are contiguous row-major `float32`; SIMD dot, norm, cosine, and
fused top-k loops operate directly on those buffers. Python owns every
allocation and lifetime.

## Development

```bash
pixi run build
pixi run test
pixi run bench
```

The benchmark task, rather than running `bench/bench.py` directly, is required
for reproducible local measurements.

## License

MIT
