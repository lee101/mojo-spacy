"""Benchmarks against spaCy 3.8 and NumPy on identical inputs."""

from __future__ import annotations

import math
import platform
import time

import numpy as np
import spacy
from spacy.matcher import Matcher as SpacyMatcher
from spacy.vectors import Vectors as SpacyVectors

import mojospacy
from mojospacy.matcher import Matcher


def timeit(function, repeat=3):
    function()
    best = math.inf
    for _ in range(repeat):
        start = time.perf_counter()
        function()
        best = min(best, time.perf_counter() - start)
    return best


def cpu_name():
    try:
        with open("/proc/cpuinfo", encoding="utf8") as file:
            for line in file:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown CPU"


def main():
    rng = np.random.default_rng(7)
    cases = []

    ours_nlp = mojospacy.blank("en")
    spacy_nlp = spacy.blank("en")
    ours_nlp.max_length = 2_000_000
    spacy_nlp.max_length = 2_000_000
    text = (
        "I can't email test@example.com, so visit https://example.com/a-b?q=1. "
        "New York-based teams paid $10.50 -- really! "
    ) * 10_000
    assert [t.text for t in ours_nlp(text)] == [t.text for t in spacy_nlp(text)]
    cases.append(
        ("Tokenizer (1.2M chars)", lambda: ours_nlp(text), lambda: spacy_nlp(text), "spaCy")
    )

    match_text = "red car blue bike one two x 3 " * 8_000
    ours_doc = ours_nlp(match_text)
    spacy_doc = spacy_nlp(match_text)
    rules = [
        ("COLOR", [[{"LOWER": {"IN": ["red", "blue"]}}]]),
        ("THING", [[{"LOWER": {"IN": ["red", "blue"]}}, {"IS_ALPHA": True}]]),
        ("NUMBER_RUN", [[{"LIKE_NUM": True, "OP": "+"}]]),
    ]
    ours_matcher = Matcher(ours_nlp.vocab)
    spacy_matcher = SpacyMatcher(spacy_nlp.vocab)
    for key, patterns in rules:
        ours_matcher.add(key, patterns)
        spacy_matcher.add(key, patterns)
    assert len(ours_matcher(ours_doc)) == len(spacy_matcher(spacy_doc))
    cases.append(
        (
            "Matcher (64k tokens, 3 rules)",
            lambda: ours_matcher(ours_doc),
            lambda: spacy_matcher(spacy_doc),
            "spaCy",
        )
    )

    first = rng.normal(size=4_000_000).astype(np.float32)
    second = rng.normal(size=4_000_000).astype(np.float32)

    def numpy_cosine():
        return float(np.dot(first, second) / np.sqrt(np.dot(first, first) * np.dot(second, second)))

    assert close(mojospacy.cosine_similarity(first, second), numpy_cosine(), 3e-6)
    cases.append(
        (
            "Cosine similarity (4M dims)",
            lambda: mojospacy.cosine_similarity(first, second),
            numpy_cosine,
            "NumPy",
        )
    )

    matrix = rng.normal(size=(100_000, 64)).astype(np.float32)

    def numpy_normalize():
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return matrix / np.maximum(norms, np.finfo(np.float32).tiny)

    assert np.allclose(mojospacy.normalize_vectors(matrix), numpy_normalize(), atol=2e-6)
    cases.append(
        (
            "Normalize (100k x 64)",
            lambda: mojospacy.normalize_vectors(matrix),
            numpy_normalize,
            "NumPy",
        )
    )

    data = rng.normal(size=(20_000, 64)).astype(np.float32)
    keys = [f"key-{i}" for i in range(len(data))]
    ours_vectors = mojospacy.Vectors(data=data, keys=keys)
    spacy_vectors = SpacyVectors(data=data.copy(), keys=keys)
    queries = rng.normal(size=(16, 64)).astype(np.float32)
    ours_result = ours_vectors.most_similar(queries, n=10)
    spacy_result = spacy_vectors.most_similar(queries, n=10)
    assert np.array_equal(ours_result[1], spacy_result[1])
    assert np.allclose(ours_result[2], spacy_result[2], atol=2e-4)
    cases.append(
        (
            "Vectors.most_similar (20k x 64, q=16)",
            lambda: ours_vectors.most_similar(queries, n=10),
            lambda: spacy_vectors.most_similar(queries, n=10),
            "spaCy",
        )
    )

    print(f"Machine: {cpu_name()}; {platform.system()} {platform.release()}; Python {platform.python_version()}")
    print()
    print("| case | mojo-spacy | reference | ratio | result |")
    print("| --- | ---: | ---: | ---: | --- |")
    for name, ours, reference, reference_name in cases:
        ours_time = timeit(ours)
        reference_time = timeit(reference)
        ratio = reference_time / ours_time
        result = "faster" if ratio >= 1 else "slower"
        print(
            f"| {name} | {ours_time * 1000:.2f} ms | {reference_time * 1000:.2f} ms "
            f"({reference_name}) | {ratio:.2f}x | {result} |"
        )


def close(actual, expected, tolerance):
    return abs(actual - expected) <= tolerance


if __name__ == "__main__":
    main()
