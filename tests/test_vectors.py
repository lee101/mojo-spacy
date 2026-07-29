import numpy as np
import pytest

import mojospacy
from mojospacy import vectors as vector_module

spacy_vectors = pytest.importorskip("spacy.vectors")
spacy_util = pytest.importorskip("spacy.util")


def populated(vector_type):
    vectors = vector_type(shape=(5, 4))
    values = {
        "north": [1.0, 0.0, 0.0, 0.0],
        "near_north": [0.8, 0.2, 0.0, 0.0],
        "east": [0.0, 1.0, 0.0, 0.0],
        "up": [0.05, 0.15, 0.9, 0.0],
    }
    for key, vector in values.items():
        vectors.add(key, vector=np.asarray(vector, dtype=np.float32))
    return vectors


def test_cosine_similarity_parity():
    rng = np.random.default_rng(4)
    first = rng.normal(size=257).astype("f")
    second = rng.normal(size=257).astype("f")
    nlp = pytest.importorskip("spacy").blank("en")
    nlp.vocab.set_vector("first", first)
    nlp.vocab.set_vector("second", second)
    upstream = nlp("first")[0].similarity(nlp("second")[0])
    assert mojospacy.cosine_similarity(first, second) == pytest.approx(upstream, abs=2e-7)
    assert mojospacy.cosine_similarity(np.zeros(3, dtype="f"), np.ones(3, dtype="f")) == 0.0


@pytest.mark.parametrize("size", [1, 3, 7, 8, 9, 31, 33, 257])
def test_cosine_similarity_simd_tails(size):
    rng = np.random.default_rng(size)
    first = rng.normal(size=size).astype(np.float32)
    second = rng.normal(size=size).astype(np.float32)
    expected = float(
        np.dot(first, second)
        / np.sqrt(np.dot(first, first) * np.dot(second, second))
    )
    assert mojospacy.cosine_similarity(first, second) == pytest.approx(expected, abs=3e-7)


def test_cosine_similarity_parallel_threshold(monkeypatch):
    monkeypatch.setattr(vector_module, "_COSINE_PARALLEL_THRESHOLD", 256)
    rng = np.random.default_rng(12)
    first = rng.normal(size=259).astype(np.float32)
    second = rng.normal(size=259).astype(np.float32)
    expected = float(
        np.dot(first, second)
        / np.sqrt(np.dot(first, first) * np.dot(second, second))
    )
    assert mojospacy.cosine_similarity(first, second) == pytest.approx(expected, abs=3e-7)


def test_add_find_and_properties_match_spacy():
    ours = populated(mojospacy.Vectors)
    theirs = populated(spacy_vectors.Vectors)
    assert ours.shape == theirs.shape
    assert ours.n_keys == theirs.n_keys
    assert ours.is_full == theirs.is_full
    assert list(ours.keys()) == list(theirs.keys())
    assert list(ours.values()) == list(theirs.values())
    assert ours.find(keys=["north", "missing"]).tolist() == theirs.find(
        keys=["north", "missing"]
    ).tolist()
    assert ours.find(rows=[0, 1, 2]) .tolist() == theirs.find(rows=[0, 1, 2]).tolist()


def test_most_similar_matches_spacy():
    ours = populated(mojospacy.Vectors)
    theirs = populated(spacy_vectors.Vectors)
    queries = np.asarray([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
    ours_result = ours.most_similar(queries, n=3)
    theirs_result = theirs.most_similar(queries, n=3)
    assert np.array_equal(ours_result[0], theirs_result[0])
    assert np.array_equal(ours_result[1], theirs_result[1])
    assert np.allclose(ours_result[2], theirs_result[2], atol=2e-4)


def test_most_similar_validates_n_and_handles_empty_query_batch():
    vectors = populated(mojospacy.Vectors)
    with pytest.raises(ValueError, match="at least 1"):
        vectors.most_similar(np.ones((1, 4), dtype=np.float32), n=0)
    result = vectors.most_similar(np.empty((0, 4), dtype=np.float32), n=2)
    assert [part.shape for part in result] == [(0, 2), (0, 2), (0, 2)]


def test_normalize_vectors():
    values = np.asarray([[3, 4], [0, 0], [1, -1]], dtype=np.float32)
    result = mojospacy.normalize_vectors(values)
    expected = values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)
    assert np.allclose(result, expected)
    assert np.array_equal(values, np.asarray([[3, 4], [0, 0], [1, -1]], dtype=np.float32))


def test_empty_vector_inputs_do_not_cross_ffi():
    empty_vector = np.asarray([], dtype=np.float32)
    assert mojospacy.cosine_similarity(empty_vector, empty_vector) == 0.0
    empty = np.empty((0, 3), dtype=np.float32)
    assert mojospacy.normalize_vectors(empty).shape == (0, 3)


@pytest.mark.parametrize("dtype", [np.float64, np.int64])
def test_vector_kernels_reject_silent_dtype_narrowing(dtype):
    values = np.ones(3, dtype=dtype)
    with pytest.raises(TypeError, match="must use float32"):
        mojospacy.cosine_similarity(values, values)
    with pytest.raises(TypeError, match="must use float32"):
        mojospacy.normalize_vectors(values)


def test_inplace_normalization_contract():
    values = np.asarray([[3, 4]], dtype=np.float32)
    assert mojospacy.normalize_vectors(values, inplace=True) is values
    assert np.allclose(values, [[0.6, 0.8]])
    with pytest.raises(ValueError, match="C-contiguous float32"):
        mojospacy.normalize_vectors(np.ones((2, 2), dtype=np.float64), inplace=True)
    with pytest.raises(ValueError, match="C-contiguous float32"):
        mojospacy.normalize_vectors(np.ones((2, 2), dtype=np.float32).T, inplace=True)


def test_resize_and_removed_keys_match_spacy():
    ours = populated(mojospacy.Vectors)
    theirs = populated(spacy_vectors.Vectors)
    assert ours.resize((2, 4)) == theirs.resize((2, 4))
    assert np.array_equal(ours.data, theirs.data)
    assert dict(ours.key2row) == dict(theirs.key2row)


def test_get_set_and_shared_rows():
    vectors = populated(mojospacy.Vectors)
    vectors.add("alias", row=0)
    assert np.array_equal(vectors["alias"], vectors["north"])
    vectors["east"] = np.asarray([0, 2, 0, 0], dtype=np.float32)
    assert np.array_equal(vectors["east"], [0, 2, 0, 0])


def test_vocab_vector_access():
    vocab = mojospacy.Vocab()
    vocab.set_vector("hello", np.asarray([1, 2, 3], dtype=np.float32))
    assert vocab.has_vector("hello")
    assert np.array_equal(vocab.get_vector("hello"), [1, 2, 3])
    doc = mojospacy.Doc(vocab, words=["hello"], spaces=[False])
    assert doc[0].has_vector
    assert np.array_equal(doc.vector, [1, 2, 3])
    assert np.array_equal(doc[0:1].vector, [1, 2, 3])
