import pytest

import mojospacy

spacy = pytest.importorskip("spacy")


@pytest.fixture(scope="module")
def tokenizers():
    return mojospacy.blank("en"), spacy.blank("en")


@pytest.mark.parametrize(
    "text",
    [
        "Hello, world!",
        "I can't believe it's not butter.",
        "U.S.A. paid $10.50 -- really...",
        "New York-based café—open 24/7.",
        "one  two\nthree\t four",
        "(hello) 'quoted' don't won't",
        "Email test@example.com or visit https://example.com/a-b?q=1.",
        "",
    ],
)
def test_token_boundaries_and_whitespace_match_spacy(tokenizers, text):
    ours, theirs = (nlp(text) for nlp in tokenizers)
    assert [(t.text, t.idx, t.whitespace_) for t in ours] == [
        (t.text, t.idx, t.whitespace_) for t in theirs
    ]
    assert "".join(token.text_with_ws for token in ours) == text


def test_lexical_attributes_match_spacy(tokenizers):
    text = "Hello 123 24/7 café -- $ !"
    ours, theirs = (nlp(text) for nlp in tokenizers)
    assert [
        (
            token.lower_,
            token.shape_,
            token.is_alpha,
            token.is_digit,
            token.is_ascii,
            token.is_space,
            token.is_punct,
            token.like_num,
        )
        for token in ours
    ] == [
        (
            token.lower_,
            token.shape_,
            token.is_alpha,
            token.is_digit,
            token.is_ascii,
            token.is_space,
            token.is_punct,
            token.like_num,
        )
        for token in theirs
    ]


def test_doc_index_slice_and_span_text(tokenizers):
    ours, theirs = (nlp("A small test.") for nlp in tokenizers)
    assert ours.text == theirs.text
    assert ours[-1].text == theirs[-1].text
    assert ours[1:3].text == theirs[1:3].text
    assert ours[1:3].text_with_ws == theirs[1:3].text_with_ws
    assert [(t.i, len(t), str(t)) for t in ours] == [
        (t.i, len(t), str(t)) for t in theirs
    ]


def test_pipe_matches_individual_calls(tokenizers):
    texts = ["one test", "two tests", "can't stop"]
    ours = list(tokenizers[0].pipe(texts))
    assert [[t.text for t in doc] for doc in ours] == [
        [t.text for t in tokenizers[1](text)] for text in texts
    ]


def test_blank_language_and_make_doc():
    nlp = mojospacy.blank("en")
    assert isinstance(nlp, mojospacy.English)
    assert isinstance(nlp, mojospacy.Language)
    assert nlp.lang == "en"
    assert [token.text for token in nlp.make_doc("Hello!")] == ["Hello", "!"]


def test_manual_doc_constructor_parity():
    ours = mojospacy.Doc(mojospacy.Vocab(), words=["New", "York"], spaces=[True, False])
    assert ours.text == "New York"
    assert ours[0].whitespace_ == " "
    assert ours[0:2].text == "New York"


def test_hash_string_matches_spacy():
    from spacy.strings import hash_string as upstream_hash

    for text in ("", "a", "hello", "HELLO", "café", "MATCH_ID"):
        assert mojospacy.hash_string(text) == upstream_hash(text)


def test_string_store_round_trip():
    store = mojospacy.StringStore()
    key = store.add("hello")
    assert store[key] == "hello"
    assert store["hello"] == key
    assert "hello" in store
