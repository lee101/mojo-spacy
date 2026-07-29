import pytest

import mojospacy
from mojospacy.matcher import Matcher

spacy = pytest.importorskip("spacy")
UpstreamMatcher = pytest.importorskip("spacy.matcher").Matcher


@pytest.fixture(scope="module")
def docs():
    ours_nlp = mojospacy.blank("en")
    spacy_nlp = spacy.blank("en")
    text = "hello one two x 3 A A b red car and blue bike"
    return ours_nlp(text), spacy_nlp(text)


def normalized(matcher, doc):
    return [(doc.vocab.strings[match_id], start, end) for match_id, start, end in matcher(doc)]


def build_pair(docs, rules):
    ours = Matcher(docs[0].vocab)
    theirs = UpstreamMatcher(docs[1].vocab)
    for key, patterns in rules:
        ours.add(key, patterns)
        theirs.add(key, patterns)
    return ours, theirs


def test_exact_lower_and_boolean_matches(docs):
    rules = [
        ("HELLO", [[{"LOWER": "hello"}]]),
        ("NUM", [[{"LIKE_NUM": True, "OP": "+"}]]),
    ]
    ours, theirs = build_pair(docs, rules)
    assert normalized(ours, docs[0]) == normalized(theirs, docs[1])


@pytest.mark.parametrize("operator", ["?", "*", "+", "!", "{2}", "{1,3}", "{2,}", "{,2}"])
def test_quantifier_parity(operator):
    ours_doc = mojospacy.blank("en")("a a b")
    theirs_doc = spacy.blank("en")("a a b")
    rules = [("X", [[{"LOWER": "a", "OP": operator}]])]
    ours, theirs = build_pair((ours_doc, theirs_doc), rules)
    assert normalized(ours, ours_doc) == normalized(theirs, theirs_doc)


def test_multi_token_pattern_and_order(docs):
    rules = [
        ("RED_THING", [[{"LOWER": "red"}, {"IS_ALPHA": True}]]),
        ("COLOR", [[{"LOWER": {"IN": ["red", "blue"]}}]]),
    ]
    ours, theirs = build_pair(docs, rules)
    assert normalized(ours, docs[0]) == normalized(theirs, docs[1])


def test_token_attributes_are_cached(docs):
    matcher = Matcher(docs[0].vocab)
    matcher.add("HELLO", [[{"LOWER": "hello"}]])
    matcher(docs[0])
    cached = docs[0]._lexical_attrs
    assert cached is not None
    assert matcher(docs[0]) == matcher(docs[0])
    assert docs[0]._lexical_attrs is cached


def test_not_in_predicate(docs):
    rules = [("NOT_COLOR", [[{"LOWER": {"NOT_IN": ["red", "blue"]}, "IS_ALPHA": True}]])]
    ours, theirs = build_pair(docs, rules)
    assert normalized(ours, docs[0]) == normalized(theirs, docs[1])


def test_as_spans(docs):
    rules = [("GREETING", [[{"LOWER": "hello"}]])]
    ours, theirs = build_pair(docs, rules)
    assert [(span.text, span.label_) for span in ours(docs[0], as_spans=True)] == [
        (span.text, span.label_) for span in theirs(docs[1], as_spans=True)
    ]


def test_add_get_remove_and_contains(docs):
    matcher = Matcher(docs[0].vocab)
    patterns = [[{"LOWER": "hello"}]]
    matcher.add("HELLO", patterns)
    assert "HELLO" in matcher
    assert len(matcher) == 1
    assert matcher.get("HELLO") == (None, patterns)
    matcher.remove("HELLO")
    assert "HELLO" not in matcher
    assert len(matcher) == 0


def test_callback_receives_final_matches(docs):
    calls = []

    def callback(matcher, doc, index, matches):
        calls.append((index, len(matches), doc.text))

    matcher = Matcher(docs[0].vocab)
    matcher.add("A", [[{"LOWER": "a"}]], on_match=callback)
    matches = matcher(docs[0])
    assert calls == [(0, len(matches), docs[0].text), (1, len(matches), docs[0].text)]


def test_invalid_attribute_is_rejected(docs):
    matcher = Matcher(docs[0].vocab)
    with pytest.raises(ValueError):
        matcher.add("BAD", [[{"NOT_AN_ATTRIBUTE": True}]])


@pytest.mark.parametrize(
    ("attribute", "value"),
    [
        ("ORTH", "hello"),
        ("TEXT", "hello"),
        ("SHAPE", "xxxx"),
        ("LENGTH", 5),
        ("IS_ALPHA", True),
        ("IS_DIGIT", False),
        ("IS_PUNCT", False),
        ("IS_SPACE", False),
        ("IS_ASCII", True),
    ],
)
def test_documented_attributes_match_spacy(docs, attribute, value):
    rules = [("ATTRIBUTE", [[{attribute: value}]])]
    ours, theirs = build_pair(docs, rules)
    assert normalized(ours, docs[0]) == normalized(theirs, docs[1])


def test_invalid_numeric_attribute_is_rejected_even_without_validation(docs):
    matcher = Matcher(docs[0].vocab, validate=False)
    matcher.add("BAD", [[{999: True}]])
    with pytest.raises(ValueError, match="attribute ID"):
        matcher(docs[0])
