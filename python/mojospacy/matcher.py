"""Token Matcher compatible with spaCy's common rule and quantifier forms."""

from __future__ import annotations

import re

import numpy as np

from . import attrs
from ._lib import addr, lib
from .strings import hash_string
from .tokens import Doc, Span


def _signed(value: int) -> int:
    return value if value < (1 << 63) else value - (1 << 64)


def _quantifier(value):
    if value is None:
        return 1, 1, False
    if value == "!":
        return 1, 1, True
    if value == "?":
        return 0, 1, False
    if value == "*":
        return 0, -1, False
    if value == "+":
        return 1, -1, False
    match = re.fullmatch(r"\{(\d*),?(\d*)\}", str(value))
    if not match:
        raise ValueError(f"Invalid token quantifier: {value!r}")
    left, right = match.groups()
    if "," not in str(value):
        minimum = maximum = int(left)
    else:
        minimum = int(left) if left else 0
        maximum = int(right) if right else -1
    if maximum >= 0 and minimum > maximum:
        raise ValueError(f"Invalid token quantifier range: {value!r}")
    return minimum, maximum, False


class Matcher:
    def __init__(self, vocab, validate=True, fuzzy_compare=None):
        self.vocab = vocab
        self.validate = validate
        self.fuzzy_compare = fuzzy_compare
        self._rules: dict[int, dict] = {}
        self._compiled = None

    def _key(self, key) -> int:
        return self.vocab.strings.add(key) if isinstance(key, str) else int(key)

    def add(self, key, patterns, *, on_match=None, greedy=None):
        if greedy not in (None, "FIRST", "LONGEST"):
            raise ValueError("greedy must be None, 'FIRST', or 'LONGEST'")
        patterns = list(patterns)
        if not patterns or any(not isinstance(pattern, list) or not pattern for pattern in patterns):
            raise ValueError("patterns must contain one or more non-empty token patterns")
        if self.validate:
            for pattern in patterns:
                for spec in pattern:
                    if not isinstance(spec, dict):
                        raise ValueError("each token specification must be a dictionary")
                    for name in spec:
                        if name == "OP":
                            continue
                        if isinstance(name, int):
                            if name not in attrs.IDS.values():
                                raise ValueError(f"Unsupported token attribute ID: {name}")
                        elif str(name).upper() not in attrs.IDS:
                            raise ValueError(f"Unsupported token attribute: {name}")
                    _quantifier(spec.get("OP"))
        match_id = self._key(key)
        if match_id in self._rules:
            self._rules[match_id]["patterns"].extend(patterns)
            self._rules[match_id]["on_match"] = on_match
            self._rules[match_id]["greedy"] = greedy
        else:
            self._rules[match_id] = {
                "patterns": patterns,
                "on_match": on_match,
                "greedy": greedy,
            }
        self._compiled = None

    def remove(self, key):
        match_id = self._key(key)
        if match_id not in self._rules:
            raise KeyError(key)
        del self._rules[match_id]
        self._compiled = None

    def get(self, key, default=None):
        rule = self._rules.get(self._key(key))
        if rule is None:
            return default
        return rule["on_match"], rule["patterns"]

    def __contains__(self, key):
        return self._key(key) in self._rules

    def __len__(self):
        return len(self._rules)

    def _criterion(self, attr_id, value):
        operation = 0
        if isinstance(value, dict):
            if len(value) != 1:
                raise ValueError("attribute predicates must contain exactly one operator")
            operator, value = next(iter(value.items()))
            operator = str(operator).upper()
            if operator == "IN":
                operation = 2
            elif operator == "NOT_IN":
                operation = 3
            elif operator in ("==", "EQ"):
                operation = 0
            elif operator in ("!=", "NEQ"):
                operation = 1
            else:
                raise ValueError(f"Unsupported attribute predicate: {operator}")
        values = value if isinstance(value, (list, tuple, set)) else [value]
        encoded = []
        for item in values:
            if attr_id in (attrs.ORTH, attrs.LOWER, attrs.SHAPE):
                encoded.append(_signed(hash_string(str(item))))
            else:
                encoded.append(int(item))
        return operation, encoded

    def _compile(self):
        pattern_offsets = [0]
        pattern_ids = []
        spec_crit_offsets = [0]
        spec_mins = []
        spec_maxes = []
        spec_negated = []
        crit_attrs = []
        crit_ops = []
        value_offsets = [0]
        values = []
        for match_id, rule in self._rules.items():
            for pattern in rule["patterns"]:
                pattern_ids.append(match_id)
                for spec in pattern:
                    minimum, maximum, negated = _quantifier(spec.get("OP"))
                    spec_mins.append(minimum)
                    spec_maxes.append(maximum)
                    spec_negated.append(int(negated))
                    for name, value in spec.items():
                        if name == "OP":
                            continue
                        attr_id = int(name) if isinstance(name, int) else attrs.IDS[str(name).upper()]
                        if attr_id not in attrs.IDS.values():
                            raise ValueError(f"Unsupported token attribute ID: {attr_id}")
                        operation, encoded = self._criterion(attr_id, value)
                        crit_attrs.append(attr_id)
                        crit_ops.append(operation)
                        values.extend(encoded)
                        value_offsets.append(len(values))
                    spec_crit_offsets.append(len(crit_attrs))
                pattern_offsets.append(len(spec_mins))

        def i64(items, dummy=False):
            if dummy and not items:
                items = [0]
            return np.asarray(items, dtype=np.int64)

        self._compiled = (
            i64(pattern_offsets),
            np.asarray(pattern_ids, dtype=np.uint64),
            i64(spec_crit_offsets),
            i64(spec_mins, True),
            i64(spec_maxes, True),
            i64(spec_negated, True),
            i64(crit_attrs, True),
            i64(crit_ops, True),
            i64(value_offsets),
            i64(values, True),
        )
        return self._compiled

    def _token_matrix(self, doc):
        if doc._lexical_attrs is not None:
            return doc._lexical_attrs
        matrix = np.empty((len(doc), 10), dtype=np.int64)
        for i, token in enumerate(doc):
            matrix[i] = (
                _signed(token.orth),
                _signed(token.lower),
                _signed(token.shape),
                len(token),
                token.is_alpha,
                token.is_digit,
                token.is_punct,
                token.is_space,
                token.like_num,
                token.is_ascii,
            )
        doc._lexical_attrs = matrix
        return matrix

    def _apply_greedy(self, matches):
        result = []
        for match_id in self._rules:
            candidates = [match for match in matches if match[0] == match_id]
            greedy = self._rules[match_id]["greedy"]
            if not greedy:
                result.extend(candidates)
                continue
            if greedy == "LONGEST":
                candidates.sort(key=lambda item: (-(item[2] - item[1]), item[1], item[2]))
            else:
                candidates.sort(key=lambda item: (item[1], item[2]))
            chosen = []
            for candidate in candidates:
                if not any(candidate[1] < other[2] and other[1] < candidate[2] for other in chosen):
                    chosen.append(candidate)
            result.extend(chosen)
        result.sort(key=lambda item: (item[2], item[1]))
        return result

    def __call__(self, doclike, *, as_spans=False, allow_missing=False, with_alignments=False):
        del allow_missing
        if not isinstance(doclike, (Doc, Span)):
            raise TypeError("Matcher expects a Doc or Span")
        if isinstance(doclike, Span):
            doc = doclike.doc
            offset = doclike.start
            working = Doc(
                doc.vocab,
                words=[token.text for token in doclike],
                spaces=[bool(token.whitespace_) for token in doclike],
            )
        else:
            doc = working = doclike
            offset = 0
        if not self._rules or not len(working):
            return []
        compiled = self._compiled or self._compile()
        (
            pattern_offsets,
            pattern_ids,
            spec_crit_offsets,
            spec_mins,
            spec_maxes,
            spec_negated,
            crit_attrs,
            crit_ops,
            value_offsets,
            values,
        ) = compiled
        tokens = self._token_matrix(working)
        scratch_a = np.empty(len(working) + 1, dtype=np.int64)
        scratch_b = np.empty(len(working) + 1, dtype=np.int64)
        capacity = max(32, len(working) * len(pattern_ids) * 2)
        while True:
            out_ids = np.empty(capacity, dtype=np.uint64)
            out_starts = np.empty(capacity, dtype=np.int64)
            out_ends = np.empty(capacity, dtype=np.int64)
            count = lib().msp_match(
                addr(tokens),
                len(working),
                tokens.shape[1],
                addr(pattern_offsets),
                addr(pattern_ids),
                len(pattern_ids),
                addr(spec_crit_offsets),
                addr(spec_mins),
                addr(spec_maxes),
                addr(spec_negated),
                addr(crit_attrs),
                addr(crit_ops),
                addr(value_offsets),
                addr(values),
                addr(out_ids),
                addr(out_starts),
                addr(out_ends),
                capacity,
                addr(scratch_a),
                addr(scratch_b),
            )
            if count >= 0:
                break
            if -count <= capacity:
                raise RuntimeError(f"Mojo matcher returned invalid count {count}")
            capacity = -count
        matches = [
            (int(out_ids[i]), int(out_starts[i]) + offset, int(out_ends[i]) + offset)
            for i in range(count)
        ]
        matches.sort(key=lambda item: (item[2], item[1]))
        matches = self._apply_greedy(matches)
        for index, match in enumerate(matches):
            callback = self._rules[match[0]]["on_match"]
            if callback is not None:
                callback(self, doc, index, matches)
        if as_spans:
            return [Span(doc, start, end, label=match_id) for match_id, start, end in matches]
        if with_alignments:
            return [
                (match_id, start, end, list(range(end - start)))
                for match_id, start, end in matches
            ]
        return matches
