"""Hot tokenizer, matcher, and vector loops exposed through a C ABI."""

from std.math import sqrt
from std.runtime import initialize_runtime
from std.runtime.asyncrt import TaskGroup
from std.sys import simd_width_of

comptime U32Ptr = UnsafePointer[UInt32, AnyOrigin[mut=True]]
comptime U8Ptr = UnsafePointer[UInt8, AnyOrigin[mut=True]]
comptime I64Ptr = UnsafePointer[Int64, AnyOrigin[mut=True]]
comptime U64Ptr = UnsafePointer[UInt64, AnyOrigin[mut=True]]
comptime F32Ptr = UnsafePointer[Float32, AnyOrigin[mut=True]]
comptime W = simd_width_of[DType.float32]()


@fieldwise_init
struct CosineSums(ImplicitlyCopyable):
    var aa: Float32
    var bb: Float32
    var ab: Float32


@fieldwise_init
struct DotNorm(ImplicitlyCopyable):
    var dot: Float32
    var norm2: Float32


def u32p(addr: Int) -> U32Ptr:
    return U32Ptr(unsafe_from_address=addr)


def u8p(addr: Int) -> U8Ptr:
    return U8Ptr(unsafe_from_address=addr)


def i64p(addr: Int) -> I64Ptr:
    return I64Ptr(unsafe_from_address=addr)


def u64p(addr: Int) -> U64Ptr:
    return U64Ptr(unsafe_from_address=addr)


def f32p(addr: Int) -> F32Ptr:
    return F32Ptr(unsafe_from_address=addr)


def is_space(c: UInt32) -> Bool:
    return (
        c == 9 or c == 10 or c == 11 or c == 12 or c == 13 or c == 32
        or c == 0x85 or c == 0xA0 or c == 0x1680
        or (c >= 0x2000 and c <= 0x200A)
        or c == 0x2028 or c == 0x2029 or c == 0x202F
        or c == 0x205F or c == 0x3000
    )


def is_ascii_alnum(c: UInt32) -> Bool:
    return (
        (c >= 48 and c <= 57)
        or (c >= 65 and c <= 90)
        or (c >= 97 and c <= 122)
    )


def is_digit(c: UInt32) -> Bool:
    return c >= 48 and c <= 57


def is_url_chunk(chars: U32Ptr, start: Int, end: Int) -> Bool:
    for i in range(start, end):
        if chars[i] == 64:
            return True
        if i + 2 < end and chars[i] == 58 and chars[i + 1] == 47 and chars[i + 2] == 47:
            return True
    return (
        end - start >= 4
        and (chars[start] == 119 or chars[start] == 87)
        and (chars[start + 1] == 119 or chars[start + 1] == 87)
        and (chars[start + 2] == 119 or chars[start + 2] == 87)
        and chars[start + 3] == 46
    )


def has_internal_period(chars: U32Ptr, start: Int, end: Int) -> Bool:
    for i in range(start + 1, end - 1):
        if chars[i] == 46 and is_ascii_alnum(chars[i - 1]) and is_ascii_alnum(chars[i + 1]):
            return True
    return False


def is_basic_punct(c: UInt32) -> Bool:
    return (
        c == 33 or c == 34 or c == 35 or c == 36 or c == 37 or c == 38
        or c == 40 or c == 41 or c == 42 or c == 43 or c == 44
        or c == 45 or c == 47 or c == 58 or c == 59 or c == 60
        or c == 61 or c == 62 or c == 63 or c == 91 or c == 92
        or c == 93 or c == 94 or c == 96 or c == 123 or c == 124
        or c == 125 or c == 126 or c == 0x2013 or c == 0x2014
        or c == 0x2026 or c == 0x2018 or c == 0x2019
        or c == 0x201C or c == 0x201D
    )


def emit_span(starts: I64Ptr, ends: I64Ptr, kinds: U8Ptr, count: Int, start: Int, end: Int, kind: UInt8):
    starts[count] = Int64(start)
    ends[count] = Int64(end)
    kinds[count] = kind


@export("msp_tokenize")
def msp_tokenize(
    chars_addr: Int,
    n: Int,
    starts_addr: Int,
    ends_addr: Int,
    kinds_addr: Int,
) abi("C") -> Int:
    var chars = u32p(chars_addr)
    var starts = i64p(starts_addr)
    var ends = i64p(ends_addr)
    var kinds = u8p(kinds_addr)
    var count = 0
    var i = 0
    while i < n:
        if is_space(chars[i]):
            var j = i + 1
            while j < n and is_space(chars[j]):
                j += 1
            emit_span(starts, ends, kinds, count, i, j, UInt8(1))
            count += 1
            i = j
            continue

        var chunk_end = i + 1
        while chunk_end < n and not is_space(chars[chunk_end]):
            chunk_end += 1
        var urlish = is_url_chunk(chars, i, chunk_end)
        var abbreviation = has_internal_period(chars, i, chunk_end)
        var j = i
        var word_start = i
        while j < chunk_end:
            var c = chars[j]
            var boundary = is_basic_punct(c)
            if c == 39:
                boundary = not (
                    j > i and j + 1 < chunk_end
                    and is_ascii_alnum(chars[j - 1]) and is_ascii_alnum(chars[j + 1])
                )
            elif c == 46:
                boundary = True
                if j > i and j + 1 < chunk_end and is_ascii_alnum(chars[j - 1]) and is_ascii_alnum(chars[j + 1]):
                    boundary = False
                elif j + 1 == chunk_end and abbreviation and not urlish:
                    boundary = False
                elif urlish and j + 1 < chunk_end:
                    boundary = False
            elif c == 47:
                boundary = not (
                    urlish or (
                        j > i and j + 1 < chunk_end
                        and is_digit(chars[j - 1]) and is_digit(chars[j + 1])
                    )
                )
            elif c == 58 or c == 63 or c == 61 or c == 38 or c == 35:
                boundary = not urlish
            elif c == 45:
                boundary = not urlish

            if not boundary:
                j += 1
                continue
            if word_start < j:
                emit_span(starts, ends, kinds, count, word_start, j, UInt8(0))
                count += 1
            var punct_end = j + 1
            if c == 45 or c == 46:
                while punct_end < chunk_end and chars[punct_end] == c:
                    punct_end += 1
            emit_span(starts, ends, kinds, count, j, punct_end, UInt8(0))
            count += 1
            j = punct_end
            word_start = j
        if word_start < chunk_end:
            emit_span(starts, ends, kinds, count, word_start, chunk_end, UInt8(0))
            count += 1
        i = chunk_end
    return count


def criterion_matches(
    tokens: I64Ptr,
    token: Int,
    attr_count: Int,
    criterion: Int,
    crit_attrs: I64Ptr,
    crit_ops: I64Ptr,
    value_offsets: I64Ptr,
    values: I64Ptr,
) -> Bool:
    var actual = tokens[token * attr_count + Int(crit_attrs[criterion])]
    var begin = Int(value_offsets[criterion])
    var end = Int(value_offsets[criterion + 1])
    var found = False
    for i in range(begin, end):
        if actual == values[i]:
            found = True
            break
    var op = crit_ops[criterion]
    if op == 0 or op == 2:
        return found
    return not found


def spec_matches(
    tokens: I64Ptr,
    token: Int,
    attr_count: Int,
    spec: Int,
    spec_crit_offsets: I64Ptr,
    crit_attrs: I64Ptr,
    crit_ops: I64Ptr,
    value_offsets: I64Ptr,
    values: I64Ptr,
) -> Bool:
    var begin = Int(spec_crit_offsets[spec])
    var end = Int(spec_crit_offsets[spec + 1])
    for c in range(begin, end):
        if not criterion_matches(
            tokens, token, attr_count, c, crit_attrs, crit_ops, value_offsets, values
        ):
            return False
    return True


def append_unique(frontier: I64Ptr, count: Int, value: Int) -> Int:
    for i in range(count):
        if frontier[i] == Int64(value):
            return count
    frontier[count] = Int64(value)
    return count + 1


@export("msp_match")
def msp_match(
    tokens_addr: Int,
    n: Int,
    attr_count: Int,
    pattern_offsets_addr: Int,
    pattern_ids_addr: Int,
    pattern_count: Int,
    spec_crit_offsets_addr: Int,
    spec_mins_addr: Int,
    spec_maxes_addr: Int,
    spec_negated_addr: Int,
    crit_attrs_addr: Int,
    crit_ops_addr: Int,
    value_offsets_addr: Int,
    values_addr: Int,
    out_ids_addr: Int,
    out_starts_addr: Int,
    out_ends_addr: Int,
    capacity: Int,
    scratch_a_addr: Int,
    scratch_b_addr: Int,
) abi("C") -> Int:
    var tokens = i64p(tokens_addr)
    var pattern_offsets = i64p(pattern_offsets_addr)
    var pattern_ids = u64p(pattern_ids_addr)
    var spec_crit_offsets = i64p(spec_crit_offsets_addr)
    var spec_mins = i64p(spec_mins_addr)
    var spec_maxes = i64p(spec_maxes_addr)
    var spec_negated = i64p(spec_negated_addr)
    var crit_attrs = i64p(crit_attrs_addr)
    var crit_ops = i64p(crit_ops_addr)
    var value_offsets = i64p(value_offsets_addr)
    var values = i64p(values_addr)
    var out_ids = u64p(out_ids_addr)
    var out_starts = i64p(out_starts_addr)
    var out_ends = i64p(out_ends_addr)
    var scratch_a = i64p(scratch_a_addr)
    var scratch_b = i64p(scratch_b_addr)
    var total = 0

    for pattern in range(pattern_count):
        var spec_begin = Int(pattern_offsets[pattern])
        var spec_end = Int(pattern_offsets[pattern + 1])
        for start in range(n):
            scratch_a[0] = Int64(start)
            var current_count = 1
            var use_a = True
            for spec in range(spec_begin, spec_end):
                var next_count = 0
                var minimum = Int(spec_mins[spec])
                var maximum = Int(spec_maxes[spec])
                for fi in range(current_count):
                    var position = Int(scratch_a[fi]) if use_a else Int(scratch_b[fi])
                    if minimum == 0:
                        if use_a:
                            next_count = append_unique(scratch_b, next_count, position)
                        else:
                            next_count = append_unique(scratch_a, next_count, position)
                    var consumed = 0
                    while position + consumed < n and (maximum < 0 or consumed < maximum):
                        var matched = spec_matches(
                            tokens, position + consumed, attr_count, spec,
                            spec_crit_offsets, crit_attrs, crit_ops, value_offsets, values
                        )
                        if spec_negated[spec] != 0:
                            matched = not matched
                        if not matched:
                            break
                        consumed += 1
                        if consumed >= minimum:
                            if use_a:
                                next_count = append_unique(scratch_b, next_count, position + consumed)
                            else:
                                next_count = append_unique(scratch_a, next_count, position + consumed)
                current_count = next_count
                use_a = not use_a
                if current_count == 0:
                    break
            for fi in range(current_count):
                var match_end = Int(scratch_a[fi]) if use_a else Int(scratch_b[fi])
                if match_end > start:
                    if total < capacity:
                        out_ids[total] = pattern_ids[pattern]
                        out_starts[total] = Int64(start)
                        out_ends[total] = Int64(match_end)
                    total += 1
    if total > capacity:
        return -total
    return total


def dot_f32(a: F32Ptr, b: F32Ptr, n: Int) -> Float32:
    var acc = SIMD[DType.float32, W](0.0)
    var i = 0
    while i + W <= n:
        acc += a.load[width=W](i) * b.load[width=W](i)
        i += W
    var result = acc.reduce_add()
    while i < n:
        result += a[i] * b[i]
        i += 1
    return result


def dot_norm_f32(a: F32Ptr, b: F32Ptr, n: Int) -> DotNorm:
    var acc_dot = SIMD[DType.float32, W](0.0)
    var acc_norm = SIMD[DType.float32, W](0.0)
    var i = 0
    while i + W <= n:
        var av = a.load[width=W](i)
        var bv = b.load[width=W](i)
        acc_dot += av * bv
        acc_norm += bv * bv
        i += W
    var result = DotNorm(acc_dot.reduce_add(), acc_norm.reduce_add())
    while i < n:
        var av = a[i]
        var bv = b[i]
        result.dot += av * bv
        result.norm2 += bv * bv
        i += 1
    return result


def cosine_f32(a: F32Ptr, b: F32Ptr, begin: Int, end: Int) -> CosineSums:
    var acc_aa = SIMD[DType.float32, W](0.0)
    var acc_bb = SIMD[DType.float32, W](0.0)
    var acc_ab = SIMD[DType.float32, W](0.0)
    var i = begin
    while i + W <= end:
        var av = a.load[width=W](i)
        var bv = b.load[width=W](i)
        acc_aa += av * av
        acc_bb += bv * bv
        acc_ab += av * bv
        i += W
    var result = CosineSums(
        acc_aa.reduce_add(), acc_bb.reduce_add(), acc_ab.reduce_add()
    )
    while i < end:
        var av = a[i]
        var bv = b[i]
        result.aa += av * av
        result.bb += bv * bv
        result.ab += av * bv
        i += 1
    return result


@export("msp_cosine")
def msp_cosine(a_addr: Int, b_addr: Int, n: Int) abi("C") -> Float64:
    var a = f32p(a_addr)
    var b = f32p(b_addr)
    var sums = cosine_f32(a, b, 0, n)
    if sums.aa == 0.0 or sums.bb == 0.0:
        return 0.0
    return Float64(sums.ab / sqrt(sums.aa * sums.bb))


@export("msp_cosine_parallel")
def msp_cosine_parallel(
    a_addr: Int,
    b_addr: Int,
    n: Int,
    scratch_addr: Int,
    workers: Int,
) abi("C") -> Float64:
    initialize_runtime()
    var a = f32p(a_addr)
    var b = f32p(b_addr)
    var scratch = f32p(scratch_addr)
    var chunk_size = (n + workers - 1) // workers

    @__parameter
    async def reduce_chunk(chunk: Int):
        var begin = chunk * chunk_size
        var end = min(begin + chunk_size, n)
        var sums = cosine_f32(a, b, begin, end)
        var scratch_offset = chunk * 16
        scratch[scratch_offset] = sums.aa
        scratch[scratch_offset + 1] = sums.bb
        scratch[scratch_offset + 2] = sums.ab

    var tasks = TaskGroup()
    for chunk in range(workers):
        tasks.create_task(reduce_chunk(chunk))
    tasks.wait()
    var sums = CosineSums(0.0, 0.0, 0.0)
    for chunk in range(workers):
        var scratch_offset = chunk * 16
        sums.aa += scratch[scratch_offset]
        sums.bb += scratch[scratch_offset + 1]
        sums.ab += scratch[scratch_offset + 2]
    if sums.aa == 0.0 or sums.bb == 0.0:
        return 0.0
    return Float64(sums.ab / sqrt(sums.aa * sums.bb))


@export("msp_normalize")
def msp_normalize(data_addr: Int, rows: Int, dims: Int) abi("C"):
    var data = f32p(data_addr)
    for row in range(rows):
        var vector = data + row * dims
        var norm2 = dot_f32(vector, vector, dims)
        if norm2 == 0.0:
            continue
        var inv = Float32(1.0) / sqrt(norm2)
        var wide_inv = SIMD[DType.float32, W](inv)
        var i = 0
        while i + W <= dims:
            vector.store(i, vector.load[width=W](i) * wide_inv)
            i += W
        while i < dims:
            vector[i] *= inv
            i += 1


def most_similar_query(
    data: F32Ptr,
    rows: I64Ptr,
    row_count: Int,
    queries: F32Ptr,
    q: Int,
    dims: Int,
    nbest: Int,
    best_rows: I64Ptr,
    scores: F32Ptr,
):
    for k in range(nbest):
        best_rows[q * nbest + k] = -1
        scores[q * nbest + k] = -2.0
    var query = queries + q * dims
    var qnorm2 = dot_f32(query, query, dims)
    if qnorm2 == 0.0:
        return
    for ri in range(row_count):
        var row = Int(rows[ri])
        var candidate = data + row * dims
        var sums = dot_norm_f32(query, candidate, dims)
        var score = Float32(0.0)
        if sums.norm2 != 0.0:
            score = sums.dot / sqrt(qnorm2 * sums.norm2)
        var insert_at = nbest
        for k in range(nbest):
            if score > scores[q * nbest + k]:
                insert_at = k
                break
        if insert_at < nbest:
            var k = nbest - 1
            while k > insert_at:
                scores[q * nbest + k] = scores[q * nbest + k - 1]
                best_rows[q * nbest + k] = best_rows[q * nbest + k - 1]
                k -= 1
            scores[q * nbest + insert_at] = score
            best_rows[q * nbest + insert_at] = Int64(row)


@export("msp_most_similar")
def msp_most_similar(
    data_addr: Int,
    rows_addr: Int,
    row_count: Int,
    queries_addr: Int,
    query_count: Int,
    dims: Int,
    nbest: Int,
    best_rows_addr: Int,
    scores_addr: Int,
) abi("C"):
    var data = f32p(data_addr)
    var rows = i64p(rows_addr)
    var queries = f32p(queries_addr)
    var best_rows = i64p(best_rows_addr)
    var scores = f32p(scores_addr)
    for q in range(query_count):
        most_similar_query(
            data, rows, row_count, queries, q, dims, nbest, best_rows, scores
        )


@export("msp_most_similar_parallel")
def msp_most_similar_parallel(
    data_addr: Int,
    rows_addr: Int,
    row_count: Int,
    queries_addr: Int,
    query_count: Int,
    dims: Int,
    nbest: Int,
    best_rows_addr: Int,
    scores_addr: Int,
    workers: Int,
) abi("C"):
    initialize_runtime()
    var data = f32p(data_addr)
    var rows = i64p(rows_addr)
    var queries = f32p(queries_addr)
    var best_rows = i64p(best_rows_addr)
    var scores = f32p(scores_addr)
    var chunk_size = (query_count + workers - 1) // workers

    @__parameter
    async def search_chunk(chunk: Int):
        var begin = chunk * chunk_size
        var end = min(begin + chunk_size, query_count)
        for q in range(begin, end):
            most_similar_query(
                data, rows, row_count, queries, q, dims, nbest, best_rows, scores
            )

    var tasks = TaskGroup()
    for chunk in range(workers):
        tasks.create_task(search_chunk(chunk))
    tasks.wait()
