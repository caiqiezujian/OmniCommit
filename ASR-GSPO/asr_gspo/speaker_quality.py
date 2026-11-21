"""Score local turn boundaries, not persistent speaker identity or DER.

Borrow the correct-word/wrong-speaker principle from arXiv:2611.19136, adapted
to Fourth's change-only protocol. Exact boundaries anchor monotone matching;
a lost marker affects its next reference turn without a global parity cascade.
Character-weighted movement grades short and long shifts differently. Deleted
new-turn content also counts, so removing its head cannot make attribution
perfect. This is not an implementation of fixed-role SA-WER.
"""
from bisect import bisect_left, bisect_right
from collections import Counter

from .character_quality import characters


def _exact_metrics(task, ref, hyp, operations, predicted_offsets):
    scored = task.get("speaker_scorable", False)
    if not scored:
        return dict(speaker_scorable=False, speaker_reference_changes=0, speaker_predicted_changes=len(predicted_offsets),
                    speaker_correct_changes=0, speaker_missed_changes=0, speaker_extra_changes=0, speaker_error=0.)
    ref_starts = [u.start for u in ref]
    hyp_starts = [u.start for u in hyp]
    expected = Counter(bisect_left(ref_starts, p) for p in task["speaker_changes"])
    exact = {h: r for op, r, h in operations if op == "match"}
    observed = Counter()
    unanchored = 0
    for p in predicted_offsets:
        index = bisect_left(hyp_starts, p)
        # A marker in the middle of a lexical word is not a valid boundary.
        if index and hyp[index-1].end > p:
            unanchored += 1
            continue
        right, left = exact.get(index), exact.get(index-1)
        if right is not None and (left is None or right == left + 1):
            observed[right] += 1
        elif left is not None and right is None:
            observed[left + 1] += 1
        else:
            unanchored += 1
    correct = sum((expected & observed).values())
    missed = sum(expected.values()) - correct
    extra = sum(observed.values()) + unanchored - correct
    return dict(speaker_scorable=True, speaker_reference_changes=sum(expected.values()),
                speaker_predicted_changes=len(predicted_offsets), speaker_correct_changes=correct,
                speaker_missed_changes=missed, speaker_extra_changes=extra,
                speaker_error=min(1., (missed + extra) / max(1, sum(expected.values()), len(predicted_offsets))))


def _project_boundaries(hyp, operations, offsets, reference_size):
    """Retain intervals for deletion gaps instead of inventing exact positions."""
    starts = [unit.start for unit in hyp]
    diagonal = {j: i for op, i, j in operations if op in ("match", "sub")}
    aligned = sorted(diagonal)
    projections = []
    for serial, offset in enumerate(offsets):
        index = bisect_left(starts, offset)
        if index and hyp[index - 1].end > offset:
            continue  # Never repair a marker inside a word.
        right_index = bisect_left(aligned, index)
        left = diagonal[aligned[right_index - 1]] + 1 if right_index else 0
        right = diagonal[aligned[right_index]] if right_index < len(aligned) else reference_size
        if aligned and left <= right:
            projections.append((serial, left, right))
    return projections


def _pair_boundaries(expected, projected, prefix):
    """Monotone character-distance pairing, partitioned by exact anchors.

    Within each partition, skipping a boundary costs the following reference
    turn's character mass. Movement has no arbitrary fixed-word search window.
    """
    size = len(prefix) - 1

    def gap_expected(index):
        start = expected[index]
        end = expected[index + 1] if index + 1 < len(expected) else size
        return max(1, prefix[end] - prefix[start])

    def gap_observed(index):
        _, lo, _ = projected[index]
        k = bisect_right(expected, lo)
        end = expected[k] if k < len(expected) else size
        return max(1, prefix[end] - prefix[lo])

    lookup = {position: i for i, position in enumerate(expected)}
    anchors, used, pairs = [], set(), {}
    for j, (_, lo, hi) in enumerate(projected):
        i = lookup.get(lo) if lo == hi else None
        if i is not None and i not in used:
            anchors.append((i, j))
            used.add(i)

    def solve(begin_i, end_i, begin_j, end_j):
        rows, columns = end_i - begin_i, end_j - begin_j
        costs = [[0] * (columns + 1) for _ in range(rows + 1)]
        path = [[0] * (columns + 1) for _ in range(rows + 1)]
        for i in range(1, rows + 1):
            costs[i][0] = costs[i - 1][0] + gap_expected(begin_i + i - 1)
            path[i][0] = 1
        for j in range(1, columns + 1):
            costs[0][j] = costs[0][j - 1] + gap_observed(begin_j + j - 1)
            path[0][j] = 2
        for i in range(1, rows + 1):
            r = expected[begin_i + i - 1]
            for j in range(1, columns + 1):
                _, lo, hi = projected[begin_j + j - 1]
                p = min(max(r, lo), hi)
                options = (costs[i - 1][j - 1] + abs(prefix[r] - prefix[p]),
                           costs[i - 1][j] + gap_expected(begin_i + i - 1),
                           costs[i][j - 1] + gap_observed(begin_j + j - 1))
                costs[i][j] = min(options)
                path[i][j] = options.index(costs[i][j])
        i, j = rows, columns
        while i or j:
            direction = path[i][j]
            if direction == 0:
                pairs[begin_i + i - 1] = begin_j + j - 1
                i -= 1
                j -= 1
            elif direction == 1:
                i -= 1
            else:
                j -= 1

    previous_i = previous_j = -1
    for i, j in anchors + [(len(expected), len(projected))]:
        solve(previous_i + 1, i, previous_j + 1, j)
        if i < len(expected):
            pairs[i] = j
        previous_i, previous_j = i, j
    return pairs


def speaker_metrics(task, ref, hyp, operations, predicted_offsets, *, return_details=False):
    """Optionally return ``(metrics, correctly_attributed_matched_ref_indices)``.

    The details support quality-conditioned latency, not persistent identities.
    No output is reconstructed or repaired from the reference.
    """
    matched = {i for op, i, _ in operations if op == "match"}
    metrics = _exact_metrics(task, ref, hyp, operations, predicted_offsets)
    metrics.update(speaker_attribution_error=0., speaker_shifted_changes=0,
                   speaker_unpaired_changes=0, speaker_misattributed_characters=0,
                   speaker_deleted_turn_characters=0, speaker_attribution_characters=0,
                   speaker_ambiguous_changes=0)
    if not metrics["speaker_scorable"]:
        return (metrics, matched) if return_details else metrics
    starts = [u.start for u in ref]
    expected = sorted(bisect_left(starts, p) for p in task["speaker_changes"])
    weights = [max(1, len(characters(u.value))) for u in ref]
    prefix = [0]
    for weight in weights:
        prefix.append(prefix[-1] + weight)
    metrics["speaker_attribution_characters"] = prefix[-1]
    projected = _project_boundaries(hyp, operations, predicted_offsets, len(ref))
    pairs = _pair_boundaries(expected, projected, prefix)
    deleted = {i for op, i, _ in operations if op == "del"}
    wrong, deleted_turn, local = set(), set(), []
    used = set(pairs.values())
    for index, r in enumerate(expected):
        previous = expected[index - 1] if index else 0
        following = expected[index + 1] if index + 1 < len(expected) else len(ref)
        missing_here = deleted.intersection(range(r, following))
        deleted_turn.update(missing_here)
        deletion_fraction = sum(weights[i] for i in missing_here) / max(1, prefix[following] - prefix[r])
        if index not in pairs:
            wrong.update(range(r, following))
            local.append(1.)
            metrics["speaker_unpaired_changes"] += 1
            continue
        _, lo, hi = projected[pairs[index]]
        p = min(max(r, lo), hi)
        metrics["speaker_ambiguous_changes"] += lo != hi
        metrics["speaker_shifted_changes"] += p != r
        wrong.update(range(min(p, r), max(p, r)))
        denominator = prefix[following] - prefix[r] if p >= r else prefix[r] - prefix[previous]
        movement_fraction = min(1., abs(prefix[p] - prefix[r]) / max(1, denominator))
        local.append(max(deletion_fraction, movement_fraction))

    claimed_positions = {min(max(expected[i], projected[j][1]), projected[j][2]) for i, j in pairs.items()}
    for j, (_, lo, hi) in enumerate(projected):
        if j in used:
            continue
        # Empty initial/final and duplicate markers get the exact-boundary
        # penalty; they do not prove a different persistent person owns all text.
        if lo != hi or lo in (0, len(ref)) or lo in claimed_positions:
            continue
        turn = bisect_right(expected, lo)
        start = expected[turn - 1] if turn else 0
        stop = expected[turn] if turn < len(expected) else len(ref)
        wrong.update(range(lo, stop))
        local.append((prefix[stop] - prefix[lo]) / max(1, prefix[stop] - prefix[start]))
    # An illegal intra-word tag cannot be assigned a precise reference gap.
    # If the containing word is aligned, conservatively exclude that word and
    # the remainder of its reference turn from correct-attribution timing.
    diagonal = {j: i for op, i, j in operations if op in ("match", "sub")}
    hyp_starts = [u.start for u in hyp]
    invalid_positions = set()
    for offset in predicted_offsets:
        j = bisect_left(hyp_starts, offset) - 1
        if j >= 0 and hyp[j].start < offset < hyp[j].end and j in diagonal:
            invalid_positions.add(diagonal[j])
    for lo in invalid_positions:
        turn = bisect_right(expected, lo)
        start = expected[turn - 1] if turn else 0
        stop = expected[turn] if turn < len(expected) else len(ref)
        wrong.update(range(lo, stop))
        local.append((prefix[stop] - prefix[lo]) / max(1, prefix[stop] - prefix[start]))
    metrics["speaker_misattributed_characters"] = sum(weights[i] for i in wrong & matched)
    metrics["speaker_deleted_turn_characters"] = sum(weights[i] for i in deleted_turn)
    # The local worst/mean mixture keeps a short damaged turn visible in a
    # long recording, without saturating every one-character displacement.
    metrics["speaker_attribution_error"] = .5 * (max(local) + sum(local) / len(local)) if local else 0.
    return (metrics, matched - wrong) if return_details else metrics
