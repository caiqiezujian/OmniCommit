"""Reference-verified commit delay with a bounded switch-confirmation allowance.

This is a discrete-round adaptation of onset buffers, not an implementation of
the fixed-role SA-WER paper's latency formula. No guessed word times, mandatory
WAITs, or future observations are added to the actor. Raw delays remain visible.
"""
from bisect import bisect_left
from collections import Counter
from math import ceil


def timing_metrics(ref, changes, speaker_changes, grid, onsets, offsets, matches,
                   *, grace_turns=1, head_units=3, attributed_units=None, alignment=()):
    if type(grace_turns) is not int or not 0 <= grace_turns <= 4:
        raise ValueError("switch_grace_turns must be an integer in 0..4")
    starts = [u.start for u in ref]
    boundaries = sorted({bisect_left(starts, lo) for lo, _ in changes}
                        | {bisect_left(starts, lo) for lo in speaker_changes})
    boundaries = [i for i in boundaries if i < len(ref)]
    heads, deadlines = set(), {}
    coarse_boundaries = 0
    for n, lo in enumerate(boundaries):
        stop = boundaries[n + 1] if n + 1 < len(boundaries) else len(ref)
        head = range(lo, min(lo + head_units, stop))
        heads.update(head)
        # One shared deadline per boundary. Later head words cannot each earn
        # another round of postponement. Coarse spans supply only real endpoints.
        anchored = [i for i in head if i in offsets]
        if not anchored:
            # Merged semantic units may have only an outer end. Use that real
            # completion, ONLY if its anchor starts with this run's first unit
            # and does not straddle the next change. Never interpolate word ends.
            candidates = []
            for anchor in alignment:
                first = bisect_left(starts, anchor["char_start"])
                last = bisect_left(starts, anchor["char_end"]) - 1
                if (first == lo and lo <= last < stop and last in offsets
                        and anchor.get("end_ms") is not None):
                    candidates.append(last)
            if candidates:
                anchored = [min(candidates)]
                heads.add(anchored[0])
                coarse_boundaries += 1
        if not anchored:
            continue
        first = anchored[0]
        ready = bisect_left(grid, offsets[first] - 1e-6)
        if ready >= len(grid):
            raise ValueError("alignment end is outside the audio schedule")
        deadline = grid[min(ready + grace_turns, len(grid) - 1)]
        # All text waiting behind the same head shares this deadline. It expires
        # once, so a five-character short turn can be confirmed together without
        # giving each later word a new allowance.
        deadlines.update({i: deadline for i in range(lo, stop)})

    early = {i for i in matches if i in onsets and matches[i] + 1e-6 < onsets[i]}
    delays, switch_delays, excess_delays = [], [], []
    missing = wrong_side = grace_used = 0
    for i, end in offsets.items():
        index = bisect_left(grid, end - 1e-6)
        if index >= len(grid):
            raise ValueError("alignment end is outside the audio schedule")
        if i not in matches or i in early:
            missing += 1
            continue
        if attributed_units is not None and i not in attributed_units:
            # Correct spelling under the wrong turn is not a fast correct commit.
            missing += 1
            wrong_side += 1
            continue
        eligible = grid[index]
        delay = max(0., matches[i] - eligible)
        deadline = max(eligible, deadlines.get(i, eligible))
        excess = max(0., matches[i] - deadline)
        delays.append(delay)
        excess_delays.append(excess)
        grace_used += int(delay > excess + 1e-6)
        if i in heads:
            switch_delays.append(delay)
    delays.sort()
    costs = [d / (4. + d) for d in excess_delays] + [1.] * missing
    return {
        "timed_units": len(offsets), "timed_matches": sum(i in matches for i in offsets),
        "attributed_timed_matches": len(delays), "timing_missing_or_early_units": missing,
        "timing_wrong_side_units": wrong_side, "early_units": len(early),
        "delay_samples": len(delays), "switch_timed_units": sum(i in heads for i in offsets),
        "switch_grace_turns": grace_turns, "switch_grace_used_units": grace_used,
        "switch_coarse_completion_boundaries": coarse_boundaries,
        "timing_cost": sum(costs) / len(costs) if costs else 0.,
        "extra_wait_ms_histogram": dict(Counter(round(d * 1000) for d in delays)),
        "switch_extra_wait_ms_histogram": dict(Counter(round(d * 1000) for d in switch_delays)),
        "excess_wait_ms_histogram": dict(Counter(round(d * 1000) for d in excess_delays)),
        "mean_extra_wait_seconds": sum(delays) / len(delays) if delays else None,
        "delay_p95_seconds": delays[ceil(.95 * len(delays)) - 1] if delays else None,
    }
