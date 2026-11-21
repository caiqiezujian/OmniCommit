"""Small per-rank counters for component curves in the existing TensorBoard logger."""
from collections import Counter
import json
import os
from pathlib import Path

TOTALS = Counter()
FIELDS = ("character_errors", "reference_characters", "speaker_missed_changes", "speaker_extra_changes",
          "speaker_reference_changes", "speaker_predicted_changes", "extra_internal_periods",
          "unsupported_short_breaks", "scored_internal_gaps", "language_penalty", "speaker_penalty",
          "punctuation_penalty", "latency_penalty", "switch_content_fully_missed", "switch_content_spans")
FIELDS += ("speaker_misattributed_characters", "speaker_deleted_turn_characters", "speaker_attribution_characters",
           "speaker_shifted_changes", "speaker_unpaired_changes", "speaker_attribution_penalty",
           "extra_internal_stops", "switch_grace_used_units", "switch_timed_units", "timing_wrong_side_units",
           "attributed_timed_matches", "timed_units")


def publish_metrics(directory, records):
    from finalize_export import atomic_write
    for record in records:
        TOTALS.update(trajectories=1, invalid=int(not record["valid"]), reward=record["reward"])
        if record["valid"]:
            TOTALS.update(valid=1)
            TOTALS.update({k: record.get(k, 0) for k in FIELDS})
    path = Path(directory) / f"metrics-rank{os.environ.get('RANK', '0')}-pid{os.getpid()}.json"
    atomic_write(path, json.dumps(dict(TOTALS), allow_nan=False).encode("utf-8"))


def snapshots(directory):
    result = {}
    if directory:
        for path in Path(directory).glob("metrics-rank*.json"):
            try:
                result[str(path)] = Counter(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, ValueError):
                continue  # A replaced/rotated log must not fail a training step.
    return result


def update_logs(previous, current, logs):
    totals = Counter()
    for key, counts in current.items():
        old = previous.get(key, Counter())
        # Counter subtraction discards negative values (e.g. cumulative reward),
        # so subtract each field explicitly instead.
        totals.update({name: value - old[name] for name, value in counts.items()})
    previous.clear()
    previous.update(current)
    if not totals["trajectories"]:
        return
    logs["asr_invalid_fraction"] = totals["invalid"] / totals["trajectories"]
    if totals["reference_characters"]:
        logs["asr_cer"] = totals["character_errors"] / totals["reference_characters"]
    if totals["valid"]:
        for field in ("language_penalty", "speaker_penalty", "speaker_attribution_penalty", "punctuation_penalty", "latency_penalty"):
            logs["asr_" + field] = totals[field] / totals["valid"]
    for field in ("speaker_missed_changes", "speaker_extra_changes", "extra_internal_periods",
                  "unsupported_short_breaks", "switch_content_fully_missed", "speaker_misattributed_characters",
                  "speaker_deleted_turn_characters", "speaker_shifted_changes", "extra_internal_stops",
                  "switch_grace_used_units", "timing_wrong_side_units"):
        logs["asr_" + field] = totals[field]
    if totals["timed_units"]:
        logs["asr_attributed_timing_coverage"] = totals["attributed_timed_matches"] / totals["timed_units"]
