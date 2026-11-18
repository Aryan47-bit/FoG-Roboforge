import numpy as np

from src.inference.postprocessing import alarm_times


def event_counts(record, decision_times, probabilities, config, pre=5., post=1.):
    """Match alarms one-to-one within [onset-pre, onset+post].

    Score only the interval after the first decision and before the source's
    right-censored tail. Retain the full probability trace for causal FSM state.
    Non-FoG exposure is measured once from source samples, not window counts.
    Onsets need their full early-warning and post-onset opportunity observed.
    """
    t = np.asarray(decision_times)
    if len(t) < 2:
        raise ValueError("Continuous evaluation needs at least two decisions")
    if t[0] < record.timestamps[0] or t[-1] > record.timestamps[-1]:
        raise ValueError("Decisions must lie inside the recording")
    step = np.median(np.diff(t))
    if np.any(np.diff(t) > 1.5 * step):
        raise ValueError("Evaluation requires an uninterrupted decision trace")
    begin, end = t[0], min(t[-1], record.timestamps[-1] - pre)
    if end <= begin:
        raise ValueError("No uncensored monitoring interval")
    all_onsets = record.timestamps[1:][(record.labels[:-1] == 0) & (record.labels[1:] == 1)]
    onsets = all_onsets[(all_onsets - pre >= begin) & (all_onsets + post <= end)]
    alarms = alarm_times(t, probabilities, config)
    alarms = alarms[(alarms >= begin) & (alarms < end)]
    matched, latencies, false, unscored = set(), [], 0, 0
    for alarm in alarms:
        candidates = np.flatnonzero((alarm >= onsets-pre) & (alarm <= onsets+post))
        candidates = [int(i) for i in candidates if int(i) not in matched]
        if candidates:
            index = candidates[0]
            matched.add(index)
            latencies.append(float(alarm - onsets[index]))
        else:
            # Alerts associated with an already matched or boundary-censored
            # event, or ongoing FoG, are reported separately, not as normal-gait FA.
            associated = np.any((alarm >= all_onsets-pre) & (alarm <= all_onsets+post))
            source_index = np.searchsorted(record.timestamps, alarm, side="right") - 1
            if associated or record.labels[source_index] == 1:
                unscored += 1
            else:
                false += 1
    # Integrate sample support clipped to the scored interval.
    durations = np.maximum(0, np.minimum(record.timestamps[1:], end) - np.maximum(record.timestamps[:-1], begin))
    normal_seconds = float(durations[record.labels[:-1] == 0].sum())
    return {"episodes": len(onsets), "detected": len(matched), "false_alarms": false,
            "non_fog_seconds": normal_seconds, "latencies_seconds": latencies,
            "other_alarms": unscored, "monitoring_seconds": float(end-begin)}


def summarize(counts):
    result = {key: sum(c[key] for c in counts) for key in
              ("episodes", "detected", "false_alarms", "non_fog_seconds", "other_alarms", "monitoring_seconds")}
    latencies = [v for c in counts for v in c["latencies_seconds"]]
    result.update(episode_sensitivity=result["detected"] / result["episodes"] if result["episodes"] else None,
                  fa_per_hour=result["false_alarms"] * 3600 / result["non_fog_seconds"] if result["non_fog_seconds"] else None,
                  median_latency_seconds=float(np.median(latencies)) if latencies else None,
                  early_detections=sum(v < 0 for v in latencies),
                  warnings_5_to_10_seconds=sum(-10 <= v <= -5 for v in latencies))
    return result


def evaluate_traces(traces, config):
    by_subject = {}
    for record, times, probabilities in traces:
        by_subject.setdefault(record.subject_key, []).append(event_counts(record, times, probabilities, config))
    return {"pooled": summarize([c for rows in by_subject.values() for c in rows]),
            "subjects": {subject: summarize(rows) for subject, rows in by_subject.items()}}
