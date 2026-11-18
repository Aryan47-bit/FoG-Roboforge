import numpy as np


def subject_split(subject_keys, seed=42):
    """Deterministic per-dataset 60/20/20 split (6/2/2 for ten subjects)."""
    keys = sorted(set(subject_keys))
    result = {}
    for dataset in sorted({key.split(":", 1)[0] for key in keys}):
        people = [key for key in keys if key.startswith(dataset + ":")]
        if len(people) < 3:
            raise ValueError(f"{dataset} needs at least three subjects for train/val/test")
        np.random.default_rng(seed).shuffle(people)
        n = max(1, round(len(people) * 0.2))
        for split, group in (("test", people[:n]), ("validation", people[n:2*n]), ("train", people[2*n:])):
            result.update({key: split for key in group})
    return result


def loso(subject_keys, validation_count=2):
    keys = sorted(set(subject_keys))
    if validation_count < 1 or len(keys) < validation_count + 2:
        raise ValueError("LOSO requires nonempty, separate train/validation/test groups")
    for test in keys:
        others = [key for key in keys if key != test]
        yield {key: ("test" if key == test else "validation" if key in others[:validation_count] else "train")
               for key in keys}
