from pathlib import Path

import numpy as np
import pandas as pd

from .base_dataset import Recording


CHANNELS = tuple(f"{site}_{kind}_{axis}" for site in ("ankleL", "ankleR", "back", "wrist")
                 for kind in ("acc", "gyro") for axis in "xyz")


def load_fogstar(path, subjects=None, chunksize=50000):
    """Read bounded CSV chunks, then assemble each selected subject's recordings.

    Re-scan per subject to bound memory; interleaved sessions and chunk boundaries
    never become artificial recording boundaries. Gyro units are left explicit as
    source units until their physical scaling is verified for deployment.
    """
    path = Path(path)
    keys = ["subjectID", "sessionID", "taskID"]
    if subjects is None:
        ids = set()
        for chunk in pd.read_csv(path, usecols=["subjectID"], chunksize=chunksize):
            if chunk.subjectID.isna().any():
                raise ValueError("Missing Fogstar subject ID")
            ids.update(chunk.subjectID.unique())
        subjects = sorted(ids)
    for subject in subjects:
        pieces = []
        for chunk in pd.read_csv(path, usecols=keys + ["timestamp", "fog"] + list(CHANNELS), chunksize=chunksize):
            selected = chunk[chunk.subjectID == subject]
            if not selected.empty:
                pieces.append(selected)
        if not pieces:
            raise ValueError(f"Fogstar subject not found: {subject}")
        frame = pd.concat(pieces, ignore_index=True)
        if frame[keys].isna().any().any():
            raise ValueError("Missing Fogstar recording identity")
        for (person, session, task), group in frame.groupby(keys, sort=True):
            group = group.sort_values("timestamp", kind="stable")
            t = group.timestamp.to_numpy(float)
            dt = np.median(np.diff(t))
            if np.isclose(dt, 1000 / 60, rtol=0.01):
                t /= 1000
            elif not np.isclose(dt, 1 / 60, rtol=0.01):
                raise ValueError(f"Unsupported Fogstar timestamp interval: {dt}")
            y = group.fog.to_numpy()
            if not np.isin(y, [0, 1]).all():
                raise ValueError("Invalid Fogstar labels")
            units = tuple("g" if "_acc_" in name else "source_gyro" for name in CHANNELS)
            yield Recording("fogstar", str(person), str(session), f"{person}_{session}_{task}",
                            path, 60, t, group[list(CHANNELS)].to_numpy("float32"),
                            y.astype("int8"), CHANNELS, units).validate()
