"""Manifest-based loading; never invent patient IDs for bundled sample files."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .base_dataset import Recording


def load_multimodal(manifest):
    """Load aligned CSVs with explicit identity, columns, units, and time scale.

    Relative paths resolve against the manifest. Headerless files must supply
    `columns`. Timestamp columns must be numeric; original clock-time strings
    require a separately verified alignment/export step.
    """
    manifest = Path(manifest)
    entries = json.loads(manifest.read_text(encoding="utf-8"))
    if not isinstance(entries, list) or not entries:
        raise ValueError("Manifest must be a nonempty list of verified recordings")
    seen = set()
    for item in entries:
        key = (item["subject"], item["session"], item["recording"])
        if key in seen:
            raise ValueError(f"Duplicate recording identity: {key}")
        seen.add(key)
        path = manifest.parent / item["path"]
        kwargs = {"header": None, "names": item["columns"]} if "columns" in item else {}
        frame = pd.read_csv(path, **kwargs)
        channels = tuple(item["channels"])
        labels = frame[item["label_column"]].to_numpy()
        if not np.isin(labels, [0, 1]).all():
            raise ValueError("Multimodal labels must be binary")
        yield Recording("multimodal", str(item["subject"]), str(item["session"]), str(item["recording"]),
                        path, float(item["sample_rate_hz"]),
                        frame[item["time_column"]].to_numpy(float) * float(item["time_scale_seconds"]),
                        frame[list(channels)].to_numpy("float32"), labels.astype("int8"),
                        channels, tuple(item["units"])).validate()
