from pathlib import Path
import re

import numpy as np

from .base_dataset import Recording


def load_daphnet(root):
    """Retain all nine accelerometer axes; annotations 0 are invalid."""
    root = Path(root)
    paths = [root] if root.is_file() else sorted(root.rglob("S*R*.txt"))
    if not paths:
        raise FileNotFoundError(f"No Daphnet recordings under {root}")
    channels = tuple(f"{site}_acc_{axis}" for site in ("ankle", "thigh", "trunk") for axis in "xyz")
    for path in paths:
        match = re.fullmatch(r"(S\d+)(R\d+)", path.stem)
        if not match:
            raise ValueError(f"Unknown Daphnet recording name: {path}")
        data = np.loadtxt(path)
        if data.ndim != 2 or data.shape[1] != 11 or not np.isin(data[:, -1], [0, 1, 2]).all():
            raise ValueError(f"Invalid Daphnet schema: {path}")
        yield Recording("daphnet", match[1], match[2], path.stem, path, 64,
                        data[:, 0] / 1000, (data[:, 1:10] / 1000).astype("float32"),
                        (data[:, 10] - 1).astype("int8"), channels, ("g",) * 9).validate()

