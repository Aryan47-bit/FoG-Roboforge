from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parents[1]
v=np.load(root/'tests/reference_windows.npz')
rows=['{'+','.join('{'+','.join(f'{float(a):.9e}f' for a in sample)+'}' for sample in window)+'}' for window in v['raw']]
(root/'include/bench_vectors.h').write_text('#pragma once\nstatic const float bench[50][100][3]={'+','.join(rows)+'};\nstatic const float benchExpected[]={'+','.join(f'{float(p):.9e}f' for p in v['probabilities'])+'};\n')
