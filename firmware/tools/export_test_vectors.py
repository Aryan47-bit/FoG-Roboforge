import numpy as np
from pathlib import Path
base=Path(__file__).resolve().parents[2]
v=np.load(base/'_rebuild/artifacts/deployment/parity_vectors.npz')
x=v['unnormalized_model_input_mps2'][:5]
def literal(v): return f'{float(v):.9e}f'
rows=[ '{'+','.join('{'+','.join(map(literal,s))+'}' for s in w)+'}' for w in x]
(base/'firmware/include/golden_vectors.h').write_text('#pragma once\nstatic const float golden[5][100][3]={'+','.join(rows)+'};\nstatic const float expected[5]={'+','.join(map(literal,v['int8_probability'][:5]))+'};\n')
q=np.clip(np.rint(v['normalized'][:5]/0.06116848438978195)+10,-128,127).astype(int)
rows=['{'+','.join('{'+','.join(map(str,s))+'}' for s in w)+'}' for w in q]
with (base/'firmware/include/golden_vectors.h').open('a') as f:
    f.write('static const signed char expected_input[5][100][3]={'+','.join(rows)+'};\n')
