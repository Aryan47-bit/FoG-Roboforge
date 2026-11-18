"""Generate 50 raw windows and reference INT8 outputs for bench parity checks."""
from pathlib import Path
import numpy as np
import tensorflow as tf

root=Path(__file__).resolve().parents[2]
deployment=root/'_rebuild/artifacts/deployment'
reference=np.load(deployment/'parity_vectors.npz')
windows=list(reference['unnormalized_model_input_mps2'])
for path in sorted((root/'_rebuild/artifacts/recordings').glob('*.npz')):
    with np.load(path) as record:
        if len(record['x'])>=100:
            windows.append(record['x'][:100])
    if len(windows)==49:
        break
if len(windows)!=49:
    raise ValueError('Insufficient recordings for the reference bank')
windows.append(np.tile([22.,-22.,0.],(100,1)))
raw=np.asarray(windows,dtype=np.float32)
z=(np.clip(raw,-19.6133,19.6133)-reference['mean'])/reference['std']
runtime=tf.lite.Interpreter(model_path=str(deployment/'model_int8.tflite'))
runtime.allocate_tensors()
inp=runtime.get_input_details()[0]
out=runtime.get_output_details()[0]
scale,zero=inp['quantization']
q=np.clip(np.rint(z/scale)+zero,-128,127).astype('int8')
probs=[]
for values in q:
    runtime.set_tensor(inp['index'],values[None])
    runtime.invoke()
    probs.append((int(runtime.get_tensor(out['index']).item())-out['quantization'][1])*out['quantization'][0])
np.savez_compressed(root/'firmware/tests/reference_windows.npz',raw=raw,normalized=z,quantized=q,probabilities=probs)
