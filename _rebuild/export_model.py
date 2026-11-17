"""Map trained convolutions into Keras, verify predictions, export full INT8."""
import json
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import numpy as np
import torch
import tensorflow as tf
from data_pipeline import ROOT,OUT,save_json
from models import Detector
from train import Windows,evaluate

def main():
    selected=json.loads((OUT/'selected.json').read_text())['experiment']
    folder=OUT/'experiments'/selected
    result=json.loads((folder/'result.json').read_text())
    ckpt=torch.load(folder/'best.pt',map_location='cpu',weights_only=False)
    model=Detector(ckpt['kind']);model.load_state_dict(ckpt['state_dict']);model.eval()
    inputs=tf.keras.Input(shape=(100,3),name='acceleration_normalized')
    x=inputs;convs=[]
    for i,block in enumerate(model.blocks):
        old=x;x=tf.keras.layers.ZeroPadding1D(block.pad)(x)
        conv=tf.keras.layers.Conv1D(block.conv.out_channels,block.conv.kernel_size[0],dilation_rate=block.conv.dilation[0],padding='valid',name=f'conv_{i}')
        x=conv(x);convs.append((conv,block.conv))
        if block.residual:x=tf.keras.layers.Add()([old,x])
        x=tf.keras.layers.ReLU()(x)
    x=tf.keras.layers.GlobalAveragePooling1D()(x)
    dense=tf.keras.layers.Dense(1,activation='sigmoid');outputs=dense(x)
    keras=tf.keras.Model(inputs,outputs)
    for layer,conv in convs:layer.set_weights([conv.weight.detach().numpy().transpose(2,1,0),conv.bias.detach().numpy()])
    dense.set_weights([model.head.weight.detach().numpy().T,model.head.bias.detach().numpy()])
    records=json.loads((OUT/'manifest.json').read_text())
    mean=np.array(ckpt['mean'],dtype='float32');std=np.array(ckpt['std'],dtype='float32')
    ds=Windows([r for r in records if r['split']=='train'],mean,std,ckpt['overlap'],ckpt['representation'])
    rng=np.random.default_rng(20261117);indices=rng.choice(len(ds),min(512,len(ds)),replace=False)
    samples=np.stack([ds[int(i)][0] for i in indices])
    with torch.no_grad():expected=torch.sigmoid(model(torch.from_numpy(samples))).numpy()
    actual=keras(samples,training=False).numpy().ravel()
    error=float(np.max(np.abs(expected-actual)))
    if error>1e-5:raise RuntimeError(f'PyTorch/Keras parity failed: {error}')
    target=OUT/'deployment';target.mkdir(exist_ok=True)
    keras.save(target/'model.keras')
    converter=tf.lite.TFLiteConverter.from_keras_model(keras)
    floating=converter.convert();(target/'model_float.tflite').write_bytes(floating)
    converter=tf.lite.TFLiteConverter.from_keras_model(keras)
    converter.optimizations=[tf.lite.Optimize.DEFAULT]
    converter.representative_dataset=lambda: ([s[None,...]] for s in samples)
    converter.target_spec.supported_ops=[tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type=tf.int8;converter.inference_output_type=tf.int8
    quantized=converter.convert();(target/'model_int8.tflite').write_bytes(quantized)
    interpreter=tf.lite.Interpreter(model_content=quantized);interpreter.allocate_tensors()
    inp=interpreter.get_input_details()[0];out=interpreter.get_output_details()[0]
    assert inp['dtype']==np.int8 and out['dtype']==np.int8
    scale,zero=inp['quantization'];oscale,ozero=out['quantization']
    def infer(x):
        q=np.clip(np.rint(x/scale+zero),-128,127).astype('int8')
        interpreter.set_tensor(inp['index'],q[None]);interpreter.invoke()
        return float((interpreter.get_tensor(out['index']).ravel()[0].astype(float)-ozero)*oscale)
    quant_p=np.array([infer(x) for x in samples])
    test=Windows([r for r in records if r['split']=='test'],mean,std,ckpt['overlap'],ckpt['representation'])
    qtest=np.array([infer(test[i][0]) for i in range(len(test))])
    metrics=evaluate(test,qtest,result['alarm'])
    ops=sorted({v['op_name'] for v in interpreter._get_ops_details() if v['op_name']!='DELEGATE'})
    resolver_ops={'CONV_2D','MEAN','FULLY_CONNECTED','LOGISTIC','RESHAPE','PAD','ADD','EXPAND_DIMS','SQUEEZE','SPACE_TO_BATCH_ND','BATCH_TO_SPACE_ND','RELU'}
    if set(ops)-resolver_ops:raise RuntimeError(f'Firmware resolver missing operators: {set(ops)-resolver_ops}')
    report={'selected':selected,'torch_keras_max_absolute_error':error,'float_bytes':len(floating),
            'int8_bytes':len(quantized),'input_shape':inp['shape'].tolist(),
            'input_quantization':[scale,zero],'output_quantization':[oscale,ozero],
            'quantization_calibration_source':'512 train-only windows; no validation/test calibration',
            'calibration_max_probability_error':float(np.max(np.abs(quant_p-expected))),
            'operators':ops,'operators_covered_by_candidate_resolver':True,'test_int8':metrics,'test_float':result['test'],
            'tflm_board_parity':'NOT RUN; board and toolchain unavailable',
            'tensor_arena_bytes':None,'on_device_latency_ms':None}
    save_json(target/'validation.json',report)
    np.savez_compressed(target/'parity_vectors.npz',normalized=samples[:16],unnormalized_model_input_mps2=samples[:16]*std+mean,float_probability=expected[:16],int8_probability=quant_p[:16],mean=mean,std=std,representation=ckpt['representation'])
    # Only generate header after successful conversion, dtype checks and full test evaluation.
    text='#pragma once\n#include <cstddef>\n#include <cstdint>\nalignas(16) const unsigned char fog_model[] = {\n'
    text+='\n'.join(','.join(f'0x{b:02x}' for b in quantized[i:i+16])+',' for i in range(0,len(quantized),16))
    text+='\n};\nconst unsigned int fog_model_len = sizeof(fog_model);\n'
    (target/'model_data.h').write_text(text)
    cfg='#pragma once\n'
    for name,values in [('MEAN',mean),('STD',std)]:
        cfg+=f'const float FOG_{name}[3] = {{'+','.join(f'{v:.9e}f' for v in values)+'};\n'
    for name,key in [('HIGH','high'),('LOW','low'),('ALPHA','alpha')]:
        cfg+=f'constexpr float FOG_{name} = {result["alarm"][key]:.9e}f;\n'
    cfg+=f'constexpr int FOG_CONFIRM = {result["alarm"]["confirm"]};\n'
    cfg+=f'constexpr bool FOG_DYNAMIC = {str(ckpt["representation"]=="dynamic").lower()};\n'
    (target/'deployment_config.h').write_text(cfg)
    import shutil
    for name in ['model_data.h','deployment_config.h']:
        shutil.copyfile(target/name,ROOT/'firmware'/'esp32'/name)
    shutil.copyfile(ROOT/'firmware'/'preprocessing.h',ROOT/'firmware'/'esp32'/'preprocessing.h')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
