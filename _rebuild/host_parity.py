import json
import subprocess
import numpy as np
from data_pipeline import ROOT,OUT,save_json
from evaluation import Alarm

def main():
    exe=OUT/'parity_test.exe'
    subprocess.run(['g++','-std=c++17','-O2',str(ROOT/'firmware/parity_test.cpp'),'-o',str(exe)],check=True)
    rng=np.random.default_rng(20261117);x=rng.uniform(-30,30,1000).astype('float32')
    mean=rng.normal(size=1000).astype('float32');sd=rng.uniform(.1,4,1000).astype('float32')
    text=''.join(f'{a} {b} {c}\n' for a,b,c in zip(x,mean,sd))
    process=subprocess.run([str(exe)],input=text,text=True,capture_output=True,check=True)
    actual=np.fromstring(process.stdout,sep=' ')
    expected=(np.clip(x,-19.6133,19.6133)-mean)/sd
    np.testing.assert_allclose(actual,expected,rtol=1e-6,atol=2e-5)
    selected=json.loads((OUT/'selected.json').read_text())['experiment']
    config=json.loads((OUT/'experiments'/selected/'result.json').read_text())['alarm']
    probs=np.r_[np.zeros(10),np.ones(20),np.zeros(20),rng.uniform(0,1,1000)].astype('float32')
    args=[str(exe),'--alarm',str(config['high']),str(config['low']),str(config['alpha']),str(config['confirm'])]
    process=subprocess.run(args,input='\n'.join(map(str,probs)),text=True,capture_output=True,check=True)
    actual_alarm=np.fromstring(process.stdout,sep=' ').astype(bool)
    alarm=Alarm(**config);expected_alarm=np.array([alarm.step(float(p)) for p in probs])
    np.testing.assert_array_equal(actual_alarm,expected_alarm)
    save_json(ROOT/'reports/host_parity.json',{'passed':True,'normalization_cases':1000,
        'max_absolute_error':float(np.max(np.abs(actual-expected))),
        'alarm_transitions_checked':len(probs),'alarm_config':config,
        'scope':'Host C++ clipping/normalization and selected alarm FSM only; no source-filter, sensor or TFLM-board parity claim'})
    print('HOST PARITY PASSED')

if __name__=='__main__':main()
