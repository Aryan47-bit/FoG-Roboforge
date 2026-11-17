import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import unittest
import numpy as np
from data_pipeline import preprocess,runs,split_participants,audit_record
from evaluation import Alarm,event_counts

class PipelineTests(unittest.TestCase):
    def test_runs(self):
        self.assertEqual(runs([0,1,1,0,1]),[(1,3),(4,5)])
    def test_causality(self):
        t=np.arange(640)/64;x=np.ones((640,3));y=np.zeros(640)
        a=list(preprocess(t,x,y,64))[0];x[400:]=100
        b=list(preprocess(t,x,y,64))[0]
        np.testing.assert_array_equal(a[1][a[0]<t[400]],b[1][b[0]<t[400]])
    def test_missing_breaks(self):
        t=np.arange(640)/64;x=np.ones((640,3));y=np.zeros(640);x[320]=np.nan
        result=list(preprocess(t,x,y,64));self.assertEqual(len(result),2)
        self.assertLess(result[0][0][-1],t[320]);self.assertGreater(result[1][0][0],t[320])
    def test_invalid_labels_breaks(self):
        t=np.arange(640)/64;x=np.ones((640,3));y=np.zeros(640);y[320]=-1
        self.assertEqual(len(list(preprocess(t,x,y,64))),2)
    def test_rate_and_units(self):
        t=np.floor(np.arange(640)*1000/64)/1000
        m=audit_record({'dataset':'daphnet'},t,np.ones((640,3)),np.zeros(640))
        self.assertAlmostEqual(m['sampling_rate_hz'],64,places=2)
    def test_alarm_recovery_and_invalid(self):
        a=Alarm(high=.5,low=.3,confirm=2,alpha=1)
        self.assertFalse(a.step(1));self.assertTrue(a.step(1));self.assertFalse(a.step(0))
        a.step(1);a.step(1);self.assertFalse(a.step(float('nan')))
    def test_events_not_backdated(self):
        t=np.arange(200)/50;y=np.zeros(200);y[100:150]=1
        starts=np.arange(0,101,10);p=np.ones(len(starts))
        r=event_counts(t,y,starts,p,dict(high=.5,low=.3,confirm=1,alpha=1))
        self.assertEqual(r['matched'],1);self.assertEqual(r['false_events'],0)
        self.assertAlmostEqual(r['valid_seconds'],2.02)
    def test_split_namespace_and_reproducibility(self):
        rows=[{'dataset':d,'participant':f'{d}:{i}'} for d in ['a','b'] for i in range(10)]
        a=split_participants(rows);self.assertEqual(a,split_participants(rows));self.assertEqual(len(a),20)
    def test_antialias_attenuation(self):
        t=np.arange(64*20)/64;y=np.zeros(len(t))
        def rms(freq):
            x=np.tile(np.sin(2*np.pi*freq*t)[:,None],(1,3))
            out=list(preprocess(t,x,y,64))[0][1]
            return np.sqrt(np.mean(out**2))
        self.assertLess(rms(29),.1*rms(5))
    def test_sample_gap_breaks(self):
        t=np.arange(640)/64;t[320:]+=1
        self.assertEqual(len(list(preprocess(t,np.ones((640,3)),np.zeros(640),64))),2)

if __name__=='__main__':unittest.main()
