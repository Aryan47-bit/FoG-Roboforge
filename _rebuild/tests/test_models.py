import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import unittest
try:
    import torch
except ImportError:
    torch=None

@unittest.skipIf(torch is None,'CUDA PyTorch environment is not installed')
class Models(unittest.TestCase):
    def test_shapes_and_gradients(self):
        from models import Detector
        for kind in ['cnn','rescnn','tcn','longtcn']:
            m=Detector(kind);x=torch.zeros(2,100,3);out=m(x)
            self.assertEqual(tuple(out.shape),(2,));self.assertTrue(torch.isfinite(out).all())
            out.sum().backward()
            self.assertTrue(all(p.grad is not None for p in m.parameters()))
            self.assertLess(sum(p.numel() for p in m.parameters()),10000)
    def test_tcn_has_no_future_dependency(self):
        from models import Detector
        torch.manual_seed(1);m=Detector('tcn')
        x=torch.randn(1,3,100);y=x.clone();y[:,:,70:]+=100
        for block in m.blocks:x=block(x);y=block(y)
        torch.testing.assert_close(x[:,:,:70],y[:,:,:70],rtol=0,atol=0)

if __name__=='__main__':unittest.main()
