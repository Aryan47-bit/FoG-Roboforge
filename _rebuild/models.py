"""Small models with explicit padding and a direct Keras export mapping."""
import torch
from torch import nn
from torch.nn import functional as F

class Block(nn.Module):
    def __init__(self,ci,co,k=5,dilation=1,residual=False,causal=False):
        super().__init__()
        self.conv=nn.Conv1d(ci,co,k,dilation=dilation)
        self.pad=(dilation*(k-1),0) if causal else (dilation*(k-1)//2,)*2
        self.residual=residual
    def forward(self,x):
        y=self.conv(F.pad(x,self.pad))
        return F.relu(y+x if self.residual else y)

class Detector(nn.Module):
    def __init__(self,kind='cnn'):
        super().__init__()
        self.kind=kind
        self.blocks=nn.ModuleList([
            Block(3,12),
            Block(12,12,dilation=2 if kind=='tcn' else 1,residual=kind!='cnn',causal=kind=='tcn'),
            Block(12,12,dilation=4 if kind=='tcn' else 1,residual=kind!='cnn',causal=kind=='tcn')])
        if kind=='tcn': self.blocks[0].pad=(4,0)
        self.head=nn.Linear(12,1)
        if kind=='longtcn':
            self.blocks=nn.ModuleList([Block(3,24,causal=True)]+[
                Block(24,24,dilation=d,residual=True,causal=True) for d in [2,4,8]])
            self.head=nn.Linear(24,1)
    def forward(self,x):
        x=x.transpose(1,2)
        for block in self.blocks: x=block(x)
        return self.head(x.mean(dim=2)).squeeze(-1)

def verify_gpu():
    if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable; CPU training forbidden')
    device=torch.device('cuda')
    report={'torch':torch.__version__,'cuda':torch.version.cuda,'device':torch.cuda.get_device_name(0),'models':{}}
    for kind in ['cnn','rescnn','tcn']:
        model=Detector(kind).to(device)
        opt=torch.optim.Adam(model.parameters())
        x=torch.randn(16,100,3,device=device); y=torch.randint(0,2,(16,),device=device).float()
        before=model.blocks[0].conv.weight.detach().clone()
        loss=F.binary_cross_entropy_with_logits(model(x),y)
        loss.backward()
        assert all(p.grad is not None and p.grad.is_cuda and torch.isfinite(p.grad).all() for p in model.parameters())
        opt.step(); torch.cuda.synchronize()
        assert not torch.equal(before,model.blocks[0].conv.weight)
        report['models'][kind]={'loss':loss.item(),'parameter_device':str(next(model.parameters()).device),'gradient_device':str(next(model.parameters()).grad.device),'weights_changed':True}
    return report
