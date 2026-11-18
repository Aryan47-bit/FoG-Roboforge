#include "control.h"
#include "golden_vectors.h"
#include <cassert>
int main(){
  Ring r;Window w{};
  for(int i=0;i<230;++i){
    float v[3]={float(i),0,0};
    bool ready=r.push(v,w);
    assert(ready==(i>=99 && (i+1)%10==0));
    if(ready)for(int j=0;j<100;++j)assert(w.x[j][0]==i-99+j);
  }
  r.reset();assert(r.count==0);
  FSM f;
  f.step(.9,100);f.step(.9,500);assert(f.state==FSM::Warning);
  f.step(.5,550);f.step(.9,600);f.step(.9,1100);assert(f.state==FSM::Active);
  f.step(.1,1200);f.step(.9,4199);assert(f.state==FSM::Cooldown);
  f.step(.9,4200);assert(f.state==FSM::Normal);
  for(int i=0;i<5;++i)for(int j=0;j<100;++j)for(int c=0;c<3;++c)
    assert(quantize(golden[i][j][c],c)==expected_input[i][j][c]);
}
