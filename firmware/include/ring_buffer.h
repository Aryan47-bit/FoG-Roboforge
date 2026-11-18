#pragma once
#include <cstdint>
#include <cstring>
struct Window {float x[100][3]; uint32_t stamp, epoch, sequence;};
struct Ring {
  float data[100][3]; unsigned head=0,count=0,stride=0;
  void reset(){head=count=stride=0;}
  bool push(const float* v,Window& w){
    memcpy(data[head],v,12); head=(head+1)%100;
    if(count<100) ++count;
    if(++stride<10 || count<100)return false;
    stride=0;
    for(unsigned i=0;i<100;++i)memcpy(w.x[i],data[(head+i)%100],12);
    return true;
  }
};
