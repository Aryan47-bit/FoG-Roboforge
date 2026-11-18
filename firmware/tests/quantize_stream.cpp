#include "control.h"
#include <cstdio>
#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif
int main() {
#ifdef _WIN32
  _setmode(_fileno(stdin),_O_BINARY);
  _setmode(_fileno(stdout),_O_BINARY);
#endif
  float raw[3];
  while(std::fread(raw,sizeof(float),3,stdin)==3){
    int8_t q[3];
    for(int i=0;i<3;++i)q[i]=quantize(raw[i],i);
    if(std::fwrite(q,1,3,stdout)!=3)return 2;
  }
  return std::ferror(stdin)?1:0;
}
