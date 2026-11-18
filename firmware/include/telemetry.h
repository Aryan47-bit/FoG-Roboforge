#pragma once
#include <cstdint>
void logSample(uint32_t stamp,const float xyz[3],bool good);
void logPrediction(uint32_t stamp,float p,int state,uint32_t latency,uint32_t errors);
