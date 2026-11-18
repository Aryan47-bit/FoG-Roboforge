#include <Arduino.h>
#include "telemetry.h"
void logSample(uint32_t stamp,const float xyz[3],bool good){
  if(Serial.availableForWrite()>72)
    Serial.printf("S,%lu,%.5f,%.5f,%.5f,%d\n",(unsigned long)stamp,xyz[0],xyz[1],xyz[2],int(good));
}
void logPrediction(uint32_t stamp,float p,int state,uint32_t latency,uint32_t errors){
  Serial.printf("P,%lu,%.6f,%d,%lu,%lu\n",(unsigned long)stamp,p,state,(unsigned long)latency,(unsigned long)errors);
}
