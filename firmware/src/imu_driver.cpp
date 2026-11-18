#include <Arduino.h>
#include <Wire.h>
#include "imu_driver.h"
bool reg(uint8_t r,uint8_t v){
  Wire.beginTransmission(0x68); Wire.write(r);Wire.write(v);return Wire.endTransmission()==0;
}
bool initIMU(){
  return reg(0x6b,1)&&reg(0x1a,3)&&reg(0x19,19)&&reg(0x1c,0)&&reg(0x1b,8);
}
bool readAccel(float* xyz){
  Wire.beginTransmission(0x68); Wire.write(0x3b);
  if(Wire.endTransmission(false)||Wire.requestFrom(uint8_t(0x68),uint8_t(6))!=6)return false;
  for(int i=0;i<3;++i){
    int hi=Wire.read(),lo=Wire.read();int16_t v=(hi<<8)|lo;
    if(abs(int(v))>=32760)return false;
    xyz[i]=v*(9.80665f/16384.f);
  }
  return true;
}
void recover(){
  Wire.end();pinMode(22,OUTPUT_OPEN_DRAIN);digitalWrite(22,HIGH);
  for(int i=0;i<9;++i){digitalWrite(22,LOW);delayMicroseconds(5);digitalWrite(22,HIGH);delayMicroseconds(5);}
  Wire.begin(21,22,400000);Wire.setTimeOut(10);
}
