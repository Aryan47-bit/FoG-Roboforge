#include <Arduino.h>
#include "haptic_driver.h"
void initHaptic(){ledcSetup(0,1000,8);ledcAttachPin(13,0);ledcWrite(0,0);}
void driveHaptic(bool enabled,uint32_t elapsed){
  ledcWrite(0,enabled && elapsed<650 && elapsed%250<150 ? 220 : 0);
}
