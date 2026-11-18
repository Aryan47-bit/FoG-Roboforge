#pragma once
#include <cmath>
#include <cstdint>
#include <cstring>
#include "deployment_config.h"

inline int8_t quantize(float v, int axis) {
  float z=(fmaxf(-19.6133f,fminf(v,19.6133f))-FOG_MEAN[axis])/FOG_STD[axis];
  return int8_t(fmaxf(-128,fminf(127,roundf(z/0.06116848438978195f)+10)));
}
