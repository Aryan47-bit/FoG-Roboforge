#pragma once
#include <cmath>
#include <cstdint>

namespace fog {
inline float clip(float v) { return v < -19.6133f ? -19.6133f : (v > 19.6133f ? 19.6133f : v); }
inline float normalize(float value, float mean, float sd) { return (clip(value)-mean)/sd; }
struct Alarm {
  float high, low, alpha, ema=0;
  int confirmations, count=0;
  bool active=false;
  void reset() { ema=0;count=0;active=false; }
  bool step(float p, bool valid=true) {
    if (!valid || !std::isfinite(p)) { reset();return false; }
    ema=alpha*p+(1-alpha)*ema;
    count=ema>=high ? count+1 : 0;
    if (active && ema<low) active=false;
    else if (!active && count>=confirmations) active=true;
    return active;
  }
};
}
