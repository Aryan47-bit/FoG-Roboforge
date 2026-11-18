#pragma once
#include <cmath>
#include <cstdint>
#include "deployment_config.h"
struct FSM {
  enum State {Normal,Warning,Active,Cooldown} state=Normal;
  uint32_t since=0; bool confirming=false;
  void reset(){state=Normal;confirming=false;}
  void step(float p,uint32_t now){
    if(!std::isfinite(p)){reset();return;}
    if(state==Cooldown){if(now-since>=3000)reset();return;}
    if(state==Active){if(p<FOG_LOW || now-since>=3000){state=Cooldown;since=now;}return;}
    if(p<FOG_LOW){reset();return;}
    state=Warning;
    if(p<FOG_HIGH){confirming=false;return;}
    if(!confirming){confirming=true;since=now;}
    if(now-since>=500){state=Active;since=now;confirming=false;}
  }
};
