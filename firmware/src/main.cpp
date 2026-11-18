#include <Arduino.h>
#include <Wire.h>
#include <atomic>
#include <esp_task_wdt.h>
#include <esp_timer.h>
#include "control.h"
#include "imu_driver.h"
#include "telemetry.h"
#include "haptic_driver.h"
#include "tflm_engine.h"
namespace {
constexpr int button=0,led=2;
QueueHandle_t mailbox;
Ring ring;

std::atomic<uint32_t> epoch{0},errors{0},cueStart{0};
std::atomic<bool> cue{false},armed{false},ready{false},failed{false};
TaskHandle_t sampler,controller;
StaticTask_t sensorTaskStorage,controlTaskStorage;
StackType_t sensorStack[4096/sizeof(StackType_t)],controlStack[8192/sizeof(StackType_t)];
StaticQueue_t queueStorage;
alignas(4) uint8_t mailboxData[sizeof(Window)];
void sample(void*){
  Window outgoing;
  esp_task_wdt_add(nullptr);
  TickType_t wake=xTaskGetTickCount();uint32_t seq=0;bool configured=false;
  for(;;){
    vTaskDelayUntil(&wake,pdMS_TO_TICKS(20));esp_task_wdt_reset();
    if(!ready){wake=xTaskGetTickCount();continue;}
    float xyz[3]={};uint32_t stamp=millis(),start=micros();
    if(!configured)configured=initIMU();
    bool good=configured && readAccel(xyz) && micros()-start<3500;
    if(!good){
      ++epoch;++errors;cue=false;armed=false;ring.reset();
      if(errors.load()%50==1)Serial.printf("E,%lu,IMU,invalid sample\n",(unsigned long)stamp);
      if(micros()-start>=10000 || !configured)recover();
      configured=false;
    }else if(ring.push(xyz,outgoing)){
      outgoing.stamp=stamp;outgoing.epoch=epoch;outgoing.sequence=++seq;xQueueOverwrite(mailbox,&outgoing);
    }
    logSample(stamp,xyz,good);
  }
}
void control(void*){
  Window incoming;
  if(!initModel()){failed=true;Serial.println("E,0,MODEL,allocation or parity failed");vTaskDelete(nullptr);return;}
  ready=true;esp_task_wdt_add(nullptr);FSM fsm;uint32_t last=0,gen=epoch;
  for(;;){
    esp_task_wdt_reset();
    if(!xQueueReceive(mailbox,&incoming,pdMS_TO_TICKS(250))){fsm.reset();cue=false;continue;}
    bool gap=last && incoming.sequence!=last+1;last=incoming.sequence;
    if(gen!=epoch || gap || incoming.epoch!=epoch || millis()-incoming.stamp>150){
      fsm.reset();cue=false;gen=epoch;continue;
    }
    uint32_t begin=micros();float p=predict(incoming.x);uint32_t latency=micros()-begin;
    auto old=fsm.state;
    if(!std::isfinite(p)||incoming.epoch!=epoch||digitalRead(button)==LOW){fsm.reset();cue=false;}
    else{
      fsm.step(p,millis());
      if(fsm.state!=FSM::Active)cue=false;
      else if(old!=FSM::Active){cueStart=millis();cue=true;}
    }
    logPrediction(incoming.stamp,p,int(fsm.state),latency,errors.load());
  }
}
}
void updateMotor(void*){
  static bool held=false;
  bool pressed=digitalRead(button)==LOW;
  if(pressed){armed=false;cue=false;if(!held)++epoch;}held=pressed;
  uint32_t now=millis(),elapsed=now-cueStart.load();
  driveHaptic(ready && !failed && armed && !pressed && cue,elapsed);
}
void setup(){
  Serial.begin(115200);pinMode(button,INPUT_PULLUP);pinMode(led,OUTPUT);
  initHaptic();
  Wire.begin(21,22,400000);Wire.setTimeOut(10);
  esp_timer_create_args_t timerArgs{};
  timerArgs.callback=updateMotor;timerArgs.name="motor";
  static esp_timer_handle_t motorTimer;
  if(esp_timer_create(&timerArgs,&motorTimer)!=ESP_OK || esp_timer_start_periodic(motorTimer,5000)!=ESP_OK){
    failed=true;Serial.println("E,0,TIMER,motor timer unavailable");return;
  }
  mailbox=xQueueCreateStatic(1,sizeof(Window),mailboxData,&queueStorage);
  esp_task_wdt_init(3,true);
  if(mailbox){
    sampler=xTaskCreateStaticPinnedToCore(sample,"sample",sizeof(sensorStack),nullptr,3,sensorStack,&sensorTaskStorage,0);
    if(sampler)controller=xTaskCreateStaticPinnedToCore(control,"control",sizeof(controlStack),nullptr,1,controlStack,&controlTaskStorage,1);
  }
  if(!controller){failed=true;ready=false;Serial.println("E,0,TASK,startup failed");}
}
void loop(){
  static uint32_t health=0;
  uint32_t now=millis();
  if(Serial.available()){
    char cmd=Serial.read();
    if(cmd=='V' && ready && digitalRead(button)==HIGH && !cue)armed=true;
    if(cmd=='v')armed=false;
  }
  digitalWrite(led,failed?(now/100)%2:(now%1000<50));
  if(now-health>=1000 && sampler && ready){health=now;Serial.printf("H,%lu,%u,%u,%u,%u\n",(unsigned long)now,ESP.getFreeHeap(),ESP.getMinFreeHeap(),unsigned(uxTaskGetStackHighWaterMark(sampler)),unsigned(uxTaskGetStackHighWaterMark(controller)));}
  vTaskDelay(1);
}


