// Hardware integration candidate. Requires ESP32 Arduino 2.x and TensorFlow Lite Micro.
// No claim of board compilation/validation until performed with the target hardware.
#include <Arduino.h>
#include <Wire.h>
#include <atomic>
#include <esp_task_wdt.h>
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "preprocessing.h"
#include "model_data.h"
#include "deployment_config.h"

constexpr int SDA_PIN=21,SCL_PIN=22,MOTOR_PIN=13,LED_PIN=2,BUTTON_PIN=0;
constexpr uint8_t IMU_ADDR=0x68;
// Must be measured using the actual allocator on target; not a RAM usage claim.
alignas(16) uint8_t arena[100*1024];
struct Window { float x[100][3]; uint32_t timestamp,generation; };
QueueHandle_t queue;
std::atomic<uint32_t> cueUntil{0},lastInference{0},sensorErrors{0};
std::atomic<uint32_t> generation{0};
std::atomic<bool> sensorOK{false},vibrationEnabled{false}; // explicit serial V command enables vibration
float ring[100][3];int count=0,head=0,stride=0;
tflite::MicroInterpreter* interpreter=nullptr;
fog::Alarm alarm{FOG_HIGH,FOG_LOW,FOG_ALPHA,0.f,FOG_CONFIRM,0,false};

bool writeReg(uint8_t reg,uint8_t v) {
  Wire.beginTransmission(IMU_ADDR);Wire.write(reg);Wire.write(v);
  return Wire.endTransmission()==0;
}
bool readAccel(float* out) {
  Wire.beginTransmission(IMU_ADDR);Wire.write(0x3b);
  if(Wire.endTransmission(false)!=0 || Wire.requestFrom(IMU_ADDR,(uint8_t)6)!=(uint8_t)6) return false;
  for(int c=0;c<3;c++) {
    int16_t raw=(int16_t)((Wire.read()<<8)|Wire.read());
    out[c]=raw*(9.80665f/16384.f);
    if(abs((int)raw)>=32760) return false; // saturation suppresses inference
  }
  return true;
}
void samplingTask(void*) {
  esp_task_wdt_add(nullptr);
  TickType_t wake=xTaskGetTickCount();uint32_t previous=millis();float gravity[3]={0,0,0};
  for(;;) {
    esp_task_wdt_reset();
    vTaskDelayUntil(&wake,pdMS_TO_TICKS(20));uint32_t now=millis();
    float xyz[3];bool good=readAccel(xyz) && (uint32_t)(now-previous)<=30;previous=now;
    sensorOK=good;
    if(!good) { count=0;head=0;stride=0;cueUntil=0;sensorErrors++;generation++;for(float& v:gravity)v=0; }
    else {
      for(int c=0;c<3;c++) {
        float v=fog::clip(xyz[c]);gravity[c]=.02f*v+.98f*gravity[c];
        if(FOG_DYNAMIC) v-=gravity[c];
        ring[head][c]=(v-FOG_MEAN[c])/FOG_STD[c];
      }
      head=(head+1)%100;count=min(count+1,150);stride++;
      if(count>=150 && stride>=10) {
        stride=0;Window w;w.timestamp=now;w.generation=generation.load();
        for(int i=0;i<100;i++)for(int c=0;c<3;c++)w.x[i][c]=ring[(head+i)%100][c];
        xQueueOverwrite(queue,&w);
      }
      // Serial collection output; no labels are fabricated.
      if(Serial.availableForWrite()>80) Serial.printf("S,%lu,%.6f,%.6f,%.6f,%d\n",now,xyz[0],xyz[1],xyz[2],good);
    }
    bool cue=vibrationEnabled && sensorOK && (int32_t)(cueUntil-now)>0 && (uint32_t)(now-lastInference)<400;
    ledcWrite(0,cue?220:0);
  }
}
void inferenceTask(void*) {
  esp_task_wdt_add(nullptr);
  Window w;
  for(;;) {
    esp_task_wdt_reset();
    if(xQueueReceive(queue,&w,pdMS_TO_TICKS(300))!=pdTRUE) { alarm.reset();cueUntil=0;continue; }
    if(!sensorOK || millis()-w.timestamp>200 || w.generation!=generation.load()) { alarm.reset();cueUntil=0;continue; }
    TfLiteTensor* in=interpreter->input(0);
    for(int i=0;i<100;i++)for(int c=0;c<3;c++) {
      int q=lroundf(w.x[i][c]/in->params.scale)+in->params.zero_point;
      in->data.int8[i*3+c]=(int8_t)constrain(q,-128,127);
    }
    uint32_t begin=micros();
    if(interpreter->Invoke()!=kTfLiteOk) { alarm.reset();cueUntil=0;continue; }
    TfLiteTensor* out=interpreter->output(0);
    float p=(out->data.int8[0]-out->params.zero_point)*out->params.scale;
    bool was=alarm.active;bool active=alarm.step(p,sensorOK && w.generation==generation.load());lastInference=millis();
    if(active && !was)cueUntil=millis()+150; // bounded single pulse on rising edge
    Serial.printf("P,%lu,%.6f,%d,%lu,%lu\n",millis(),p,active,micros()-begin,sensorErrors.load());
  }
}
void setup() {
  Serial.begin(115200);pinMode(LED_PIN,OUTPUT);pinMode(BUTTON_PIN,INPUT_PULLUP);
  ledcSetup(0,1000,8);ledcAttachPin(MOTOR_PIN,0);ledcWrite(0,0);
  Wire.begin(SDA_PIN,SCL_PIN,400000);Wire.setTimeOut(10);
  bool ok=writeReg(0x6b,0x01)&&writeReg(0x1a,0x03)&&writeReg(0x19,0x13)&&writeReg(0x1c,0x00);
  const tflite::Model* model=tflite::GetModel(fog_model);
  static tflite::MicroMutableOpResolver<12> resolver;
  resolver.AddConv2D();resolver.AddMean();resolver.AddFullyConnected();resolver.AddLogistic();
  resolver.AddReshape();resolver.AddPad();resolver.AddAdd();resolver.AddExpandDims();
  resolver.AddSqueeze();resolver.AddSpaceToBatchNd();resolver.AddBatchToSpaceNd();resolver.AddRelu();
  static tflite::MicroInterpreter runner(model,resolver,arena,sizeof(arena));interpreter=&runner;
  if(!ok || model->version()!=TFLITE_SCHEMA_VERSION || runner.AllocateTensors()!=kTfLiteOk) {
    Serial.println("STARTUP_FAILED: vibration disabled");return;
  }
  if(runner.input(0)->type!=kTfLiteInt8 || runner.output(0)->type!=kTfLiteInt8) return;
  if(runner.input(0)->dims->size!=3 || runner.input(0)->dims->data[1]!=100 || runner.input(0)->dims->data[2]!=3) return;
  queue=xQueueCreate(1,sizeof(Window));if(!queue)return;
  esp_task_wdt_init(2,true);
  xTaskCreatePinnedToCore(samplingTask,"sample",4096,nullptr,3,nullptr,0);
  xTaskCreatePinnedToCore(inferenceTask,"infer",8192,nullptr,1,nullptr,1);
}
void loop() {
  if(Serial.available()) { char c=Serial.read();if(c=='V')vibrationEnabled=true;if(c=='v')vibrationEnabled=false; }
  if(digitalRead(BUTTON_PIN)==LOW) { vibrationEnabled=false;cueUntil=0; }
  delay(10);
}
