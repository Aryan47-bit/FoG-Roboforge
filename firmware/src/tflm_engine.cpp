#include <Arduino.h>
#include "tflm_engine.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "control.h"
#include "golden_vectors.h"
#ifdef FOG_BENCH
#include "bench_vectors.h"
#endif
#include "model_data.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"

alignas(16) uint8_t arena[32768];
tflite::MicroInterpreter* engine;
float predict(const float x[][3]){
  auto* in=engine->input(0);
  for(int i=0;i<100;++i)for(int c=0;c<3;++c)in->data.int8[3*i+c]=quantize(x[i][c],c);
  if(engine->Invoke()!=kTfLiteOk)return NAN;
  auto* out=engine->output(0);
  return (out->data.int8[0]-out->params.zero_point)*out->params.scale;
}
bool initModel(){
  static tflite::MicroMutableOpResolver<8> ops;
  ops.AddConv2D();ops.AddExpandDims();ops.AddFullyConnected();ops.AddLogistic();
  ops.AddMean();ops.AddPad();ops.AddReshape();ops.AddRelu();
  const auto* model=tflite::GetModel(fog_model);
  if(model->version()!=TFLITE_SCHEMA_VERSION)return false;
  static tflite::MicroErrorReporter reporter;
  static tflite::MicroInterpreter runtime(model,ops,arena,sizeof(arena),&reporter);engine=&runtime;
  if(engine->AllocateTensors()!=kTfLiteOk)return false;
  auto* in=engine->input(0);auto* out=engine->output(0);
  if(in->type!=kTfLiteInt8 || in->bytes!=300 || in->dims->size!=3 ||
     in->dims->data[0]!=1 || in->dims->data[1]!=100 || in->dims->data[2]!=3 ||
     out->type!=kTfLiteInt8 || out->bytes!=1 || in->params.zero_point!=10 ||
     out->params.zero_point!=-128 || fabsf(in->params.scale-.0611684844f)>1e-7f ||
     out->params.scale!=1.f/256)return false;
  for(int i=0;i<5;++i){float p=predict(golden[i]);if(!std::isfinite(p)||fabsf(p-expected[i])>.05f)return false;}
#ifdef FOG_BENCH
  for(int i=0;i<50;++i){
    float p=predict(bench[i]);
    Serial.printf("B,%d,%.6f,%.6f\n",i,p,benchExpected[i]);
    if(!std::isfinite(p)||fabsf(p-benchExpected[i])>.05f)return false;
  }
#endif
  return true;
}
