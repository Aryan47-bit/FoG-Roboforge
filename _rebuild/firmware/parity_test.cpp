#include "preprocessing.h"
#include <cassert>
#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <string>
int main(int argc, char** argv) {
  fog::Alarm a{.5f,.3f,1.f,0.f,2,0,false};
  assert(!a.step(1));assert(a.step(1));assert(!a.step(0));
  a.step(1);a.step(1);assert(!a.step(NAN));
  if(argc==6 && std::string(argv[1])=="--alarm") {
    fog::Alarm alarm{std::strtof(argv[2],nullptr),std::strtof(argv[3],nullptr),std::strtof(argv[4],nullptr),0.f,std::atoi(argv[5]),0,false};
    float p;
    while(std::cin >> p)std::cout << alarm.step(p) << '\n';
    return 0;
  }
  std::cout << std::setprecision(9);
  float value,mean,sd;
  while (std::cin >> value >> mean >> sd) std::cout << fog::normalize(value,mean,sd) << '\n';
}
