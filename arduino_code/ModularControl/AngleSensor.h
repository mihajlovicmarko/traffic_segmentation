// AngleSensor.h
#pragma once
#include <Arduino.h>

class AngleSensor {
public:
  explicit AngleSensor(uint8_t pin, float alpha) : pin_(pin), alpha_(alpha) {}
  void init();
  float readDeg();           // filtered angle
  float lastFiltered() const { return filt_; }

private:
  uint8_t pin_;
  float alpha_;
  float filt_ = 0.0f;
  float mapRawToDeg(float u01);
};