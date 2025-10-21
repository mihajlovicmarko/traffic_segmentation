// AngleSensor.cpp
#include "AngleSensor.h"

void AngleSensor::init() {
  (void)analogRead(pin_);
  delay(5);
  int raw = analogRead(pin_);
  float u = float(raw) / 1023.0f;
  if (u > 0.99f) u = 0.99f;
  if (u < 0.01f) u = 0.01f;
  filt_ = mapRawToDeg(u);
}

float AngleSensor::readDeg() {
  int raw = analogRead(pin_);
  float u = float(raw) / 1023.0f;
  if (u > 0.99f) u = 0.99f;
  if (u < 0.01f) u = 0.01f;
  float angle = mapRawToDeg(u);
  filt_ = alpha_ * angle + (1.0f - alpha_) * filt_;
  return filt_;
}

// Your mapping
float AngleSensor::mapRawToDeg(float u01) {
  return (0.25f * u01 / (1.0f - u01) - 0.36f) * 270.0f;
}