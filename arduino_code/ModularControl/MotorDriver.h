// MotorDriver.h
#pragma once
#include <Arduino.h>
#include "Config.h"

class MotorDriver {
public:
  MotorDriver(uint8_t pwmPin, uint8_t dirPin, uint8_t slewStep,
              float dirHyst, uint16_t dirMinHoldMs)
    : pwm_(pwmPin), dir_(dirPin), slew_(slewStep),
      hyst_(dirHyst), minHoldMs_(dirMinHoldMs) {}

  void begin();
  void applyControl(float u);
  uint8_t currentPwm() const { return curPwm_; }
  int currentDir() const { return curDir_; }

private:
  uint8_t pwm_, dir_, slew_;
  float   hyst_;
  uint16_t minHoldMs_;
  uint8_t curPwm_ = 0;
  int     curDir_ = 0; // 0/1
  uint32_t lastDirChange_ = 0;

  uint8_t slew(uint8_t target, uint8_t current, uint8_t step);
};