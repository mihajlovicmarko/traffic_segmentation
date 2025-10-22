// MotorDriver.cpp
#include "MotorDriver.h"
#include <math.h>

void MotorDriver::begin() {
  pinMode(pwm_, OUTPUT);
  pinMode(dir_, OUTPUT);
  digitalWrite(dir_, curDir_);
  analogWrite(pwm_, curPwm_);
  lastDirChange_ = millis();
}

uint8_t MotorDriver::slew(uint8_t target, uint8_t current, uint8_t step) {
  if (target > current + step) return current + step;
  if (target + step < current) return current - step;
  return target;
}

void MotorDriver::applyControl(float u) {
  uint32_t now = millis();

  // Direction with hysteresis + min-hold
  int desired = (u >  hyst_) ? 0 :
                (u < -hyst_) ? 1 : curDir_;
  if (desired != curDir_ && (now - lastDirChange_) >= minHoldMs_) {
    curDir_ = desired;
    lastDirChange_ = now;
    digitalWrite(dir_, curDir_);
  }

  // Magnitude -> PWM
  float mag = fabs(u) * OUTPUT_TO_PWM;
  if (mag > PWM_CAP) mag = PWM_CAP;
  uint8_t target = (uint8_t)mag;
  if (target > 0 && target < PWM_MIN) target = PWM_MIN;

  curPwm_ = slew(target, curPwm_, slew_);
  analogWrite(pwm_, curPwm_);
}

void MotorDriver::setDirectPWM(uint8_t pwm, int dir) {
  uint32_t now = millis();
  
  // Set direction if specified
  if (dir >= 0 && dir != curDir_ && (now - lastDirChange_) >= minHoldMs_) {
    curDir_ = dir;
    lastDirChange_ = now;
    digitalWrite(dir_, curDir_);
  }
  
  // Set PWM directly (with slew limiting for safety)
  curPwm_ = slew(pwm, curPwm_, slew_);
  analogWrite(pwm_, curPwm_);
}