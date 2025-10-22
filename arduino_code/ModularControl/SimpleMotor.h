// SimpleMotor.h
#pragma once
#include <Arduino.h>

class SimpleMotor {
public:
  SimpleMotor(uint8_t pwmPin, uint8_t dirPin) 
    : pwm_(pwmPin), dir_(dirPin) {}

  void begin() {
    pinMode(pwm_, OUTPUT);
    pinMode(dir_, OUTPUT);
    digitalWrite(dir_, curDir_);
    analogWrite(pwm_, curPwm_);
  }

  void setPWM(uint8_t pwm, int dir = -1) {
    // Set direction if specified
    if (dir >= 0) {
      curDir_ = (dir > 0) ? 1 : 0;
      digitalWrite(dir_, curDir_);
    }
    
    // Set PWM directly
    curPwm_ = pwm;
    analogWrite(pwm_, curPwm_);
  }

  uint8_t currentPwm() const { return curPwm_; }
  int currentDir() const { return curDir_; }

private:
  uint8_t pwm_, dir_;
  uint8_t curPwm_ = 0;
  int curDir_ = 0; // 0/1
};