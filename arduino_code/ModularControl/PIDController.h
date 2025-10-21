// PIDController.h
#pragma once
#include "Types.h"

struct PIDController {
  float Kp, Ki, Kd;
  float last_error = 0.0f;
  float i_sum = 0.0f;

  PIDController(float kp, float ki, float kd) : Kp(kp), Ki(ki), Kd(kd) {}
  PID_Errors compute(float setpoint, float measured, float dt);
  void reset() { last_error = 0.0f; i_sum = 0.0f; }
};