// PIDController.cpp
#include "PIDController.h"

PID_Errors PIDController::compute(float sp, float mv, float dt) {
  PID_Errors e;
  e.error   = sp - mv;
  e.d_error = (e.error - last_error) / dt;
  i_sum    += e.error * dt;
  e.i_error = i_sum;
  e.u       = Kp * e.error + Ki * i_sum + Kd * e.d_error;
  last_error = e.error;
  return e;
}