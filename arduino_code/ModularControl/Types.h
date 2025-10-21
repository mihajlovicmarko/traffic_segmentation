#pragma once
struct PID_Errors {
  float error;
  float d_error;
  float i_error;
  float u;       // control output
};