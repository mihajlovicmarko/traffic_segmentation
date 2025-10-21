#pragma once
#include <Arduino.h>
#include "Config.h"

// Simple ±A sine trajectory; enable with ENABLE_TRAJECTORY 1 in Config.h
inline float trajectory(uint32_t now_ms) {
  const float A = 20.0f;
  const float period_ms = 3000.0f;
  float ph = fmodf(now_ms, period_ms) / period_ms; // 0..1
  return A * sinf(2.0f * PI * ph);
}