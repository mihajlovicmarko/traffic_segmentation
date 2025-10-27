#pragma once
#include <Arduino.h>

// --- Pins ---
// Motor 1 (PID-controlled angle motor)
static const uint8_t PWM_PIN = 10;     // Nano: 3,5,6,9,10,11
static const uint8_t DIR_PIN = 3;

// Motor 2 (Direct PWM-controlled motor)
static const uint8_t MOTOR2_PWM_PIN = 9;  // Second PWM pin
static const uint8_t MOTOR2_DIR_PIN = 2;   // Second direction pin

// --- Serial ---
static const uint32_t SERIAL_BAUD = 115200;

// --- Timing ---
static const uint16_t LOOP_HZ    = 500;
static const uint32_t LOOP_DT_MS = 1000UL / LOOP_HZ;
static const float    LOOP_DT    = 1.0f / LOOP_HZ;

// --- Control gains (defaults; tunable via protocol) ---
static const float KP = 1.0f;
static const float KI = 0.005f;
static const float KD = 0.005f;

// --- Filtering / smoothing ---
static const float ANGLE_ALPHA = 0.15f;   // 0..1 EMA
static const float SET_ANGLE_ALPHA = 0.1f;  // 0..1 EMA for desired angle commands
static const float DIR_HYST = 2.0f;       // degrees of deadband
static const uint16_t DIR_MIN_HOLD_MS = 50;
static const uint8_t PWM_SLEW = 4;        // PWM change per loop

// --- Output scaling ---
static const float OUTPUT_TO_PWM = 10.0f; // PWM ≈ |u| * scale
static const uint8_t PWM_CAP = 150;
static const uint8_t PWM_MIN = 20;

// --- Angles ---
static const float ANGLE_LIMIT = 45.0f;
static const float START_SET_ANGLE = 0.0f;

// --- Debug/Traj toggles ---
#define ENABLE_TRAJECTORY 0
#define LIGHT_DEBUG 0
#define DISABLE_CHECKSUM 1  // Set to 1 to skip checksum validation for testing

// --- Motor Control ---
static const uint8_t DIRECT_PWM_MAX = 255;  // Maximum PWM for direct control


