#include "Config.h"
#include "Types.h"
#include "PIDController.h"
#include "AngleSensor.h"
#include "MotorDriver.h"
#include "Protocol.h"
#include "Trajectory.h"

PIDController pid(KP, KI, KD);
AngleSensor  sensor(A4, ANGLE_ALPHA);
MotorDriver  motor(PWM_PIN, DIR_PIN, PWM_SLEW, DIR_HYST, DIR_MIN_HOLD_MS);
Protocol     proto;

static uint32_t last_loop_ms = 0;
static float set_angle = START_SET_ANGLE;

void setup() {
  pinMode(PWM_PIN, OUTPUT);
  pinMode(DIR_PIN, OUTPUT);
  Serial.begin(SERIAL_BAUD);
  delay(50);                                     // let ADC settle
  sensor.init();
  motor.begin();
  proto.begin();

  last_loop_ms = millis();
}

void loop() {
  // Fixed-rate loop
  uint32_t now = millis();
  if (now - last_loop_ms < LOOP_DT_MS) {
    // Process serial in the slack (non-blocking)
    Protocol::Cmd cmd;
    if (proto.pollForCommand(cmd)) {
      // Handle incoming commands
      if (cmd.type == Protocol::PING) {
        proto.ackOK("PONG");
      } else if (cmd.type == Protocol::SET_ANGLE) {
        set_angle = constrain(cmd.value, -ANGLE_LIMIT, ANGLE_LIMIT);
        proto.ackOK("SET");
      } else if (cmd.type == Protocol::GET_STATE) {
        char buf[64];
        snprintf(buf, sizeof(buf), "ang=%.2f,pwm=%u,dir=%d",
                 sensor.lastFiltered(), motor.currentPwm(), motor.currentDir());
        proto.ackOK(buf);
      } else if (cmd.type == Protocol::SET_KP) {
        pid.Kp = cmd.value; proto.ackOK("KP");
      } else if (cmd.type == Protocol::SET_KI) {
        pid.Ki = cmd.value; proto.ackOK("KI");
      } else if (cmd.type == Protocol::SET_KD) {
        pid.Kd = cmd.value; proto.ackOK("KD");
      } else {
        proto.ackErr("UNKNOWN_CMD");
      }
    }
    return;
  }
  last_loop_ms += LOOP_DT_MS;

  // Optional: generate a trajectory on the target (can disable in Config.h)
#if ENABLE_TRAJECTORY
  set_angle = trajectory(now);
#endif

  // Read sensor
  float angle = sensor.readDeg();

  // PID compute (fixed dt)
  PID_Errors e = pid.compute(set_angle, angle, LOOP_DT);

  // Control output -> direction + PWM
  float u = e.u;                            // from PIDController
  motor.applyControl(u);

  // (Optional) lightweight debug each 200 ms
#if LIGHT_DEBUG
  static uint32_t last_dbg = 0;
  if (now - last_dbg >= 200) {
    last_dbg = now;
    Serial.print("set="); Serial.print(set_angle, 1);
    Serial.print(" ang="); Serial.print(sensor.lastFiltered(), 1);
    Serial.print(" pwm="); Serial.print(motor.currentPwm());
    Serial.print(" dir="); Serial.println(motor.currentDir());
  }
#endif
}
