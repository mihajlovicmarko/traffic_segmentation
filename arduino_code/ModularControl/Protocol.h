// Protocol.h
#pragma once
#include <Arduino.h>

namespace ProtocolUtil {
  inline uint8_t checksum8(const char* s) {
    uint16_t sum = 0;
    while (*s) sum += (uint8_t)(*s++);
    return (uint8_t)(sum & 0xFF);
  }
}

class Protocol {
public:
  enum CmdType { NONE, PING, SET_ANGLE, GET_STATE, SET_KP, SET_KI, SET_KD, SET_MOTOR_PWM, SET_MOTOR2_PWM };

  struct Cmd {
    CmdType type = NONE;
    float   value = 0.0f;  // used for SET_* commands
  };

  void begin(uint16_t bufSize = 128);
  bool pollForCommand(Cmd& cmd);  // Returns true if command received

  void ackOK(const char* info);
  void ackErr(const char* info);

private:
  String line_;
  uint8_t seq_ = 0;

  bool parseLineToCmd(const String& s, Cmd& out);
  void sendAck(const char* status, const char* info);
};