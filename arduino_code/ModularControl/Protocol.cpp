// Protocol.cpp
#include "Protocol.h"
#include "Config.h"

void Protocol::begin(uint16_t) {
  // Serial.begin(...) is done in main
  line_.reserve(128);
}

bool Protocol::pollForCommand(Cmd& cmd) {
  static uint32_t lastChar = 0;
  
  while (Serial.available()) {
    char c = Serial.read();
    lastChar = millis();
    if (c == '\n' || c == '\r') {
      if (line_.length()) {
        line_.trim();  // Remove any whitespace/newlines
        Serial.print("RX: "); Serial.println(line_);  // Temporary debug
        if (parseLineToCmd(line_, cmd)) {
          line_ = "";
          return true;  // Command ready
        } else {
          ackErr("BAD_FMT_OR_CSUM");
        }
      }
      line_ = "";
    } else {
      line_ += c;
      if (line_.length() > 120) line_ = ""; // avoid runaway
    }
  }
  
  // Process command after 100ms of no new characters (timeout)
  if (line_.length() > 0 && (millis() - lastChar) > 100) {
    line_.trim();  // Remove any whitespace/newlines
    Serial.print("RX: "); Serial.println(line_);  // Temporary debug
    if (parseLineToCmd(line_, cmd)) {
      line_ = "";
      return true;  // Command ready
    } else {
      ackErr("BAD_FMT_OR_CSUM");
    }
    line_ = "";
  }
  
  return false;  // No command ready
}

bool Protocol::parseLineToCmd(const String& s, Cmd& out) {
  // Expect: "CMD,<args...>,*CS"
  int star = s.lastIndexOf('*');
  if (star <= 0 || star + 2 >= s.length()) return false;

  String body = s.substring(0, star + 1); // includes '*'
  String csHex = s.substring(star + 1);
  char csBuf[4] = {0};
  csHex.toCharArray(csBuf, sizeof(csBuf));
  int got = strtoul(csBuf, NULL, 16);

  // checksum over body (including '*')
  char tmp[140];
  body.toCharArray(tmp, sizeof(tmp));
  int want = ProtocolUtil::checksum8(tmp);
  Serial.print("CS: got="); Serial.print(got, HEX); 
  Serial.print(" want="); Serial.println(want, HEX);
  
#if !DISABLE_CHECKSUM
  if (got != want) return false;
#endif

  // Remove trailing '*'
  body.remove(body.length() - 1);

  // Split by commas
  int firstComma = body.indexOf(',');
  String cmd = (firstComma < 0) ? body : body.substring(0, firstComma);
  String args = (firstComma < 0) ? ""   : body.substring(firstComma + 1);

  cmd.toUpperCase();

  out = Cmd{};
  if (cmd == "PING") { out.type = PING; return true; }
  if (cmd == "GET_STATE") { out.type = GET_STATE; return true; }

  // Commands with 1 float argument: SET_ANGLE, SET_KP/KI/KD
  float val = 0.0f;
  if (args.length()) {
    int lastComma = args.lastIndexOf(',');
    String num = (lastComma >= 0) ? args.substring(0, lastComma) : args;
    val = num.toFloat();
  }

  if (cmd == "SET_ANGLE") { out.type = SET_ANGLE; out.value = val; return true; }
  if (cmd == "SET_KP")    { out.type = SET_KP;    out.value = val; return true; }
  if (cmd == "SET_KI")    { out.type = SET_KI;    out.value = val; return true; }
  if (cmd == "SET_KD")    { out.type = SET_KD;    out.value = val; return true; }

  return false;
}

void Protocol::sendAck(const char* status, const char* info) {
  // "ACK,<seq>,<status>,<info>,*CS\r\n"
  String body = "ACK,";
  body += seq_++;
  body += ",";
  body += status;
  body += ",";
  body += info;
  body += ",*"; // include '*' for checksum coverage

  // calc checksum over body (including '*')
  char tmp[200];
  body.toCharArray(tmp, sizeof(tmp));
  uint8_t cs = ProtocolUtil::checksum8(tmp);

  char csHex[3];
  snprintf(csHex, sizeof(csHex), "%02X", cs);

  Serial.print(body);
  Serial.print(csHex);
  Serial.print("\r\n");
}

void Protocol::ackOK(const char* info)  { sendAck("OK",  info); }
void Protocol::ackErr(const char* info) { sendAck("ERR", info); }