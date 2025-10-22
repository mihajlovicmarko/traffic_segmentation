# arducom.py
"""
Lightweight Arduino comms module.
Protocol: "CMD,arg1,arg2,...,*CS\\r\\n"
CS = 2-hex-digit checksum of all bytes up to and INCLUDING '*'
Arduino replies: "ACK,<seq>,<status>,<info>,*CS"

Example Arduino commands expected:
- PING,
- SET_ANGLE,<deg>,
- GET_STATE,
- SET_KP,<val>,  SET_KI,<val>,  SET_KD,<val>,
- SET_MOTOR_PWM,<pwm_value>,

Example:
    from arducom import ArduinoClient
    cli = ArduinoClient()
    cli.connect()
    cli.ping()
    cli.set_angle(20.0)
    cli.set_motor_pwm(128)  # Set motor to half speed
    print(cli.get_state())
"""

from __future__ import annotations
import sys, time, glob
from typing import Dict, Optional, Iterator
import serial  # pip install pyserial


# ---------- Utilities ----------
def _checksum8(data: bytes) -> int:
    return sum(data) & 0xFF

def _build_line(body: str) -> bytes:
    """
    Build a command line. body should NOT include the checksum.
    Example bodies: "PING,", "SET_ANGLE,20.00,", "GET_STATE,"
    """
    if not body.endswith(","):
        body = body + ","
    payload = (body + "*").encode("ascii")
    cs = _checksum8(payload)
    return payload + f"{cs:02X}\r\n".encode("ascii")

def _parse_ack(line: bytes) -> Dict[str, str]:
    """
    Parse "ACK,<seq>,<status>,<info>,*CS" → dict with keys: seq, status, info
    Raises ValueError on format/checksum problems.
    """
    s = line.strip().decode("ascii", errors="ignore")
    star = s.rfind("*")
    if star < 0 or star + 3 > len(s):
        raise ValueError("bad format (checksum missing/short)")
    body_with_star = s[:star+1]     # includes '*'
    cs_hex = s[star+1:]
    try:
        got = int(cs_hex, 16)
    except Exception:
        raise ValueError("checksum not hex")
    want = _checksum8(body_with_star.encode("ascii"))
    if got != want:
        raise ValueError("bad checksum")

    body = body_with_star[:-1]      # drop '*'
    parts = body.split(",")
    if len(parts) < 4 or parts[0] != "ACK":
        raise ValueError("not an ACK")
    seq = parts[1]
    status = parts[2]
    info = ",".join(parts[3:])      # info may contain commas
    return {"seq": seq, "status": status, "info": info}

def list_ports() -> list[str]:
    """Cross-platform-ish serial port discovery."""
    if sys.platform.startswith("win"):
        return [f"COM{i}" for i in range(1, 256)]
    elif sys.platform.startswith("linux") or sys.platform.startswith("cygwin"):
        return glob.glob("/dev/ttyACM*") + glob.glob("/dev/ttyUSB*")
    elif sys.platform.startswith("darwin"):
        return glob.glob("/dev/tty.usbmodem*") + glob.glob("/dev/tty.usbserial*")
    return []


# ---------- Client ----------
class ArduinoClient:
    def __init__(
        self,
        port: Optional[str] = None,
        baud: int = 115200,
        timeout: float = 0.25,
        boot_wait: float = 1.8,
        retries: int = 3,
    ) -> None:
        """
        port: explicit serial port, or None to auto-scan
        baud: must match Serial.begin on Arduino
        timeout: read timeout (s)
        boot_wait: delay after opening port (s) to let board reboot
        retries: request retries before reconnect/raise
        """
        self.port = port
        self.baud = baud
        self.timeout = timeout
        self.boot_wait = boot_wait
        self.retries = retries
        self.ser: Optional[serial.Serial] = None

    # --- lifecycle ---
    def connect(self) -> None:
        """Open serial port; auto-scan if port is None."""
        if self.port:
            self._open(self.port)
            return
        for p in list_ports():
            try:
                self._open(p)
                self.port = p
                return
            except Exception:
                continue
        raise RuntimeError("No Arduino serial port found.")

    def _open(self, p: str) -> None:
        s = serial.Serial(p, self.baud, timeout=self.timeout)
        time.sleep(self.boot_wait)     # many boards auto-reset on open
        s.reset_input_buffer()
        self.ser = s

    def close(self) -> None:
        if self.ser:
            try:
                self.ser.close()
            finally:
                self.ser = None

    def __enter__(self) -> "ArduinoClient":
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _ensure(self) -> None:
        if self.ser is None or not self.ser.is_open:
            # attempt reconnect on same port (or rescan if unknown)
            self.connect()

    # --- core request API ---
    def request(self, body: str) -> Dict[str, str]:
        """
        Send one command and parse one ACK dict.
        On final failure: drops connection and raises.
        """
        pkt = _build_line(body)
        last_err: Optional[Exception] = None
        for attempt in range(1, self.retries + 1):
            try:
                self._ensure()
                self.ser.write(pkt)       # type: ignore[union-attr]
                self.ser.flush()          # type: ignore[union-attr]
                line = self.ser.readline()  # type: ignore[union-attr]
                if not line:
                    raise TimeoutError("no response")
                return _parse_ack(line)
            except Exception as e:
                last_err = e
                # final attempt → close and raise
                if attempt == self.retries:
                    try:
                        self.close()
                    except Exception:
                        pass
                    raise last_err
                # brief backoff then retry
                time.sleep(0.1)
        # unreachable
        raise RuntimeError("request() failed unexpectedly")

    # --- convenience methods ---
    def ping(self) -> bool:
        res = self.request("PING,")
        return res.get("status") == "OK"

    def set_angle(self, angle_deg: float) -> bool:
        # Clamp host-side too (match your firmware’s ANGLE_LIMIT if desired)
        if angle_deg < -90: angle_deg = -90
        if angle_deg >  90: angle_deg =  90
        res = self.request(f"SET_ANGLE,{angle_deg:.2f},")
        return res.get("status") == "OK"

    def set_kp(self, kp: float) -> bool:
        res = self.request(f"SET_KP,{kp:.6f},")
        return res.get("status") == "OK"

    def set_ki(self, ki: float) -> bool:
        res = self.request(f"SET_KI,{ki:.6f},")
        return res.get("status") == "OK"

    def set_kd(self, kd: float) -> bool:
        res = self.request(f"SET_KD,{kd:.6f},")
        return res.get("status") == "OK"

    def set_motor_pwm(self, pwm_value: int) -> bool:
        """
        Set PWM value for the main motor.
        pwm_value: PWM value (typically 0-255, but depends on Arduino implementation)
        """
        # Clamp PWM value to valid range (0-255 for 8-bit PWM)
        if pwm_value < 0: pwm_value = 0
        if pwm_value > 255: pwm_value = 255
        res = self.request(f"SET_MOTOR_PWM,{pwm_value},")
        return res.get("status") == "OK"

    def set_motor2_pwm(self, pwm_value: int) -> bool:
        """
        Set PWM value for the second motor (driving motor).
        pwm_value: PWM value (typically 0-255, but depends on Arduino implementation)
        """
        # Clamp PWM value to valid range (0-255 for 8-bit PWM)
        if pwm_value < 0: pwm_value = 0
        if pwm_value > 255: pwm_value = 255
        res = self.request(f"SET_MOTOR2_PWM,{pwm_value},")
        return res.get("status") == "OK"

    def get_state(self) -> Dict[str, str]:
        """Returns dict parsed from info (e.g., {'ang':'12.3','pwm':'110','dir':'0', ...})."""
        res = self.request("GET_STATE,")
        info = res.get("info", "")
        out: Dict[str, str] = {}
        for kv in info.split(","):
            if "=" in kv:
                k, v = kv.split("=", 1)
                out[k.strip()] = v.strip()
        out["_status"] = res.get("status", "")
        out["_seq"] = res.get("seq", "")
        return out

    # --- simple polling generator ---
    def watch_state(self, period_s: float = 0.2) -> Iterator[Dict[str, str]]:
        """
        Poll GET_STATE at fixed intervals; yields dict each cycle.
        Usage:
            for st in cli.watch_state(0.2):
                print(st)
        Ctrl+C to stop.
        """
        next_t = time.monotonic()
        while True:
            next_t += period_s
            try:
                yield self.get_state()
            except Exception as e:
                # surface error but keep trying to reconnect
                yield {"_error": str(e)}
                time.sleep(0.5)
            # sleep until next tick
            now = time.monotonic()
            if next_t > now:
                time.sleep(next_t - now)
            else:
                next_t = now  # catch up if we fell behind


if __name__ == "__main__":
    with ArduinoClient() as cli:
        print("Ping:", cli.ping())
        cli.set_kp(3.0); cli.set_ki(0.0); cli.set_kd(0.005)

        target = 10.0
        last = time.monotonic()

        for state in cli.watch_state(0.2):  # ~5 Hz telemetry
            print(state)
            if time.monotonic() - last > 2.0:
                target = -target
                cli.set_angle(target)
                last = time.monotonic()