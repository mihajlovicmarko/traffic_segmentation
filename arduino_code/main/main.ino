// --- Pins ---
const uint8_t PWM_PIN = 10;           // Nano: 3,5,6,9,10,11
const uint8_t DIR_PIN = 3;

// --- Control targets ---
float set_angle = 20.0f;

// --- PID gains ---
const float Kp = 3.0f;
const float Ki = 0.0f;                // start at 0; add later
const float Kd = 0.005f;

// --- Loop timing (fixed-rate control) ---
const uint16_t LOOP_HZ = 500;         // 100 Hz control
const uint32_t LOOP_DT_MS = 1000UL / LOOP_HZ;
const float    LOOP_DT = 1.0f / LOOP_HZ;

// --- Filtering / smoothing ---
const float ANGLE_ALPHA = 0.15f;      // EMA low-pass (0..1), lower = smoother
const float DIR_HYST = 2.0f;          // degrees around 0 before flipping direction
const uint16_t DIR_MIN_HOLD_MS = 50;  // minimum time before next direction flip
const uint8_t PWM_SLEW = 4;           // max PWM change per loop step

// --- State ---
struct PID_Errors {
  float error;
  float d_error;
  float i_error;
};

float last_error = 0.0f;
float i_sum = 0.0f;
float angle_filt = 0.0f;
uint32_t last_loop_ms = 0;
uint32_t last_dir_change_ms = 0;
int current_dir = 0;       // 0 or 1
uint8_t current_pwm = 0;   // 0..255

// --- Helpers ---
PID_Errors computeError(float set_a, float cur_a, float last_e, float i_prev, float dt) {
  PID_Errors e;
  e.error   = set_a - cur_a;
  e.d_error = (e.error - last_e) / dt;
  e.i_error = i_prev + e.error * dt;
  return e;
}

// Safe, filtered angle from A4
float readAngleDeg() {
  // raw 0..1023
  int raw = analogRead(A4);
  float u = float(raw) / 1023.0f;

  // avoid division blow-up near 1.0
  if (u > 0.99f) u = 0.99f;
  if (u < 0.01f) u = 0.01f;

  // your mapping
  float angle = (0.25f * u / (1.0f - u) - 0.36f) * 270.0f;

  // EMA low-pass
  angle_filt = ANGLE_ALPHA * angle + (1.0f - ANGLE_ALPHA) * angle_filt;
  return angle_filt;
}

// Limit how fast PWM can change (slew limiter)
uint8_t slew(uint8_t target, uint8_t current, uint8_t step) {
  if (target > current + step) return current + step;
  if (target + step < current) return current - step;
  return target;
}

void setup() {
  pinMode(PWM_PIN, OUTPUT);
  pinMode(DIR_PIN, OUTPUT);
  Serial.begin(115200);  // faster if you keep prints

  // Initialize timing and filtered angle
  delay(50);             // let ADC settle
  angle_filt = (0.25f * (float(analogRead(A4))/1023.0f) / (1.0f - float(analogRead(A4))/1023.0f) - 0.36f) * 270.0f;
  last_loop_ms = millis();
  last_dir_change_ms = last_loop_ms;

  digitalWrite(DIR_PIN, current_dir);
  analogWrite(PWM_PIN, current_pwm);
}

void loop() {
  // run at fixed rate
  uint32_t now = millis();
  if (now - last_loop_ms < LOOP_DT_MS) return;
  last_loop_ms += LOOP_DT_MS; // keeps cadence even if we slip one tick

  // toggle target every 2 s
  static uint32_t last_toggle = 0;
  static bool toggle_state = false;
  if (now - last_toggle >= 2000) {       // 2 seconds
    toggle_state = !toggle_state;
    set_angle = toggle_state ? 20.0f : -20.0f;
    last_toggle = now;

       // filtered last angle
  }

  // read & filter sensor
  float angle = readAngleDeg();

  // PID (with fixed dt)
  PID_Errors err = computeError(set_angle, angle, last_error, i_sum, LOOP_DT);
  i_sum = err.i_error;
  float u = Kp * err.error + Ki * i_sum + Kd * err.d_error;

  // decide direction with hysteresis & min-hold
  int desired_dir = (u > DIR_HYST) ? 0 : (u < -DIR_HYST) ? 1 : current_dir;
  if (desired_dir != current_dir && (now - last_dir_change_ms) >= DIR_MIN_HOLD_MS) {
    current_dir = desired_dir;
    last_dir_change_ms = now;
    digitalWrite(DIR_PIN, current_dir);
  }

  // magnitude → PWM (cap to 0..255)
  float mag = fabs(u) * 10.0f;
  if (mag > 150.0f) mag = 150.0f;
  uint8_t target_pwm = (uint8_t)mag;

  // optional floor to overcome static friction
  const uint8_t PWM_MIN = 20;
  if (target_pwm > 0 && target_pwm < PWM_MIN) target_pwm = PWM_MIN;

  // apply slew limit for smoothness
  current_pwm = slew(target_pwm, current_pwm, PWM_SLEW);
  analogWrite(PWM_PIN, current_pwm);

  last_error = err.error;
    Serial.print("granica = ");
    Serial.print("0.0 ");
    //Serial.print("set_angle = ");
    //Serial.print(set_angle);
    Serial.print(" | read_angle = ");
    
    Serial.println(angle_filt);   
    

  // lightweight debug (comment out if not needed)
  // Serial.print("ang="); Serial.println(angle);
}

