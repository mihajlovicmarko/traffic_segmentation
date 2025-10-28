#!/usr/bin/env python3
"""
Simple test script for manual control functionality using CV2 window input.
This script tests the keyboard input handling using OpenCV windows.
Run this to verify manual control works on your Raspberry Pi before running the full client.
"""

import sys
import os
import time
import cv2
import numpy as np
import argparse
import logging

# Add parent directory to Python path to import arducom
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

try:
    from arducom import ArduinoClient
    ARDUINO_AVAILABLE = True
    print("Arduino communication module loaded successfully")
except ImportError as e:
    print(f"Warning: Arduino communication not available: {e}")
    ARDUINO_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

class TestManualControl:
    """Test version of manual control using CV2 window input with Arduino support."""
    
    def __init__(self, args):
        self.current_angle = 0.0
        self.current_pwm = 0
        self.running = False
        self.max_angle = args.manual_max_angle
        self.angle_step = args.manual_angle_step
        self.max_pwm = args.manual_max_pwm
        self.pwm_step = args.manual_pwm_step
        
        # Arduino setup
        self.arduino = None
        self.arduino_enabled = args.arduino
        self.arduino_port = args.arduino_port
        
        if self.arduino_enabled and ARDUINO_AVAILABLE:
            self._setup_arduino()
        elif self.arduino_enabled and not ARDUINO_AVAILABLE:
            logging.error("Arduino communication requested but arducom module not available")
        
        # Use WASD keys which are more reliable than arrow keys in OpenCV
        self.use_wasd = True
        print("Using WASD controls for better compatibility:")
        print("  W = Forward, S = Reverse, A = Left, D = Right")
        print("  Space = Stop, Q/ESC = Quit")
    
    def _setup_arduino(self):
        """Setup Arduino connection."""
        try:
            self.arduino = ArduinoClient(port=self.arduino_port)
            self.arduino.connect()
            logging.info(f"Arduino connected on port: {self.arduino.port}")
            
            # Test connection
            if self.arduino.ping():
                logging.info("Arduino ping successful")
                
                # Initialize motors to stopped state
                success_angle = self.arduino.set_angle(0.0)
                success_pwm = self.arduino.set_motor2_pwm(0)
                
                if success_angle and success_pwm:
                    logging.info("Arduino: Initialized to stopped state (angle=0°, PWM=0)")
                else:
                    logging.warning("Arduino: Failed to initialize to stopped state")
            else:
                logging.warning("Arduino ping failed")
                
        except Exception as e:
            logging.error(f"Failed to connect to Arduino: {e}")
            self.arduino = None
    
    def _send_to_arduino(self, angle=None, pwm=None):
        """Send commands to Arduino if connected."""
        if not self.arduino:
            return
        
        try:
            # Send steering angle
            if angle is not None:
                success = self.arduino.set_angle(angle)
                if success:
                    logging.info(f"Arduino: Set steering angle to {angle:.1f}°")
                else:
                    logging.warning(f"Arduino: Failed to set angle {angle:.1f}°")
            
            # Send PWM command
            if pwm is not None:
                # Handle reverse PWM (negative values)
                if pwm < 0:
                    # For now, use absolute value for reverse
                    abs_pwm = abs(pwm)
                    success = self.arduino.set_motor2_pwm(abs_pwm)
                    if success:
                        logging.info(f"Arduino: Set REVERSE driving speed to PWM {abs_pwm} (original: {pwm})")
                    else:
                        logging.warning(f"Arduino: Failed to set reverse driving speed {abs_pwm}")
                else:
                    success = self.arduino.set_motor2_pwm(pwm)
                    if success:
                        logging.info(f"Arduino: Set driving speed to PWM {pwm}")
                    else:
                        logging.warning(f"Arduino: Failed to set driving speed {pwm}")
                        
        except Exception as e:
            logging.error(f"Arduino communication error: {e}")
            # Emergency stop on exception
            try:
                self.arduino.set_motor2_pwm(0)
                logging.info("Emergency stop activated")
            except:
                pass
        
    def start(self):
        """Start the CV2 keyboard test."""
        self.running = True
        print("Manual Control Test Started (CV2 Window Mode)")
        print("============================================")
        
        # Show Arduino status
        if self.arduino_enabled:
            if self.arduino:
                print(f"🟢 Arduino: CONNECTED on {self.arduino.port}")
                print("   Commands will be sent to real Arduino!")
            else:
                print("🔴 Arduino: FAILED TO CONNECT")
                print("   Running in simulation mode only")
        else:
            print("⚪ Arduino: DISABLED")
            print("   Running in simulation mode only")
        
        print("\nControls (focus on the CV2 window):")
        print("  W - Increase forward speed")
        print("  S - Increase reverse speed")  
        print("  A - Turn left")
        print("  D - Turn right")
        print("  SPACE - Emergency stop (reset to 0)")
        print("  Q/ESC - Quit")
        print(f"\nSettings: Max Angle=±{self.max_angle}°, Max PWM=±{self.max_pwm}")
        print("A control window will open. Click on it to focus, then use WASD keys.")
        print("Key presses will be shown in console for debugging.")
        print("Current values will be displayed in the window and console.\n")
        
        self._cv2_handler()
    
    def _cv2_handler(self):
        """Handle keyboard input using CV2 window."""
        # Create a control display window
        window_name = "Manual Control Test"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 600, 400)
        
        print("CV2 window opened. Click on it to focus, then use arrow keys.")
        
        try:
            while self.running:
                # Create control display image
                img = self._create_control_display()
                
                # Show the image
                cv2.imshow(window_name, img)
                
                # Wait for key press (30ms timeout for better key detection)
                key = cv2.waitKey(30) & 0xFF
                
                # Process the key
                if self._process_key(key):
                    break
        
        except Exception as e:
            print(f"CV2 handler error: {e}")
        finally:
            cv2.destroyAllWindows()
            print("CV2 windows closed")
            self.cleanup()
    
    def _create_control_display(self):
        """Create a visual display showing current control state."""
        # Create black image
        img = np.zeros((400, 600, 3), dtype=np.uint8)
        
        # Title
        cv2.putText(img, "Manual Control Test", (150, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        
        # Current values
        cv2.putText(img, f"Steering Angle: {self.current_angle:+6.1f}°", (50, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(img, f"PWM Speed: {self.current_pwm:+4d}", (50, 140), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # Draw steering wheel visualization
        center_x, center_y = 300, 220
        wheel_radius = 60
        
        # Steering wheel circle
        cv2.circle(img, (center_x, center_y), wheel_radius, (100, 100, 100), 2)
        
        # Steering indicator
        angle_rad = np.radians(self.current_angle)
        end_x = int(center_x + wheel_radius * 0.8 * np.sin(angle_rad))
        end_y = int(center_y - wheel_radius * 0.8 * np.cos(angle_rad))
        cv2.line(img, (center_x, center_y), (end_x, end_y), (0, 255, 0), 3)
        
        # PWM bar visualization
        bar_x, bar_y = 450, 180
        bar_width, bar_height = 20, 120
        
        # PWM bar background
        cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
        
        # PWM level
        pwm_ratio = self.current_pwm / self.max_pwm
        if pwm_ratio > 0:  # Forward
            fill_height = int(bar_height * pwm_ratio * 0.5)
            cv2.rectangle(img, (bar_x, bar_y + bar_height//2 - fill_height), 
                         (bar_x + bar_width, bar_y + bar_height//2), (0, 255, 0), -1)
        elif pwm_ratio < 0:  # Reverse
            fill_height = int(bar_height * abs(pwm_ratio) * 0.5)
            cv2.rectangle(img, (bar_x, bar_y + bar_height//2), 
                         (bar_x + bar_width, bar_y + bar_height//2 + fill_height), (0, 0, 255), -1)
        
        # Center line
        cv2.line(img, (bar_x, bar_y + bar_height//2), (bar_x + bar_width, bar_y + bar_height//2), (255, 255, 255), 1)
        
        # Arduino status
        if self.arduino_enabled:
            if self.arduino:
                arduino_text = "Arduino: CONNECTED"
                arduino_color = (0, 255, 0)  # Green
            else:
                arduino_text = "Arduino: DISCONNECTED"
                arduino_color = (0, 0, 255)  # Red
        else:
            arduino_text = "Arduino: DISABLED"
            arduino_color = (128, 128, 128)  # Gray
        
        cv2.putText(img, arduino_text, (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, arduino_color, 2)
        
        # Controls instructions
        y_start = 330
        cv2.putText(img, "Controls (WASD):", (50, y_start), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(img, "W=Forward S=Reverse A=Left D=Right", (50, y_start + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(img, "SPACE=Stop   Q/ESC=Quit", (50, y_start + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(img, "Key presses shown in console", (50, y_start + 65), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        
        return img
    
    def _process_key(self, key):
        """Process a key from cv2.waitKey(). Returns True if quit was requested."""
        if key == 255 or key == -1:  # No key pressed
            return False
        
        # Show what key was pressed for debugging
        if 32 <= key <= 126:  # Printable ASCII
            key_char = chr(key)
            print(f"Key pressed: '{key_char}' (code: {key})")
        else:
            print(f"Special key pressed: {key} (0x{key:X})")
            
        # Use simple character-based controls (more reliable than arrow keys)
        key_char = key if key < 127 else 0
        
        if key_char == ord('w') or key_char == ord('W'):
            self._press_key('up')
        elif key_char == ord('s') or key_char == ord('S'):
            self._press_key('down')
        elif key_char == ord('a') or key_char == ord('A'):
            self._press_key('left')
        elif key_char == ord('d') or key_char == ord('D'):
            self._press_key('right')
        elif key_char == ord(' ') or key_char == 32:  # Spacebar
            self._press_key('stop')
        elif key_char == ord('q') or key_char == ord('Q') or key == 27:  # ESC
            self._press_key('quit')
            return True
        else:
            # Try arrow keys with common codes
            if key == 2490368 or key == 82:  # Up arrow
                self._press_key('up')
            elif key == 2621440 or key == 84:  # Down arrow
                self._press_key('down')
            elif key == 2424832 or key == 81:  # Left arrow
                self._press_key('left')
            elif key == 2555904 or key == 83:  # Right arrow
                self._press_key('right')
            else:
                print(f"Unknown key. Use: W=forward, S=reverse, A=left, D=right, Space=stop, Q=quit")
        
        return False
    
    def _press_key(self, key):
        """Handle key press events and send commands to Arduino."""
        if key == 'up':
            self.current_pwm = min(self.max_pwm, self.current_pwm + self.pwm_step)
            print(f"\r🔼 Forward PWM: {self.current_pwm:3d} | Steering: {self.current_angle:+6.1f}°", end='', flush=True)
            self._send_to_arduino(pwm=self.current_pwm)
            
        elif key == 'down':
            self.current_pwm = max(-self.max_pwm, self.current_pwm - self.pwm_step)
            print(f"\r🔽 Reverse PWM: {self.current_pwm:3d} | Steering: {self.current_angle:+6.1f}°", end='', flush=True)
            self._send_to_arduino(pwm=self.current_pwm)
            
        elif key == 'left':
            self.current_angle = max(-self.max_angle, self.current_angle - self.angle_step)
            print(f"\r◀️ Left angle: {self.current_angle:+6.1f}° | PWM: {self.current_pwm:3d}", end='', flush=True)
            self._send_to_arduino(angle=self.current_angle)
            
        elif key == 'right':
            self.current_angle = min(self.max_angle, self.current_angle + self.angle_step)
            print(f"\r▶️ Right angle: {self.current_angle:+6.1f}° | PWM: {self.current_pwm:3d}", end='', flush=True)
            self._send_to_arduino(angle=self.current_angle)
            
        elif key == 'stop':
            self.current_pwm = 0
            self.current_angle = 0.0
            print(f"\r⏹️ STOP - PWM: {self.current_pwm:3d} | Angle: {self.current_angle:+6.1f}°", end='', flush=True)
            # Send both commands for emergency stop
            self._send_to_arduino(angle=self.current_angle, pwm=self.current_pwm)
            
        elif key == 'quit':
            print(f"\r❌ QUIT requested - Final PWM: {self.current_pwm:3d} | Angle: {self.current_angle:+6.1f}°")
            # Emergency stop before quit
            if self.arduino:
                self._send_to_arduino(angle=0.0, pwm=0)
    
    def cleanup(self):
        """Clean up Arduino connection."""
        if self.arduino:
            try:
                # Emergency stop
                self.arduino.set_angle(0.0)
                self.arduino.set_motor2_pwm(0)
                logging.info("Arduino: Emergency stop - all motors stopped")
                self.arduino.close()
                logging.info("Arduino connection closed")
            except Exception as e:
                logging.warning(f"Error during Arduino cleanup: {e}")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Manual Control Test with Arduino Support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_manual_control.py                    # Simulation only
  python test_manual_control.py --arduino          # With Arduino (auto-detect port)
  python test_manual_control.py --arduino --arduino-port COM3  # Specific port
  python test_manual_control.py --arduino --manual-max-pwm 80  # Custom PWM limit
        """
    )
    
    # Arduino settings
    parser.add_argument("--arduino", action="store_true", 
                       help="Enable Arduino communication")
    parser.add_argument("--arduino-port", default=None, 
                       help="Specific Arduino serial port (auto-detect if not specified)")
    
    # Manual control parameters
    parser.add_argument("--manual-max-angle", type=float, default=90.0,
                       help="Maximum steering angle in degrees (default: 90.0)")
    parser.add_argument("--manual-angle-step", type=float, default=5.0,
                       help="Steering angle increment per key press in degrees (default: 5.0)")
    parser.add_argument("--manual-max-pwm", type=int, default=120,
                       help="Maximum PWM for driving control (default: 120)")
    parser.add_argument("--manual-pwm-step", type=int, default=10,
                       help="PWM increment per key press (default: 10)")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    print("Manual Control Test using CV2 Windows with Arduino Support")
    print("==========================================================")
    print(f"Platform detected: {sys.platform}")
    print("Using OpenCV window-based keyboard input")
    print("This works great on Raspberry Pi and all platforms!")
    print()
    
    # Check if CV2 is available
    try:
        import cv2
        print(f"OpenCV version: {cv2.__version__}")
    except ImportError:
        print("ERROR: OpenCV (cv2) is not installed!")
        print("Install it with: pip install opencv-python")
        return
    
    # Show configuration
    print(f"\nConfiguration:")
    print(f"  Arduino: {'ENABLED' if args.arduino else 'DISABLED'}")
    if args.arduino:
        print(f"  Arduino Port: {args.arduino_port or 'auto-detect'}")
    print(f"  Max Steering Angle: ±{args.manual_max_angle}°")
    print(f"  Angle Step: {args.manual_angle_step}°")
    print(f"  Max PWM: ±{args.manual_max_pwm}")
    print(f"  PWM Step: {args.manual_pwm_step}")
    print()
    
    controller = TestManualControl(args)
    try:
        controller.start()
    except KeyboardInterrupt:
        print("\n\nKeyboard interrupt received - exiting")
        controller.cleanup()
    except Exception as e:
        print(f"\nError: {e}")
        controller.cleanup()
    
    print("Test completed.")

if __name__ == "__main__":
    main()