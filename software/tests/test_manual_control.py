#!/usr/bin/env python3
"""
Simple test script for manual control functionality.
This script tests the keyboard input handling without requiring the full socket client setup.
Run this to verify manual control works on your Raspberry Pi before running the full client.
"""

import sys
import os
import time
import threading
import argparse

# Platform-specific imports for keyboard handling
if sys.platform == 'win32':
    import msvcrt
else:
    import select
    import tty
    import termios

class TestManualControl:
    """Test version of manual control for debugging."""
    
    def __init__(self):
        self.current_angle = 0.0
        self.current_pwm = 0
        self.running = False
        self.max_angle = 90.0
        self.angle_step = 5.0
        self.max_pwm = 120
        self.pwm_step = 10
        
    def start(self):
        """Start the keyboard test."""
        self.running = True
        print("Manual Control Test Started")
        print("Controls:")
        print("  ↑ (Up Arrow)    - Increase forward speed")
        print("  ↓ (Down Arrow)  - Increase reverse speed")
        print("  ← (Left Arrow)  - Turn left")
        print("  → (Right Arrow) - Turn right")
        print("  s               - Stop (reset to 0)")
        print("  q               - Quit")
        print("\nCurrent values will be displayed as you press keys.")
        print("Press keys now...\n")
        
        self._keyboard_handler()
    
    def _keyboard_handler(self):
        """Handle keyboard input."""
        old_settings = None
        try:
            # Save terminal settings for Linux/Raspberry Pi
            if sys.platform != 'win32':
                try:
                    old_settings = termios.tcgetattr(sys.stdin)
                    tty.setraw(sys.stdin.fileno())
                    print("Terminal set to raw mode for Linux/Raspberry Pi")
                except Exception as e:
                    print(f"Warning: Could not set raw terminal mode: {e}")
            
            while self.running:
                if sys.platform == 'win32':
                    # Windows keyboard handling
                    if msvcrt.kbhit():
                        key = msvcrt.getch()
                        if key == b'\xe0':  # Special key prefix
                            key = msvcrt.getch()
                            self._handle_special_key(key)
                        else:
                            self._handle_regular_key(key.decode('utf-8', errors='ignore'))
                else:
                    # Linux/Raspberry Pi keyboard handling
                    if select.select([sys.stdin], [], [], 0.05)[0]:
                        key = sys.stdin.read(1)
                        if key == '\x1b':  # ESC sequence (arrow keys)
                            # Read the rest of the escape sequence
                            if select.select([sys.stdin], [], [], 0.05)[0]:
                                key += sys.stdin.read(1)
                                if key == '\x1b[':
                                    if select.select([sys.stdin], [], [], 0.05)[0]:
                                        key += sys.stdin.read(1)
                                        self._handle_arrow_key(key)
                        else:
                            self._handle_regular_key(key)
                
                time.sleep(0.02)  # 50 Hz update rate
        
        except Exception as e:
            print(f"Keyboard handler error: {e}")
        finally:
            # Restore terminal settings on Linux/Raspberry Pi
            if sys.platform != 'win32' and old_settings is not None:
                try:
                    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
                    print("\nTerminal settings restored")
                except Exception as e:
                    print(f"Warning: Could not restore terminal settings: {e}")
    
    def _handle_special_key(self, key):
        """Handle Windows special keys (arrow keys)."""
        if key == b'H':  # Up arrow
            self._press_key('up')
        elif key == b'P':  # Down arrow
            self._press_key('down')
        elif key == b'K':  # Left arrow
            self._press_key('left')
        elif key == b'M':  # Right arrow
            self._press_key('right')
    
    def _handle_arrow_key(self, key_seq):
        """Handle Linux arrow key sequences."""
        if key_seq == '\x1b[A':  # Up arrow
            self._press_key('up')
        elif key_seq == '\x1b[B':  # Down arrow
            self._press_key('down')
        elif key_seq == '\x1b[D':  # Left arrow
            self._press_key('left')
        elif key_seq == '\x1b[C':  # Right arrow
            self._press_key('right')
    
    def _handle_regular_key(self, key):
        """Handle regular keys."""
        if key.lower() == 's':
            self._press_key('stop')
        elif key.lower() == 'q':
            self._press_key('quit')
            self.running = False
    
    def _press_key(self, key):
        """Handle key press events."""
        if key == 'up':
            self.current_pwm = min(self.max_pwm, self.current_pwm + self.pwm_step)
            print(f"\r🔼 Forward PWM: {self.current_pwm:3d} | Steering: {self.current_angle:+6.1f}°", end='', flush=True)
        elif key == 'down':
            self.current_pwm = max(-self.max_pwm, self.current_pwm - self.pwm_step)
            print(f"\r🔽 Reverse PWM: {self.current_pwm:3d} | Steering: {self.current_angle:+6.1f}°", end='', flush=True)
        elif key == 'left':
            self.current_angle = max(-self.max_angle, self.current_angle - self.angle_step)
            print(f"\r◀️ Left angle: {self.current_angle:+6.1f}° | PWM: {self.current_pwm:3d}", end='', flush=True)
        elif key == 'right':
            self.current_angle = min(self.max_angle, self.current_angle + self.angle_step)
            print(f"\r▶️ Right angle: {self.current_angle:+6.1f}° | PWM: {self.current_pwm:3d}", end='', flush=True)
        elif key == 'stop':
            self.current_pwm = 0
            self.current_angle = 0.0
            print(f"\r⏹️ STOP - PWM: {self.current_pwm:3d} | Angle: {self.current_angle:+6.1f}°", end='', flush=True)
        elif key == 'quit':
            print(f"\r❌ QUIT requested - Final PWM: {self.current_pwm:3d} | Angle: {self.current_angle:+6.1f}°")

def main():
    print("Manual Control Test for Raspberry Pi")
    print("====================================")
    print(f"Platform detected: {sys.platform}")
    
    if sys.platform != 'win32':
        print("Linux/Unix platform - using termios for keyboard input")
    else:
        print("Windows platform - using msvcrt for keyboard input")
    
    print()
    
    controller = TestManualControl()
    try:
        controller.start()
    except KeyboardInterrupt:
        print("\n\nKeyboard interrupt received - exiting")
    except Exception as e:
        print(f"\nError: {e}")
    
    print("Test completed.")

if __name__ == "__main__":
    main()