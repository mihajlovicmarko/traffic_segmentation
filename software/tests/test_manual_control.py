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

class TestManualControl:
    """Test version of manual control using CV2 window input."""
    
    def __init__(self):
        self.current_angle = 0.0
        self.current_pwm = 0
        self.running = False
        self.max_angle = 90.0
        self.angle_step = 5.0
        self.max_pwm = 120
        self.pwm_step = 10
        
        # Key codes for different platforms
        self.key_codes = {
            'up': [2490368, 82, 0],      # Up arrow (different platforms/OpenCV versions)
            'down': [2621440, 84, 1],    # Down arrow
            'left': [2424832, 81, 2],    # Left arrow  
            'right': [2555904, 83, 3],   # Right arrow
            'stop': [ord('s'), ord('S')], # S key
            'quit': [ord('q'), ord('Q'), 27]  # Q key and ESC
        }
        
    def start(self):
        """Start the CV2 keyboard test."""
        self.running = True
        print("Manual Control Test Started (CV2 Window Mode)")
        print("============================================")
        print("Controls (focus on the CV2 window):")
        print("  ↑ (Up Arrow)    - Increase forward speed")
        print("  ↓ (Down Arrow)  - Increase reverse speed")
        print("  ← (Left Arrow)  - Turn left")
        print("  → (Right Arrow) - Turn right")
        print("  s               - Stop (reset to 0)")
        print("  q/ESC           - Quit")
        print("\nA control window will open. Click on it and use arrow keys.")
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
                
                # Wait for key press (1ms timeout for responsiveness)
                key = cv2.waitKey(1) & 0xFF
                
                # Process the key
                if self._process_key(key):
                    break
                    
                # Small delay for CPU
                time.sleep(0.01)
        
        except Exception as e:
            print(f"CV2 handler error: {e}")
        finally:
            cv2.destroyAllWindows()
            print("CV2 windows closed")
    
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
        
        # Controls instructions
        y_start = 320
        cv2.putText(img, "Controls:", (50, y_start), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(img, "↑↓ = Speed   ←→ = Steering", (50, y_start + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(img, "S = Stop     Q/ESC = Quit", (50, y_start + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        return img
    
    def _process_key(self, key):
        """Process a key from cv2.waitKey(). Returns True if quit was requested."""
        if key == 255 or key == -1:  # No key pressed
            return False
            
        # Handle different key codes for arrow keys
        if key in self.key_codes['up']:
            self._press_key('up')
        elif key in self.key_codes['down']:
            self._press_key('down')
        elif key in self.key_codes['left']:
            self._press_key('left')
        elif key in self.key_codes['right']:
            self._press_key('right')
        elif key in self.key_codes['stop']:
            self._press_key('stop')
        elif key in self.key_codes['quit']:
            self._press_key('quit')
            return True
        else:
            # Debug: print unknown key codes to help with platform differences
            print(f"Debug: Key code {key} pressed")
        
        return False
    
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
    print("Manual Control Test using CV2 Windows")
    print("=====================================")
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