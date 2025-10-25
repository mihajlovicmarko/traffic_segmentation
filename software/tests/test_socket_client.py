# client_async.py
"""
Autonomous vehicle client with dual-window display and Arduino integration.

Display Windows:
- Bird's Eye View: Road detection and navigation visualization (left window)
- Combined Camera View: Side-by-side camera feeds with status info (right window)

Controls:
- Press 'q' in any window to quit

Features:
- Real-time road detection and collision avoidance
- Arduino-based steering and driving control
- Advanced obstacle avoidance with scanning and path selection
- Configurable PWM control for motors
- Live visualization of detection results

Example usage:
python test_socket_client.py --arduino --enable-driving --show --payload viz
python test_socket_client.py --arduino --camera1 0 --camera2 1 --payload viz --show  # Live cameras
python test_socket_client.py --arduino --enable-driving --enable-obstacle-avoidance --show --payload viz  # With obstacle avoidance
"""
import os, time, socket, struct, logging, argparse
import cv2, numpy as np
from concurrent.futures import ThreadPoolExecutor
import threading
import sys
import os
from enum import Enum
# Add parent directory to Python path to import arducom
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

try:
    from arducom import ArduinoClient
    ARDUINO_AVAILABLE = True
    logging.info("Arduino communication module loaded successfully")
except ImportError as e:
    logging.warning(f"Arduino communication not available: {e}")
    ARDUINO_AVAILABLE = False


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

class ObstacleAvoidanceState(Enum):
    NORMAL = "normal"
    OBSTACLE_DETECTED = "obstacle_detected"
    REVERSING = "reversing"
    SCANNING_LEFT = "scanning_left"
    SCANNING_RIGHT = "scanning_right"
    TURNING_TO_BEST = "turning_to_best"

class ObstacleAvoidanceController:
    """
    Advanced obstacle avoidance controller with scanning and best path selection.
    """
    def __init__(self, args):
        self.enabled = args.enable_obstacle_avoidance
        self.collision_timeout = args.obstacle_collision_timeout
        self.reverse_max_pwm = args.obstacle_reverse_max_pwm
        self.scan_angle = args.obstacle_scan_angle
        self.scan_speed = args.obstacle_scan_speed
        self.path_weight = args.obstacle_path_weight
        
        # State tracking
        self.state = ObstacleAvoidanceState.NORMAL
        self.collision_start_time = None
        self.last_collision_time = None
        self.reverse_start_time = None
        self.scan_start_time = None
        self.current_angle = 0.0
        self.scan_start_angle = 0.0
        self.best_angle = 0.0
        self.best_score = 0.0
        self.scan_data = {}  # angle -> (road_score, collision_score, combined_score)
        
        logging.info(f"Obstacle avoidance initialized: enabled={self.enabled}")
        if self.enabled:
            logging.info(f"  Collision timeout: {self.collision_timeout}s")
            logging.info(f"  Reverse max PWM: {self.reverse_max_pwm}")
            logging.info(f"  Scan angle: ±{self.scan_angle}°")
            logging.info(f"  Scan speed: {self.scan_speed}°/s")
            logging.info(f"  Path weight: {self.path_weight}")
    
    def update(self, detection_data, arduino, current_time):
        """
        Update obstacle avoidance state and return control commands.
        Returns: (should_override_normal_control, steering_angle, driving_pwm)
        """
        if not self.enabled:
            return False, None, None
            
        collision_rects = detection_data.get("collision_rectangles", [])
        road_rects = detection_data.get("road_rectangles", [])
        
        # Check if we have collision-free paths
        has_collision_free_path = len(collision_rects) > 0
        
        if has_collision_free_path:
            self.last_collision_time = None
            if self.state != ObstacleAvoidanceState.NORMAL:
                logging.info("Obstacle avoidance: Collision-free path found, returning to normal")
                self.state = ObstacleAvoidanceState.NORMAL
                self._reset_state()
            return False, None, None
        else:
            # No collision-free paths available
            if self.last_collision_time is None:
                self.last_collision_time = current_time
                self.collision_start_time = current_time
                logging.info("Obstacle avoidance: No collision-free paths detected")
            
            # Check if we've been without collision-free paths for too long
            collision_duration = current_time - self.collision_start_time
            if collision_duration >= self.collision_timeout:
                if self.state == ObstacleAvoidanceState.NORMAL:
                    logging.info("Obstacle avoidance: Activating obstacle avoidance sequence")
                    self.state = ObstacleAvoidanceState.REVERSING
                    self.reverse_start_time = current_time
                
                return self._handle_obstacle_avoidance(detection_data, arduino, current_time)
        
        return False, None, None
    
    def _handle_obstacle_avoidance(self, detection_data, arduino, current_time):
        """Handle the obstacle avoidance state machine."""
        
        if self.state == ObstacleAvoidanceState.REVERSING:
            return self._handle_reversing(current_time)
        
        elif self.state == ObstacleAvoidanceState.SCANNING_LEFT:
            return self._handle_scanning_left(detection_data, arduino, current_time)
        
        elif self.state == ObstacleAvoidanceState.SCANNING_RIGHT:
            return self._handle_scanning_right(detection_data, arduino, current_time)
        
        elif self.state == ObstacleAvoidanceState.TURNING_TO_BEST:
            return self._handle_turning_to_best(detection_data, arduino, current_time)
        
        return False, None, None
    
    def _handle_reversing(self, current_time):
        """Handle reversing state with gradually increasing power."""
        reverse_duration = current_time - self.reverse_start_time
        
        # Gradually increase reverse power up to max
        reverse_progress = min(1.0, reverse_duration / 2.0)  # 2 seconds to reach max
        reverse_pwm = int(-self.reverse_max_pwm * reverse_progress)  # Negative for reverse
        
        logging.info(f"Obstacle avoidance: Reversing at PWM {abs(reverse_pwm)} (progress: {reverse_progress:.1%})")
        
        # After 3 seconds of reversing, start scanning
        if reverse_duration >= 3.0:
            logging.info("Obstacle avoidance: Starting left scan")
            self.state = ObstacleAvoidanceState.SCANNING_LEFT
            self.scan_start_time = current_time
            self.scan_start_angle = self.current_angle
            self.scan_data = {}
            return True, self.current_angle, 0  # Stop driving
        
        return True, self.current_angle, reverse_pwm
    
    def _handle_scanning_left(self, detection_data, arduino, current_time):
        """Handle scanning left to find best path."""
        scan_duration = current_time - self.scan_start_time
        
        # Calculate target angle (scan left from starting position)
        angle_progress = scan_duration * self.scan_speed
        target_angle = self.scan_start_angle - min(angle_progress, self.scan_angle)
        
        # Record current position's scores
        self._record_scan_data(target_angle, detection_data)
        
        # Check if we've completed the left scan
        if angle_progress >= self.scan_angle:
            logging.info("Obstacle avoidance: Starting right scan")
            self.state = ObstacleAvoidanceState.SCANNING_RIGHT
            self.scan_start_time = current_time
            return True, target_angle, 0
        
        return True, target_angle, 0
    
    def _handle_scanning_right(self, detection_data, arduino, current_time):
        """Handle scanning right to find best path."""
        scan_duration = current_time - self.scan_start_time
        
        # Calculate target angle (scan right from starting position)
        angle_progress = scan_duration * self.scan_speed
        target_angle = self.scan_start_angle + min(angle_progress, self.scan_angle)
        
        # Record current position's scores
        self._record_scan_data(target_angle, detection_data)
        
        # Check if we've completed the right scan
        if angle_progress >= self.scan_angle:
            # Find best angle from scan data
            self._find_best_angle()
            logging.info(f"Obstacle avoidance: Scan complete, turning to best angle: {self.best_angle:.1f}° (score: {self.best_score:.3f})")
            self.state = ObstacleAvoidanceState.TURNING_TO_BEST
            self.scan_start_time = current_time
            return True, target_angle, 0
        
        return True, target_angle, 0
    
    def _handle_turning_to_best(self, detection_data, arduino, current_time):
        """Handle turning to the best found angle."""
        # Check if we have collision-free paths at current angle
        collision_rects = detection_data.get("collision_rectangles", [])
        if len(collision_rects) > 0:
            logging.info("Obstacle avoidance: Collision-free path found during turn, returning to normal")
            self.state = ObstacleAvoidanceState.NORMAL
            self._reset_state()
            return False, None, None
        
        # Continue turning towards best angle
        angle_diff = self.best_angle - self.current_angle
        if abs(angle_diff) > 2.0:  # 2 degree tolerance
            # Turn slowly towards best angle
            turn_speed = self.scan_speed * 0.5  # Half speed for final approach
            turn_direction = 1 if angle_diff > 0 else -1
            target_angle = self.current_angle + turn_direction * turn_speed * 0.1  # 0.1s time step
            
            return True, target_angle, 0
        else:
            # Reached target angle, wait for collision-free path
            logging.info("Obstacle avoidance: Reached best angle, waiting for collision-free path")
            return True, self.best_angle, 0
    
    def _record_scan_data(self, angle, detection_data):
        """Record scan data for current angle."""
        road_rects = detection_data.get("road_rectangles", [])
        collision_rects = detection_data.get("collision_rectangles", [])
        
        # Calculate road score
        road_score = max((r.get("score", 0.0) for r in road_rects), default=0.0)
        
        # Calculate collision score
        collision_score = max((c.get("score", 0.0) for c in collision_rects), default=0.0)
        
        # Combined score with weighting
        combined_score = road_score + self.path_weight * collision_score
        
        self.scan_data[angle] = (road_score, collision_score, combined_score)
        
        logging.info(f"Obstacle avoidance: Scan at {angle:.1f}° - road: {road_score:.3f}, collision: {collision_score:.3f}, combined: {combined_score:.3f}")
    
    def _find_best_angle(self):
        """Find the best angle from scan data."""
        if not self.scan_data:
            self.best_angle = self.scan_start_angle
            self.best_score = 0.0
            return
        
        # Find angle with highest combined score
        best_angle, (road_score, collision_score, combined_score) = max(
            self.scan_data.items(),
            key=lambda item: item[1][2]  # Sort by combined score
        )
        
        self.best_angle = best_angle
        self.best_score = combined_score
        
        logging.info(f"Obstacle avoidance: Best angle found: {best_angle:.1f}° with score {combined_score:.3f}")
    
    def _reset_state(self):
        """Reset state variables."""
        self.collision_start_time = None
        self.last_collision_time = None
        self.reverse_start_time = None
        self.scan_start_time = None
        self.scan_data = {}
        self.best_angle = 0.0
        self.best_score = 0.0
    
    def update_current_angle(self, angle):
        """Update the current steering angle."""
        self.current_angle = angle

def recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk: raise ConnectionError("Socket closed while receiving data")
        buf += chunk
    return buf

def open_cap(path: str) -> cv2.VideoCapture:
    # If path is int, treat as camera index
    if isinstance(path, int):
        cap = cv2.VideoCapture(path, cv2.CAP_V4L2)
    else:
        cap = cv2.VideoCapture(path, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video/camera: {path}")
    return cap

def make_writer(path: str, fps: float, w: int, h: int) -> cv2.VideoWriter:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    return cv2.VideoWriter(path, fourcc, fps, (w, h))

def encode_jpg(img: np.ndarray, quality: int) -> bytes:
    ok, buf = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok: raise RuntimeError("JPEG encode failed")
    return buf.tobytes()

def calculate_driving_speed(detection_data, args, obstacle_controller=None):
    """
    Calculate driving motor PWM based on road detection and collision avoidance.
    
    Args:
        detection_data: Dict containing road_rectangles and collision_rectangles
        args: Command line arguments with thresholds and PWM limits
        obstacle_controller: Optional obstacle avoidance controller
    
    Returns:
        int: PWM value for driving motor (0 = stop, negative = reverse)
    """
    road_rects = detection_data.get("road_rectangles", [])
    collision_rects = detection_data.get("collision_rectangles", [])
    
    # Safety first: Check for collision-free paths
    if not collision_rects:
        # Check if obstacle avoidance should handle this
        if obstacle_controller and obstacle_controller.enabled:
            current_time = time.time()
            should_override, _, driving_pwm = obstacle_controller.update(detection_data, None, current_time)
            if should_override and driving_pwm is not None:
                return driving_pwm
        
        logging.warning("No collision-free paths detected - stopping")
        return 0
    
    # Get the best collision-free path (first in sorted list)
    best_collision_path = collision_rects[0]
    collision_score = best_collision_path.get("score", 0.0)
    
    # Check collision threshold
    if collision_score < args.collision_threshold:
        logging.info(f"Collision score {collision_score:.2f} below threshold {args.collision_threshold:.2f} - stopping")
        return 0
    
    # Get road detection score
    road_score = 0.0
    if road_rects:
        selected_road = next((r for r in road_rects if r.get("is_selected")), None)
        if selected_road:
            road_score = selected_road.get("score", 0.0)
        else:
            # Use best road if none selected
            road_score = max(r.get("score", 0.0) for r in road_rects)
    
    # Check minimum road score
    if road_score < args.road_score_min:
        logging.info(f"Road score {road_score:.2f} below minimum {args.road_score_min:.2f} - stopping")
        return 0
    
    # Calculate PWM based on road score and collision safety
    # Combine road score and collision score for speed calculation
    combined_score = (road_score + collision_score) / 2.0
    
    # Map combined score to PWM range
    pwm_range = args.driving_pwm_max - args.driving_pwm_min
    pwm = args.driving_pwm_min + int(combined_score * pwm_range)
    
    # Clamp to valid range
    pwm = max(args.driving_pwm_min, min(args.driving_pwm_max, pwm))
    
    logging.info(f"Driving calculation: road={road_score:.2f}, collision={collision_score:.2f}, combined={combined_score:.2f}, PWM={pwm}")
    
    return pwm

# ----------------- async PAIR -----------------
def run_pair(args):
    # Initialize obstacle avoidance controller
    obstacle_controller = ObstacleAvoidanceController(args)
    
    # Initialize Arduino communication if requested
    arduino = None
    if args.arduino and ARDUINO_AVAILABLE:
        try:
            arduino = ArduinoClient(port=args.arduino_port)
            arduino.connect()
            logging.info(f"Arduino connected on port: {arduino.port}")
            # Test connection
            if arduino.ping():
                logging.info("Arduino ping successful")
                
                # Initialize driving motor to stopped state if driving is enabled
                if args.enable_driving:
                    success = arduino.set_motor2_pwm(0)
                    if success:
                        logging.info("Arduino: Initialized driving motor to stopped state")
                    else:
                        logging.warning("Arduino: Failed to initialize driving motor")
            else:
                logging.warning("Arduino ping failed")
        except Exception as e:
            logging.error(f"Failed to connect to Arduino: {e}")
            arduino = None
    elif args.arduino and not ARDUINO_AVAILABLE:
        logging.error("Arduino communication requested but arducom module not available")

    # Select camera or video for each input
    src1 = args.camera1 if hasattr(args, 'camera1') and args.camera1 is not None else args.video1
    src2 = args.camera2 if hasattr(args, 'camera2') and args.camera2 is not None else args.video2
    cap1 = open_cap(src1); cap2 = open_cap(src2)
    cap1.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap2.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    fps = cap1.get(cv2.CAP_PROP_FPS) or 25.0
    w1,h1 = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w2,h2 = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # Only write output if not using camera
    wri1 = make_writer(args.out1, fps, w1, h1) if (not hasattr(args, 'camera1') or args.camera1 is None) else None
    wri2 = make_writer(args.out2, fps, w2, h2) if (not hasattr(args, 'camera2') or args.camera2 is None) else None

    inflight_sem = threading.Semaphore(args.max_inflight)
    send_timestamps = {}  # frame_idx -> send_time
    sender_done = threading.Event()
    frame_sent = 0
    start = time.time()



    # Determine if we should show live display
    show_live = (hasattr(args, 'camera1') and args.camera1 is not None) or \
                (hasattr(args, 'camera2') and args.camera2 is not None) or args.show
    
    if show_live:
        logging.info("Live display enabled - BEV and Combined Camera windows will appear when processing starts")
        logging.info("Controls: Press 'q' in any window to quit")

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s, ThreadPoolExecutor(max_workers=2) as pool:
        s.connect((args.host, args.port))
        logging.info(f"Connected to server {args.host}:{args.port} (pair mode, max_inflight={args.max_inflight})")

        def sender():
            print("Sender started")
            nonlocal frame_sent
            try:
                while True:
                    ret1,f1 = cap1.read(); ret2,f2 = cap2.read()
                    if not ret1 or not ret2: break
                    inflight_sem.acquire()  # throttle
                    frame_sent += 1
                    idx = frame_sent
                    send_timestamps[idx] = time.time()

                    # Store original frames for parallel display
                    if hasattr(sender, 'latest_frames'):
                        sender.latest_frames = (f1.copy(), f2.copy())
                    else:
                        sender.latest_frames = (f1.copy(), f2.copy())

                    # parallel JPEG for small boost
                    fut1 = pool.submit(encode_jpg, f1, args.jpeg_quality)
                    fut2 = pool.submit(encode_jpg, f2, args.jpeg_quality)
                    b1, b2 = fut1.result(), fut2.result()

                    s.sendall(struct.pack('!I', idx))
                    s.sendall(struct.pack('!I', len(b1))); s.sendall(b1)
                    s.sendall(struct.pack('!I', len(b2))); s.sendall(b2)
            except Exception as e:
                logging.error(f"sender error: {e}")
            finally:
                sender_done.set()

        def receiver():
            frames_done = 0
            last_time = time.time() 
            state = None   
            windows_initialized = False
            
            if True:
                while True:
                    try:
                        logging.info("Waiting for response...")
                        resp_idx = struct.unpack('!I', recv_exact(s, 4))[0]
                        logging.info(f"Got response index: {resp_idx}")
                    except ConnectionError:
                        logging.error("Connection lost")
                        break
                    except Exception as e:
                        logging.error(f"Error receiving response: {e}")
                        break

                    # Log round-trip delay
                    if resp_idx in send_timestamps:
                        delay = time.time() - send_timestamps.pop(resp_idx)
                        logging.info(f"Frame {resp_idx} round-trip delay: {delay:.3f} sec")

                    # Read header + two payloads (pair framing)
                    sz1 = struct.unpack('!I', recv_exact(s, 4))[0]
                    rb1 = recv_exact(s, sz1)
                    sz2 = struct.unpack('!I', recv_exact(s, 4))[0]
                    rb2 = recv_exact(s, sz2)

                    # Handle different payload types
                    logging.info(f"Received payload sizes: {sz1}, {sz2} bytes, expecting {args.payload}")
                    
                    if args.payload == "ids":
                        # Decode numpy arrays and convert to colorized visualization
                        try:
                            import io
                            logging.info(f"Decoding IDs data, size: {len(rb1)} bytes")
                            ids = np.load(io.BytesIO(rb1), allow_pickle=False)
                            logging.info(f"IDs array shape: {ids.shape}, dtype: {ids.dtype}")
                            # Create a simple colormap for visualization
                            np.random.seed(42)  # Fixed seed for consistent colors
                            cm = np.random.randint(0, 255, (256, 3), dtype=np.uint8)
                            cm[0] = [0, 0, 0]  # Background black
                            img = cm[ids.astype(np.uint8)]
                            logging.info(f"Created visualization image shape: {img.shape}")
                        except Exception as e:
                            logging.error(f"Result decode failed (IDs): {e}")
                            break
                    else:
                        # For payload == "jpg" or "viz", decode as JPEG
                        img = cv2.imdecode(np.frombuffer(rb1, np.uint8), cv2.IMREAD_COLOR)
                        if img is None:
                            logging.error(f"Result decode failed (JPEG) - data size: {sz1} bytes")
                            # Try as numpy array fallback  
                            try:
                                import io
                                ids = np.load(io.BytesIO(rb1), allow_pickle=False)
                                cm = np.random.randint(0, 255, (256, 3), dtype=np.uint8)
                                img = cm[ids]
                                logging.warning("Data was actually IDs, not JPEG - server/client payload mismatch!")
                            except:
                                break
                        elif args.payload == "viz":
                            logging.info(f"Received BEV visualization frame {resp_idx}, size: {img.shape}")
                            
                            # The second payload contains detection data (JSON)
                            try:
                                import json
                                detection_data = json.loads(rb2.decode('utf-8'))
                                
                                # Log detection information
                                road_rects = detection_data.get("road_rectangles", [])
                                collision_rects = detection_data.get("collision_rectangles", [])
                                
                                logging.info(f"Road rectangles detected: {len(road_rects)}")
                                for rect in road_rects:
                                    status = "SELECTED" if rect.get("is_selected") else "detected"
                                    logging.info(f"  Road {status}: angle={rect['angle_deg']:.1f}°, score={rect['score']:.1f}")
                                
                                logging.info(f"Collision-free paths: {len(collision_rects)}")
                                for rect in collision_rects:
                                    logging.info(f"  Path: angle={rect['angle_deg']:.1f}°, score={rect['score']:.1f}")
                                
                                # Add detection info overlay on image
                                y_offset = 60
                                arduino_angle = None
                                
                                if road_rects:
                                    selected_road = next((r for r in road_rects if r.get("is_selected")), None)
                                    if selected_road:
                                        # Convert angle from BEV coordinate system (180° to 0°) to steering angle
                                        # BEV: 180° = full left, 90° = straight, 0° = full right
                                        # Arduino: -90° = full left, 0° = straight, +90° = full right
                                        bev_angle = selected_road['angle_deg']
                                        arduino_angle = -(bev_angle - 90.0)  # Convert to Arduino coordinate system
                                        
                                        text = f"Selected Road: {bev_angle:.1f}° → Arduino: {arduino_angle:.1f}° (score: {selected_road['score']:.1f})"
                                        cv2.putText(img, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                                        y_offset += 25
                                
                                if collision_rects:
                                    best_path = collision_rects[0]  # First one is typically the best
                                    text = f"Best Path: {best_path['angle_deg']:.1f}° (score: {best_path['score']:.1f})"
                                    cv2.putText(img, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                                
                                # Check for obstacle avoidance override first
                                obstacle_override = False
                                final_arduino_angle = arduino_angle
                                final_driving_pwm = None
                                
                                if obstacle_controller.enabled:
                                    current_time = time.time()
                                    should_override, override_angle, override_pwm = obstacle_controller.update(detection_data, arduino, current_time)
                                    
                                    if should_override:
                                        obstacle_override = True
                                        if override_angle is not None:
                                            final_arduino_angle = override_angle
                                        if override_pwm is not None:
                                            final_driving_pwm = override_pwm
                                        
                                        # Update obstacle controller with current angle
                                        if final_arduino_angle is not None:
                                            obstacle_controller.update_current_angle(final_arduino_angle)
                                        
                                        # Add obstacle avoidance status to display
                                        y_offset += 25
                                        state_text = f"Obstacle Avoidance: {obstacle_controller.state.value.upper()}"
                                        cv2.putText(img, state_text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)
                                
                                # Send steering angle to Arduino
                                if final_arduino_angle is not None and arduino is not None:
                                    try:
                                        success = arduino.set_angle(final_arduino_angle)
                                        if success:
                                            status = "OBSTACLE AVOIDANCE" if obstacle_override else "NORMAL"
                                            logging.info(f"Arduino: Set steering angle to {final_arduino_angle:.1f}° ({status})")
                                        else:
                                            logging.warning(f"Arduino: Failed to set angle {final_arduino_angle:.1f}°")
                                    except Exception as e:
                                        logging.error(f"Arduino communication error: {e}")
                                elif final_arduino_angle is not None:
                                    status = "OBSTACLE AVOIDANCE" if obstacle_override else "NORMAL"
                                    logging.info(f"Would set Arduino angle to {final_arduino_angle:.1f}° ({status}) (Arduino not connected)")
                                
                                # Calculate and send driving speed
                                if args.enable_driving and arduino is not None:
                                    try:
                                        if obstacle_override and final_driving_pwm is not None:
                                            driving_pwm = final_driving_pwm
                                        else:
                                            driving_pwm = calculate_driving_speed(detection_data, args, obstacle_controller)
                                        
                                        # Handle reverse PWM (negative values)
                                        if driving_pwm < 0:
                                            # For reverse, we might need to use a different Arduino command
                                            # For now, we'll use absolute value and log the reverse intent
                                            abs_pwm = abs(driving_pwm)
                                            success = arduino.set_motor2_pwm(abs_pwm)  # Use absolute value for now
                                            if success:
                                                logging.info(f"Arduino: Set REVERSE driving speed to PWM {abs_pwm} (original: {driving_pwm})")
                                            else:
                                                logging.warning(f"Arduino: Failed to set reverse driving speed {abs_pwm}")
                                        else:
                                            success = arduino.set_motor2_pwm(driving_pwm)
                                            if success:
                                                logging.info(f"Arduino: Set driving speed to PWM {driving_pwm}")
                                            else:
                                                logging.warning(f"Arduino: Failed to set driving speed {driving_pwm}")
                                        
                                        # Add driving info to display
                                        y_offset += 25
                                        if driving_pwm < 0:
                                            status_text = "REVERSING"
                                            color = (0, 255, 255)  # Yellow for reverse
                                        elif driving_pwm > 0:
                                            status_text = "DRIVING"
                                            color = (0, 255, 0)  # Green for forward
                                        else:
                                            status_text = "STOPPED"
                                            color = (0, 0, 255)  # Red for stopped
                                        
                                        mode = "OBSTACLE" if obstacle_override else "NORMAL"
                                        text = f"Driving: {status_text} (PWM: {abs(driving_pwm)}) [{mode}]"
                                        cv2.putText(img, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                                        
                                    except Exception as e:
                                        logging.error(f"Arduino driving control error: {e}")
                                elif args.enable_driving:
                                    if obstacle_override and final_driving_pwm is not None:
                                        driving_pwm = final_driving_pwm
                                    else:
                                        driving_pwm = calculate_driving_speed(detection_data, args, obstacle_controller)
                                    
                                    mode = "OBSTACLE" if obstacle_override else "NORMAL"
                                    logging.info(f"Would set Arduino driving speed to PWM {driving_pwm} [{mode}] (Arduino not connected)")
                                
                            except Exception as e:
                                logging.error(f"Failed to parse detection data: {e}")
                            
                            # Add title overlay for BEV frames
                            cv2.putText(img, "Bird's Eye View - Road Detection", (10, 30), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                    if show_live:
                        try:
                            # Initialize windows on first frame
                            if not windows_initialized:
                                cv2.namedWindow("Bird's Eye View - Road Detection", cv2.WINDOW_NORMAL)
                                cv2.namedWindow("Combined Camera View", cv2.WINDOW_NORMAL)
                                
                                # Position windows side by side
                                cv2.moveWindow("Bird's Eye View - Road Detection", 50, 50)
                                cv2.moveWindow("Combined Camera View", 650, 50)
                                
                                windows_initialized = True
                                logging.info("Display windows created: BEV and Combined Camera View")
                            
                            # Add frame counter to BEV display
                            bev_display = img.copy()
                            cv2.putText(bev_display, f"Frame: {resp_idx}", (img.shape[1] - 150, 30), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                            
                            # Display BEV visualization
                            cv2.imshow("Bird's Eye View - Road Detection", bev_display)
                            
                            # Display combined camera view if available
                            if hasattr(sender, 'latest_frames') and sender.latest_frames and len(sender.latest_frames) == 2:
                                f1_display, f2_display = sender.latest_frames
                                
                                # Create a combined view of both cameras
                                # Resize frames to fit side by side with better size
                                h1, w1 = f1_display.shape[:2]
                                h2, w2 = f2_display.shape[:2]
                                target_height = 360  # Larger size for better visibility
                                scale1 = target_height / h1
                                scale2 = target_height / h2
                                
                                f1_resized = cv2.resize(f1_display, (int(w1 * scale1), target_height))
                                f2_resized = cv2.resize(f2_display, (int(w2 * scale2), target_height))
                                
                                # Combine horizontally
                                combined = np.hstack([f1_resized, f2_resized])
                                
                                # Add comprehensive info overlay to combined view
                                cv2.putText(combined, f"Camera 1 & 2 - Frame {resp_idx}", (10, 30), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                                
                                # Add processing status
                                fps_text = f"Processing FPS: {frames_done / (time.time() - start):.1f}" if frames_done > 0 else "Processing..."
                                cv2.putText(combined, fps_text, (10, 60), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                                
                                # Add Arduino status
                                if arduino is not None:
                                    status_text = "Arduino: Connected"
                                    color = (0, 255, 0)
                                else:
                                    status_text = "Arduino: Disconnected"
                                    color = (0, 0, 255)
                                
                                cv2.putText(combined, status_text, (10, combined.shape[0] - 20), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                                
                                # Add camera labels
                                mid_point = f1_resized.shape[1]
                                cv2.putText(combined, "Camera 1", (20, combined.shape[0] - 50), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                                cv2.putText(combined, "Camera 2", (mid_point + 20, combined.shape[0] - 50), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                                
                                # Draw separator line
                                cv2.line(combined, (mid_point, 0), (mid_point, combined.shape[0]), (255, 255, 255), 2)
                                
                                cv2.imshow("Combined Camera View", combined)
                            
                            if cv2.waitKey(1) & 0xFF == ord('q'):
                                break
                        except cv2.error:
                            logging.warning("OpenCV built without GUI; disable --show or use xvfb-run.")

                    if wri1: wri1.write(img)
                    if wri2: wri2.write(img)  # both outputs get the same image

                    inflight_sem.release()
                    frames_done += 1
                    if frames_done % 20 == 0:
                        elapsed = time.time()
                        logging.info(f"RX wrote {frames_done} pairs, avg FPS: {20.0/(elapsed-last_time):.2f}")
                        last_time = time.time()
                        
                    # exit when sender finished AND all inflight drained

                    if sender_done.is_set() and inflight_sem._value == args.max_inflight:
                        break
            #except Exception as e:
            #    logging.error(f"receiver error: {e}")

        t_s = threading.Thread(target=sender, daemon=True)
        t_r = threading.Thread(target=receiver, daemon=True)
        t_s.start(); t_r.start()
        t_s.join(); t_r.join()

    cap1.release(); cap2.release()
    if wri1: wri1.release()
    if wri2: wri2.release()
    
    # Close Arduino connection
    if arduino is not None:
        try:
            # Stop driving motor before closing connection
            if args.enable_driving:
                arduino.set_motor2_pwm(0)
                logging.info("Arduino: Emergency stop - driving motor stopped")
            arduino.close()
            logging.info("Arduino connection closed")
        except Exception as e:
            logging.warning(f"Error closing Arduino connection: {e}")
    
    logging.info("Done pair.")
    # Always destroy windows if live display was used
    if show_live:
        cv2.destroyAllWindows()
        logging.info("Display windows closed")


# ----------------- async SINGLE -----------------
def run_single(args):
    
    # Select camera or video for input
    if hasattr(args, 'camera1') and args.single_source == 1 and args.camera1 is not None:
        src = args.camera1
        out = args.out1
        use_camera = True
    elif hasattr(args, 'camera2') and args.single_source == 2 and args.camera2 is not None:
        src = args.camera2
        out = args.out2
        use_camera = True
    else:
        src = args.video1 if args.single_source==1 else args.video2
        out = args.out1 if args.single_source==1 else args.out2
        use_camera = False
    cap = open_cap(src)
    fps = 10 #cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) 
    w,h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    wri = make_writer(out, fps, w, h) if not use_camera else None

    inflight_sem = threading.Semaphore(args.max_inflight)
    send_timestamps = {}  # frame_idx -> send_time
    sender_done = threading.Event()
    frame_sent = 0
    start = time.time()

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((args.host, args.port))
        logging.info(f"Connected to server {args.host}:{args.port} (single mode, max_inflight={args.max_inflight})")

        def sender():
            nonlocal frame_sent
            try:
                while True:
                    ret, f = cap.read()
                    if not ret: break
                    inflight_sem.acquire()
                    frame_sent += 1
                    idx = frame_sent
                    send_timestamps[idx] = time.time()
                    b = encode_jpg(f, args.jpeg_quality)
                    s.sendall(struct.pack('!I', idx))
                    s.sendall(struct.pack('!I', len(b))); s.sendall(b)
            finally:
                sender_done.set()

        def receiver():
            frames_done = 0
            try:
                while True:
                    try:
                        idx = struct.unpack('!I', recv_exact(s, 4))[0]
                    except ConnectionError:
                        break
                    # Log round-trip delay
                    if idx in send_timestamps:
                        delay = time.time() - send_timestamps.pop(idx)
                        logging.info(f"Frame {idx} round-trip delay: {delay:.3f} sec")
                    sz = struct.unpack('!I', recv_exact(s, 4))[0]
                    rb = recv_exact(s, sz)
                    img = cv2.imdecode(np.frombuffer(rb, np.uint8), cv2.IMREAD_COLOR)
                    if img is None: break
                    # Always show if using camera, or if --show is set
                    show_live = args.show
                    if show_live:
                        cv2.imshow("Processed Video", img)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break

                    if (img.shape[1],img.shape[0])!=(w,h): img=cv2.resize(img,(w,h))
                    if wri: wri.write(img)
                    inflight_sem.release()
                    frames_done += 1
                    if frames_done % 10 == 0:
                        elapsed = time.time()-start
                        logging.info(f"RX wrote {frames_done} frames, avg FPS: {frames_done/elapsed:.2f}")
                    if sender_done.is_set() and inflight_sem._value == args.max_inflight:
                        break
            except Exception as e:
                logging.error(f"receiver error: {e}")

        t_s = threading.Thread(target=sender, daemon=True)
        t_r = threading.Thread(target=receiver, daemon=True)
        t_s.start(); t_r.start()
        t_s.join(); t_r.join()


    cap.release()
    if wri: wri.release()
    logging.info("Done single.")
    if use_camera or args.show:
        cv2.destroyAllWindows()


def parse_args():
    ap = argparse.ArgumentParser("Async client")
    ap.add_argument("--mode", choices=["pair","single"], default="pair")
    ap.add_argument("--single-source", type=int, choices=[1,2], default=1)
    ap.add_argument("--host", default="127.0.0.1"); ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--video1", default="test_videos/test_video_5.mp4")
    ap.add_argument("--video2", default="test_videos/test_video_6.mp4")
    ap.add_argument("--camera1", type=int, default=None, help="Use camera index for input 1 (overrides --video1)")
    ap.add_argument("--camera2", type=int, default=None, help="Use camera index for input 2 (overrides --video2)")
    ap.add_argument("--out1",   default="test_results/segmented_result_5.avi")
    ap.add_argument("--out2",   default="test_results/segmented_result_6.avi")
    ap.add_argument("--jpeg-quality", type=int, default=40)
    ap.add_argument("--max-inflight", type=int, default=12)  
    ap.add_argument("--show", action="store_true",
                help="Display processed frames in a cv2 window")
    ap.add_argument("--frame-shape", nargs=2, type=int, default=[480, 640], metavar=("HEIGHT", "WIDTH"),
                help="Frame shape as HEIGHT WIDTH (default: 480 640)")
    ap.add_argument("--payload", choices=["jpg","ids","viz"], default="jpg", help="Payload type: jpg (default), ids (label map), or viz (BEV visualization)")
    ap.add_argument("--arduino", action="store_true", help="Enable Arduino communication to send steering angles")
    ap.add_argument("--arduino-port", default=None, help="Specific Arduino serial port (auto-detect if not specified)")
    
    # Driving motor control parameters
    ap.add_argument("--collision-threshold", type=float, default=0.3, help="Minimum collision score to enable driving (0.0-1.0)")
    ap.add_argument("--road-score-min", type=float, default=0.2, help="Minimum road score to move")
    ap.add_argument("--driving-pwm-min", type=int, default=30, help="Minimum PWM for driving motor")
    ap.add_argument("--driving-pwm-max", type=int, default=120, help="Maximum PWM for driving motor")
    ap.add_argument("--enable-driving", action="store_true", help="Enable automatic driving motor control")
    
    # Obstacle avoidance parameters
    ap.add_argument("--enable-obstacle-avoidance", action="store_true", 
                   help="Enable advanced obstacle avoidance with scanning and path selection")
    ap.add_argument("--obstacle-collision-timeout", type=float, default=2.0,
                   help="Time (seconds) without collision-free paths before activating obstacle avoidance (default: 2.0)")
    ap.add_argument("--obstacle-reverse-max-pwm", type=int, default=35,
                   help="Maximum PWM for reversing during obstacle avoidance (default: 35)")
    ap.add_argument("--obstacle-scan-angle", type=float, default=45.0,
                   help="Scan angle range in degrees (±degrees from center) (default: 45.0)")
    ap.add_argument("--obstacle-scan-speed", type=float, default=30.0,
                   help="Scanning speed in degrees per second (default: 30.0)")
    ap.add_argument("--obstacle-path-weight", type=float, default=0.7,
                   help="Weight coefficient for collision-free path score vs road score (default: 0.7)")
    
    return ap.parse_args()

def main():
    args = parse_args()
    if args.mode=="pair":
        run_pair(args)
    else:
        run_single(args)

if __name__ == "__main__":
    main()
