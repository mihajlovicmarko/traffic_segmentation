# client_async.py
import os, time, socket, struct, logging, argparse
import cv2, numpy as np
from concurrent.futures import ThreadPoolExecutor
import threading
import sys
import os
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

def calculate_driving_speed(detection_data, args):
    """
    Calculate driving motor PWM based on road detection and collision avoidance.
    
    Args:
        detection_data: Dict containing road_rectangles and collision_rectangles
        args: Command line arguments with thresholds and PWM limits
    
    Returns:
        int: PWM value for driving motor (0 = stop)
    """
    road_rects = detection_data.get("road_rectangles", [])
    collision_rects = detection_data.get("collision_rectangles", [])
    
    # Safety first: Check for collision-free paths
    if not collision_rects:
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
                                        arduino_angle = bev_angle - 90.0  # Convert to Arduino coordinate system
                                        
                                        text = f"Selected Road: {bev_angle:.1f}° → Arduino: {arduino_angle:.1f}° (score: {selected_road['score']:.1f})"
                                        cv2.putText(img, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                                        y_offset += 25
                                
                                if collision_rects:
                                    best_path = collision_rects[0]  # First one is typically the best
                                    text = f"Best Path: {best_path['angle_deg']:.1f}° (score: {best_path['score']:.1f})"
                                    cv2.putText(img, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                                
                                # Send steering angle to Arduino
                                if arduino_angle is not None and arduino is not None:
                                    try:
                                        success = arduino.set_angle(arduino_angle)
                                        if success:
                                            logging.info(f"Arduino: Set steering angle to {arduino_angle:.1f}°")
                                        else:
                                            logging.warning(f"Arduino: Failed to set angle {arduino_angle:.1f}°")
                                    except Exception as e:
                                        logging.error(f"Arduino communication error: {e}")
                                elif arduino_angle is not None:
                                    logging.info(f"Would set Arduino angle to {arduino_angle:.1f}° (Arduino not connected)")
                                
                                # Calculate and send driving speed
                                if args.enable_driving and arduino is not None:
                                    try:
                                        driving_pwm = calculate_driving_speed(detection_data, args)
                                        success = arduino.set_motor2_pwm(driving_pwm)
                                        if success:
                                            logging.info(f"Arduino: Set driving speed to PWM {driving_pwm}")
                                        else:
                                            logging.warning(f"Arduino: Failed to set driving speed {driving_pwm}")
                                        
                                        # Add driving info to display
                                        y_offset += 25
                                        status_text = "DRIVING" if driving_pwm > 0 else "STOPPED"
                                        color = (0, 255, 0) if driving_pwm > 0 else (0, 0, 255)
                                        text = f"Driving: {status_text} (PWM: {driving_pwm})"
                                        cv2.putText(img, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                                        
                                    except Exception as e:
                                        logging.error(f"Arduino driving control error: {e}")
                                elif args.enable_driving:
                                    driving_pwm = calculate_driving_speed(detection_data, args)
                                    logging.info(f"Would set Arduino driving speed to PWM {driving_pwm} (Arduino not connected)")
                                
                            except Exception as e:
                                logging.error(f"Failed to parse detection data: {e}")
                            
                            # Add title overlay for BEV frames
                            cv2.putText(img, "Bird's Eye View - Road Detection", (10, 30), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                    show_live = (hasattr(args, 'camera1') and args.camera1 is not None) or \
                                (hasattr(args, 'camera2') and args.camera2 is not None) or args.show

                    if show_live:
                        try:
                            cv2.imshow("Road detection and following", img)
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
    if (hasattr(args, 'camera1') and args.camera1 is not None) or (hasattr(args, 'camera2') and args.camera2 is not None) or args.show:
        cv2.destroyAllWindows()


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
                    show_live = use_camera or args.show
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
    return ap.parse_args()

def main():
    args = parse_args()
    if args.mode=="pair":
        run_pair(args)
    else:
        run_single(args)

if __name__ == "__main__":
    main()
