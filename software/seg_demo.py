import os, time, socket, struct, logging, argparse, multiprocessing as mp, threading, queue
import cv2, numpy as np
from openvino.runtime import Core  # keep as-is (deprecation warning ok)
from utils import (
    make_projector_and_grid,
    make_rotating_rect_tensor,
    process_two_bev_frames,
    npy_frame_to_bev,
    combine_viz_and_denoised,
)

import io
import datetime

# ----------------------------- Defaults -----------------------------
HOST = "127.0.0.1"
PORT = 5000
MODEL_PATH = "intel/semantic-segmentation-adas-0001/FP16-INT8/semantic-segmentation-adas-0001.xml"

INFERENCE_THREADS_PER_WORKER = 3   # overridable via CLI
JPEG_QUALITY = 40
IN_QUEUE_MAX = 1                  # per-worker queue
OUT_QUEUE_MAX = 4                # shared out queue (results)
PAIR_QUEUE_MAX = 4                # decoded frames waiting to be dispatched
MAX_INFLIGHT = 6                 # “window” of pairs/singles dispatched but not yet replied

LOG_EVERY_N_FRAMES = 20
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

# ------------------------------ Offline stream processor ------------------------------

def stream_video_ids(model_path: str, host: str, port: int, input_video: str, threads_per_worker: int):
    """
    Run segmentation on a video and stream each frame's class-ID map as .npy bytes
    to a single client over a socket using the same framing:
        idx | len | payload
    """
    try:
        cv2.setNumThreads(1)
    except Exception:
        pass
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    logging.info(f"[IDS-STREAM] Loading model: {model_path}")
    core = Core()
    model = core.read_model(model_path)
    compiled = core.compile_model(
        model, "CPU",
        {"PERFORMANCE_HINT":"LATENCY", "NUM_STREAMS":"1", "INFERENCE_NUM_THREADS": int(threads_per_worker)}
    )
    inp = compiled.input(0); outp = compiled.output(0)
    in_h, in_w = int(inp.shape[2]), int(inp.shape[3])

    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_video}")

    # Bind and wait for a single client
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((host, port))
        srv.listen(1)
        logging.info(f"[IDS-STREAM] Listening on {host}:{port}")
        conn, addr = srv.accept()
        logging.info(f"[IDS-STREAM] Client connected: {addr}")

        idx = 1
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                h, w = frame.shape[:2]
                t0 = time.perf_counter()
                if w != in_w or h != in_h:
                    frame_resized = cv2.resize(frame, (in_w, in_h), interpolation=cv2.INTER_LINEAR)
                else:
                    frame_resized = frame
                blob = frame_resized.transpose(2,0,1)[None].astype(np.uint8)
                result = compiled([blob])[outp]
                seg_map = result.squeeze().astype(np.uint8)
                seg_ids = cv2.resize(seg_map, (w, h), interpolation=cv2.INTER_NEAREST)

                buf = io.BytesIO()
                np.save(buf, seg_ids, allow_pickle=False)
                data = buf.getvalue()

                # framing: idx | len | payload
                conn.sendall(struct.pack("!I", idx))
                conn.sendall(struct.pack("!I", len(data)))
                conn.sendall(data)

                if idx % LOG_EVERY_N_FRAMES == 0:
                    logging.info(f"[IDS-STREAM {idx}] sent {len(data)} bytes")
                idx += 1
        finally:
            try: conn.close()
            except: pass
            cap.release()
            logging.info("[IDS-STREAM] Done.")


# ----------------------------- Utils -----------------------------
def recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("Socket closed while receiving data")
        buf += chunk
    return buf

# ------------------------- Worker process -------------------------
def worker_loop(model_path, in_q, out_q, worker_id, core_set, threads_per_worker, jpeg_quality, payload):
    try:
        cv2.setNumThreads(1)
    except Exception:
        pass
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    try:
        os.sched_setaffinity(0, core_set)
        logging.info(f"[W{worker_id}] pinned to cores: {sorted(core_set)}")
    except Exception:
        pass

    core = Core()
    model = core.read_model(model_path)
    compiled = core.compile_model(
        model, "CPU",
        {"PERFORMANCE_HINT":"LATENCY", "NUM_STREAMS":"1", "INFERENCE_NUM_THREADS": threads_per_worker}
    )
    inp = compiled.input(0); outp = compiled.output(0)
    in_h, in_w = int(inp.shape[2]), int(inp.shape[3])

    np.random.seed(42)
    color_map = np.random.randint(0,255,(256,3),dtype=np.uint8)

    while True:
        item = in_q.get()
        if item is None:
            break  # Stop signal
        frame_idx, frame = item

        t0 = time.perf_counter()
        if frame.shape[1] != in_w or frame.shape[0] != in_h:
            frame_resized = cv2.resize(frame, (in_w, in_h), interpolation=cv2.INTER_LINEAR)
        else:
            frame_resized = frame
        blob = frame_resized.transpose(2,0,1)[None].astype(np.uint8)
        t1 = time.perf_counter()

        result = compiled([blob])[outp]
        t2 = time.perf_counter()

        # IDs at network resolution
        seg_map = result.squeeze().astype(np.uint8)

        # Resize IDs back to original frame size
        seg_ids = cv2.resize(seg_map, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)

        # Build payload
        if payload == "jpg":
            seg_overlay = color_map[seg_ids]    # (H,W,3) uint8
            alpha = globals().get('BLEND_ALPHA', 0.5)
            blended = cv2.addWeighted(frame, 0, seg_overlay, 1, 0)
            t3 = time.perf_counter()
            ok, jpg = cv2.imencode(".jpg", blended, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
            data_bytes = jpg.tobytes() if ok else b""
            t4 = time.perf_counter()
            pre, infer, post, enc = (t1-t0, t2-t1, t3-t2, t4-t3)
        else:
            # payload in {"ids","viz"}: send .npy bytes
            t3 = time.perf_counter()
            buf = io.BytesIO()
            np.save(buf, seg_ids, allow_pickle=False)
            data_bytes = buf.getvalue()
            t4 = time.perf_counter()
            pre, infer, post, enc = (t1-t0, t2-t1, t3-t2, t4-t3)

        if frame_idx % LOG_EVERY_N_FRAMES == 0:
            logging.info(f"[W{worker_id} {frame_idx}] pre {pre:.3f}s | infer {infer:.3f}s | post {post:.3f}s | enc {enc:.3f}s")

        out_q.put((frame_idx, worker_id, data_bytes, pre, infer, post, enc))
class SegmentationServer:
    def __init__(self, model_path, host, port, mode="pair", max_inflight=MAX_INFLIGHT,
                 pair_queue_max=PAIR_QUEUE_MAX, threads_per_worker=INFERENCE_THREADS_PER_WORKER,
                 jpeg_quality=JPEG_QUALITY, payload="jpg",
                 flower_road_length=130, flower_road_thickness=5,
                 flower_collision_length=60, flower_collision_thickness=5,
                 log_videos=False):
        self.host, self.port, self.mode = host, port, mode
        self.max_inflight = int(max_inflight)
        self.pair_queue_max = int(pair_queue_max)
        self.threads_per_worker = int(threads_per_worker)
        self.jpeg_quality = int(jpeg_quality)
        self.payload = payload
        self.log_videos = log_videos
        self.data_dir = os.path.join(os.getcwd(), "collected_data")
        if self.log_videos:
            os.makedirs(self.data_dir, exist_ok=True)
        self.log_video_duration = 30  # seconds
        logging.info(f"Server mode: {self.mode} | max_inflight={self.max_inflight} | payload={self.payload}")

        # writers and sizes (initialized to None)
        self.left_orig_writer = self.right_orig_writer = None
        self.left_proc_writer = self.right_proc_writer = None
        self.post_writer = None
        self.left_orig_writer_size = self.right_orig_writer_size = None
        self.left_proc_writer_size = self.right_proc_writer_size = None
        self.post_writer_size = None

        # workers: 2 in pair mode, 1 in single mode
        self.num_workers = 2 if self.mode=="pair" else 1
        ncores = os.cpu_count() or 8
        if self.num_workers == 1:
            core_sets = [set(range(ncores))]
        else:
            h = max(1, ncores//2)
            core_sets = [set(range(0,h)), set(range(h,ncores))]

        # process-safe queues to workers
        self.in_queues = [mp.Queue(maxsize=IN_QUEUE_MAX) for _ in range(self.num_workers)]
        self.out_queue = mp.Queue(maxsize=OUT_QUEUE_MAX)
        self.procs = [
            mp.Process(target=worker_loop,
                       args=(model_path, self.in_queues[i], self.out_queue, i, core_sets[i],
                             self.threads_per_worker, self.jpeg_quality, self.payload),
                       daemon=True)
            for i in range(self.num_workers)
        ]
        for p in self.procs:
            p.start()
        logging.info(f"Started {self.num_workers} worker process(es)")

        # ---- Post-processing assets (used when payload == 'viz') ----
        self.projector_left, self.grid_left = make_projector_and_grid(
            frame_shape=(480, 640),   # <- set to your model/frame H,W (or pass via CLI if needed)
            fx=300.0, fy=int(530/380*200),
            height_m=1.0, pitch_deg_down=0.0,
            meters_per_pixel=0.20,
            forward_range=(0.5, 30.0),
            lateral_range=(-20.0, 20.0),
        )
        self.projector_right, self.grid_right = make_projector_and_grid(
            frame_shape=(480, 640),
            fx=380.0, fy=530.0,
            height_m=1.0, pitch_deg_down=0.0,
            meters_per_pixel=0.20,
            forward_range=(0.5, 30.0),
            lateral_range=(-20.0, 20.0),
        )
        self.flower_tensor_road = make_rotating_rect_tensor(
            H=236, W=330, n=100, length=flower_road_length, thickness=flower_road_thickness)
        self.flower_tensor_collision = make_rotating_rect_tensor(
            H=236, W=330, n=100, length=flower_collision_length, thickness=flower_collision_thickness)
        # streaming state (one per connection ideally; we’ll keep a simple one here)
        self.proc_state = None

    def stop(self):
        for q in self.in_queues:
            q.put(None)
        for p in self.procs:
            p.join(timeout=2.0)
        for p in self.procs:
            if p.is_alive(): p.terminate()

    # ----------------- top-level accept loop -----------------
    def serve(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind((self.host, self.port))
            server.listen(1)
            logging.info(f"Socket server listening on {self.host}:{self.port} ({self.mode} mode, payload={self.payload})")
            while True:
                conn, addr = server.accept()
                logging.info(f"Connected by {addr}")
                try:
                    if self.mode=="pair":
                        self._handle_pair_async(conn)
                    else:
                        self._handle_single_async(conn)
                except ConnectionError:
                    logging.info("Client disconnected")
                except Exception as e:
                    logging.exception(f"Error handling client: {e}")
                finally:
                    try: conn.close()
                    except: pass
                    logging.info(f"Connection closed for {addr}")

    # ----------------- ASYNC pair handler -----------------
    def _handle_pair_async(self, conn: socket.socket):
        stop_evt = threading.Event()
        inflight_sem = threading.Semaphore(self.max_inflight)
        inflight_count = 0
        inflight_lock = threading.Lock()
        dispatch_time = {}

        pending_q: "queue.Queue[tuple[int,np.ndarray,np.ndarray]]" = queue.Queue(maxsize=self.pair_queue_max)

        # --- Logging setup ---
        # Remove all early writer creation; writers are now created lazily after decoding the first frame
        if self.log_videos:
            self._log_start_time = time.time()
        else:
            self.left_orig_writer = self.right_orig_writer = None
            self.left_proc_writer = self.right_proc_writer = None
            self.post_writer = None
            self.left_orig_writer_size = self.right_orig_writer_size = None
            self.left_proc_writer_size = self.right_proc_writer_size = None
            self.post_writer_size = None
            self._log_start_time = None

        def _rotate_writers():
            for attr in [
                "left_orig_writer", "right_orig_writer",
                "left_proc_writer", "right_proc_writer",
                "post_writer"
            ]:
                writer = getattr(self, attr, None)
                if writer:
                    writer.release()
                    setattr(self, attr, None)
            time.sleep(0.1)  # brief pause to ensure file closure

            if self.left_orig_writer: self.left_orig_writer.release()
            if self.right_orig_writer: self.right_orig_writer.release()
            if self.left_proc_writer: self.left_proc_writer.release()
            if self.right_proc_writer: self.right_proc_writer.release()
            if self.post_writer: self.post_writer.release()
            self.left_orig_writer, _ = self._get_video_writer('left_original')
            self.right_orig_writer, _ = self._get_video_writer('right_original')
            self.left_proc_writer, _ = self._get_video_writer('left_processed')
            self.right_proc_writer, _ = self._get_video_writer('right_processed')
            self.post_writer, _ = self._get_video_writer('postprocessed')
            self._log_start_time = time.time()

        def recv_thread():
            try:
                while True:
                    if self.log_videos and (self._log_start_time is not None) and (time.time() - self._log_start_time > self.log_video_duration):
                        for a in ["left_orig_writer","right_orig_writer","left_proc_writer","right_proc_writer","post_writer"]:
                            wr = getattr(self, a, None)
                            if wr is not None:
                                try: wr.release()
                                except: pass
                                setattr(self, a, None)
                                setattr(self, a+"_size", None)
                        self._log_start_time = time.time()
                    idx = struct.unpack("!I", recv_exact(conn, 4))[0]
                    s1 = struct.unpack("!I", recv_exact(conn, 4))[0]
                    b1 = recv_exact(conn, s1)
                    s2 = struct.unpack("!I", recv_exact(conn, 4))[0]
                    b2 = recv_exact(conn, s2)
                    f1 = cv2.imdecode(np.frombuffer(b1, np.uint8), cv2.IMREAD_COLOR)
                    f2 = cv2.imdecode(np.frombuffer(b2, np.uint8), cv2.IMREAD_COLOR)
                    if f1 is None or f2 is None:
                        logging.warning("Decode failed for one of the frames")
                        break
                    if self.log_videos:
                        if self._log_start_time is None:
                            self._log_start_time = time.time()
                        lw = self._ensure_writer("left_orig_writer",  "left_original",  f1)
                        rw = self._ensure_writer("right_orig_writer", "right_original", f2)
                        if lw and lw.isOpened(): lw.write(f1)
                        if rw and rw.isOpened(): rw.write(f2)
                    pending_q.put((idx, f1, f2))
            except ConnectionError:
                pass
            finally:
                stop_evt.set()

        def dispatch_thread():
            nonlocal inflight_count
            while not stop_evt.is_set() or not pending_q.empty():
                try:
                    idx, f1, f2 = pending_q.get(timeout=0.05)
                except queue.Empty:
                    continue
                inflight_sem.acquire()
                with inflight_lock:
                    inflight_count += 1
                # Send to both workers
                self.in_queues[0].put((idx, f1))
                self.in_queues[1].put((idx, f2))
                dispatch_time[idx] = time.perf_counter()

        def send_thread():
            nonlocal inflight_count
            partial = {}   # idx -> [b0, b1], arrival times
            arrival = {}   # idx -> {0:t,1:t}
            timings = {}   # idx -> {wid:(pre,infer,post,enc)}
            while not stop_evt.is_set() or inflight_count>0:
                try:
                    item = self.out_queue.get(timeout=0.05)
                except queue.Empty:
                    continue
                if len(item)==7:
                    idx, wid, data, pre, inf, post, enc = item
                    timings.setdefault(idx,{})[wid] = (pre,inf,post,enc)
                else:
                    idx, wid, data = item
                partial.setdefault(idx, [None,None])[wid] = data
                arrival.setdefault(idx, {})[wid] = time.perf_counter()

                pair = partial.get(idx)
                if pair and pair[0] is not None and pair[1] is not None:
                    try:
                        if self.payload == "viz":
                            # Decode .npy -> ids arrays
                            ids1 = np.load(io.BytesIO(pair[0]), allow_pickle=False)
                            ids2 = np.load(io.BytesIO(pair[1]), allow_pickle=False)

                            # Deterministic color map for processed videos
                            if not hasattr(self, "cm"):
                                rng = np.random.default_rng(42)
                                self.cm = rng.integers(0, 255, size=(256, 3), dtype=np.uint8)

                            # Save left/right processed videos (colorized class maps)
                            if self.log_videos:
                                arr1 = self.cm[ids1]
                                arr2 = self.cm[ids2]
                                lw = self._ensure_writer("left_proc_writer",  "left_processed",  arr1)
                                rw = self._ensure_writer("right_proc_writer", "right_processed", arr2)
                                if lw and lw.isOpened(): lw.write(arr1)
                                if rw and rw.isOpened(): rw.write(arr2)

                            # BEV per side
                            bev_left  = npy_frame_to_bev(ids1, projector=self.projector_left,  grid=self.grid_left,  road_label=0)
                            bev_right = npy_frame_to_bev(ids2, projector=self.projector_right, grid=self.grid_right, road_label=0)

                            # Run post-processing
                            viz_img, denoised, self.proc_state, result = process_two_bev_frames(
                                bev_left, bev_right,
                                self.flower_tensor_road, self.flower_tensor_collision,
                                self.proc_state,
                                combine_angle_deg=-35.0,
                                combine_hshift_px=40,
                                centre_bias_coefficient=0.6,
                                sigma_scores=4.0,
                                remembrance=8.0,
                                residual_sigma=16.0,
                                convergent_with_road=5.0,
                                road_min_abs=40,
                                road_min_frac_of_max=0.10,
                                collision_min_abs=300,
                                collision_min_frac_of_max=0.30,
                                small_top_k=1,
                                road_color=(100,255,100),
                                selected_road_color=(0,255,255),
                                small_color=(0,0,255),
                                morph_kernel=4,
                                blur_ksize=11,
                                blur_sigma=6.0
                            )
                            final_img = combine_viz_and_denoised(viz_img, denoised)
                            
                            # Extract detection data
                            n_road = self.flower_tensor_road.shape[2]
                            n_collision = self.flower_tensor_collision.shape[2]
                            
                            # Calculate angles for each rectangle index
                            def calc_angle(i, n):
                                return 180.0 - i * (180.0 / (n - 1)) if n > 1 else 180.0
                            
                            # Prepare detection data
                            detection_data = {
                                "road_rectangles": [],
                                "collision_rectangles": []
                            }
                            
                            # Road rectangles (large rectangles)
                            road_scores = result.get("road_scores", np.array([]))
                            best_road_idx = result.get("best_road_idx", -1)
                            kept_indices = result.get("kept_indices", np.array([]))
                            
                            for i in kept_indices:
                                angle = calc_angle(int(i), n_road)
                                score = float(road_scores[i]) if i < len(road_scores) else 0.0
                                is_selected = bool(i == best_road_idx)  # Convert to native bool
                                detection_data["road_rectangles"].append({
                                    "index": int(i),
                                    "angle_deg": float(angle),
                                    "score": float(score),
                                    "is_selected": is_selected
                                })
                            
                            # Collision rectangles (small rectangles)
                            small_scores = result.get("small_scores_biased", np.array([]))
                            best_small_idxs = result.get("best_small_idxs", np.array([]))
                            
                            for i in best_small_idxs:
                                angle = calc_angle(int(i), n_collision)
                                score = float(small_scores[i]) if i < len(small_scores) else 0.0
                                detection_data["collision_rectangles"].append({
                                    "index": int(i),
                                    "angle_deg": float(angle),
                                    "score": float(score),
                                    "is_best": True
                                })
                            
                            # Serialize detection data as JSON
                            import json
                            detection_json = json.dumps(detection_data).encode('utf-8')
                            
                            if self.log_videos:
                                pw = self._ensure_writer("post_writer", "postprocessed", final_img)
                                if pw and pw.isOpened():
                                    pw.write(final_img)
                            
                            # Encode BEV image
                            ok, jpg = cv2.imencode(".jpg", final_img, [int(cv2.IMWRITE_JPEG_QUALITY), int(self.jpeg_quality)])
                            image_bytes = jpg.tobytes() if ok else b""
                            
                            # Send: idx | image_size | image_data | json_size | json_data | json_size | json_data (duplicate for pair protocol)
                            conn.sendall(struct.pack("!I", idx))
                            conn.sendall(struct.pack("!I", len(image_bytes))); conn.sendall(image_bytes)
                            conn.sendall(struct.pack("!I", len(detection_json))); conn.sendall(detection_json)
                        else:
                            # Existing behavior (jpg overlays or raw ids)
                            conn.sendall(struct.pack("!I", idx))
                            conn.sendall(struct.pack("!I", len(pair[0]))); conn.sendall(pair[0])
                            conn.sendall(struct.pack("!I", len(pair[1]))); conn.sendall(pair[1])
                            # Logging processed images
                            if self.log_videos and self.payload in ("ids", "jpg"):
                                if self.payload == "jpg":
                                    arr1 = cv2.imdecode(np.frombuffer(pair[0], np.uint8), cv2.IMREAD_COLOR)
                                    arr2 = cv2.imdecode(np.frombuffer(pair[1], np.uint8), cv2.IMREAD_COLOR)
                                else:
                                    cm = np.random.randint(0,255,(256,3),dtype=np.uint8)
                                    ids1 = np.load(io.BytesIO(pair[0]), allow_pickle=False)
                                    ids2 = np.load(io.BytesIO(pair[1]), allow_pickle=False)
                                    arr1 = cm[ids1]
                                    arr2 = cm[ids2]
                                lw = self._ensure_writer("left_proc_writer",  "left_processed",  arr1)
                                rw = self._ensure_writer("right_proc_writer", "right_processed", arr2)
                                if lw and lw.isOpened(): lw.write(arr1)
                                if rw and rw.isOpened(): rw.write(arr2)
                    except (BrokenPipeError, ConnectionResetError):
                        logging.warning("Client disconnected during send (pair mode)")
                        break

                    # timings
                    t_disp = dispatch_time.pop(idx, None)
                    if t_disp is not None:
                        t_after_wait = max(arrival[idx].values())
                        d_wait = t_after_wait - t_disp
                        w0 = arrival[idx].get(0, t_after_wait)-t_disp
                        w1 = arrival[idx].get(1, t_after_wait)-t_disp
                        (w0p,w0i,w0o,w0e) = timings.get(idx,{}).get(0,(0,0,0,0))
                        (w1p,w1i,w1o,w1e) = timings.get(idx,{}).get(1,(0,0,0,0))
                        logging.info(
                          f"pair {idx}: wait {d_wait:.3f}s [w0 {w0:.3f}, w1 {w1:.3f}] | "
                          f"w0 [pre {w0p:.3f}|infer {w0i:.3f}|post {w0o:.3f}|enc {w0e:.3f}] | "
                          f"w1 [pre {w1p:.3f}|infer {w1i:.3f}|post {w1o:.3f}|enc {w1e:.3f}]"
                        )

                    inflight_sem.release()
                    with inflight_lock:
                        inflight_count -= 1
                    partial.pop(idx, None)
                    arrival.pop(idx, None)
                    timings.pop(idx, None)

        t_recv = threading.Thread(target=recv_thread, daemon=True)
        t_disp = threading.Thread(target=dispatch_thread, daemon=True)
        t_send = threading.Thread(target=send_thread, daemon=True)
        t_recv.start(); t_disp.start(); t_send.start()
        t_recv.join(); t_disp.join(); t_send.join()
        if self.left_orig_writer: self.left_orig_writer.release()
        if self.right_orig_writer: self.right_orig_writer.release()
        if self.left_proc_writer: self.left_proc_writer.release()
        if self.right_proc_writer: self.right_proc_writer.release()
        if self.post_writer: self.post_writer.release()

    # ----------------- ASYNC single handler -----------------
    def _handle_single_async(self, conn: socket.socket):
        stop_evt = threading.Event()
        inflight_sem = threading.Semaphore(self.max_inflight)
        inflight_count = 0
        inflight_lock = threading.Lock()
        dispatch_time = {}
        pending_q: "queue.Queue[tuple[int,np.ndarray]]" = queue.Queue(maxsize=self.pair_queue_max)
        if self.log_videos:
            orig_writer = proc_writer = post_writer = None
        else:
            orig_writer = proc_writer = post_writer = None
        def dispatch_thread():
            nonlocal inflight_count
            while not stop_evt.is_set() or not pending_q.empty():
                try:
                    idx, f = pending_q.get(timeout=0.05)
                except queue.Empty:
                    continue
                inflight_sem.acquire()
                with inflight_lock:
                    inflight_count += 1
                self.in_queues[0].put((idx, f))
                dispatch_time[idx] = time.perf_counter()
        def recv_thread():
            try:
                while True:
                    idx = struct.unpack("!I", recv_exact(conn, 4))[0]
                    s1 = struct.unpack("!I", recv_exact(conn, 4))[0]
                    b1 = recv_exact(conn, s1)
                    f = cv2.imdecode(np.frombuffer(b1, np.uint8), cv2.IMREAD_COLOR)
                    if f is None:
                        logging.warning("Decode failed for frame")
                        break
                    if self.log_videos:
                        orig_writer = self._ensure_writer("orig_writer", "original_single", f)
                        if orig_writer and orig_writer.isOpened():
                            orig_writer.write(f)
                    pending_q.put((idx, f))
            except ConnectionError:
                pass
            finally:
                stop_evt.set()
        def send_thread():
            nonlocal inflight_count
            while not stop_evt.is_set() or inflight_count>0:
                try:
                    item = self.out_queue.get(timeout=0.05)
                except queue.Empty:
                    continue
                if len(item)==7:
                    idx, wid, data, pre, inf, post, enc = item
                else:
                    idx, wid, data = item
                try:
                    # send immediately
                    conn.sendall(struct.pack("!I", idx))
                    conn.sendall(struct.pack("!I", len(data))); conn.sendall(data)
                    if self.log_videos:
                        if self.payload == "viz":
                            arr = np.frombuffer(data, np.uint8)
                            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                            post_writer = self._ensure_writer("post_writer", "postprocessed_single", img)
                            if post_writer and post_writer.isOpened() and img is not None:
                                post_writer.write(img)
                        elif self.payload == "jpg":
                            arr = np.frombuffer(data, np.uint8)
                            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                            proc_writer = self._ensure_writer("proc_writer", "processed_single", img)
                            if proc_writer and proc_writer.isOpened() and img is not None:
                                proc_writer.write(img)
                        elif self.payload == "ids":
                            arr = np.load(io.BytesIO(data), allow_pickle=False)
                            color_map = np.random.randint(0,255,(256,3),dtype=np.uint8)
                            img = color_map[arr]
                            proc_writer = self._ensure_writer("proc_writer", "processed_single", img)
                            if proc_writer and proc_writer.isOpened():
                                proc_writer.write(img)
                except (BrokenPipeError, ConnectionResetError):
                    logging.warning("Client disconnected during send (single mode)")
                    break

                t_disp = dispatch_time.pop(idx, None)
                if t_disp is not None and len(item)==7:
                    d_wait = time.perf_counter() - t_disp
                    logging.info(f"single {idx}: wait {d_wait:.3f}s | worker [pre {pre:.3f}|infer {inf:.3f}|post {post:.3f}|enc {enc:.3f}]")

                inflight_sem.release()
                with inflight_lock:
                    inflight_count -= 1

        t_recv = threading.Thread(target=recv_thread, daemon=True)
        t_disp = threading.Thread(target=dispatch_thread, daemon=True)
        t_send = threading.Thread(target=send_thread, daemon=True)
        t_recv.start(); t_disp.start(); t_send.start()
        t_recv.join(); t_disp.join(); t_send.join()
        if orig_writer: orig_writer.release()
        if proc_writer: proc_writer.release()
        if post_writer: post_writer.release()

    def _get_video_writer(self, name, width, height):
        now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{name}_{now}.avi"
        path = os.path.join(self.data_dir, filename)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        wr = cv2.VideoWriter(path, fourcc, 20.0, (width, height))
        if not wr.isOpened():
            logging.warning(f"Failed to open VideoWriter for {filename} ({width}x{height})")
        return wr, path

    def _ensure_writer(self, attr_base: str, name: str, frame: np.ndarray):
        """Create or recreate a writer to match the frame size."""
        h, w = frame.shape[:2]
        size_attr = attr_base + "_size"
        writer_attr = attr_base
        cur_size = getattr(self, size_attr, None)
        writer = getattr(self, writer_attr, None)
        if writer is None or (cur_size is not None and cur_size != (w, h)) or not writer.isOpened():
            if writer is not None:
                try: writer.release()
                except: pass
            writer, _ = self._get_video_writer(name, width=w, height=h)
            setattr(self, writer_attr, writer)
            setattr(self, size_attr, (w, h))
        return writer

# ------------------------------ Main ------------------------------
def parse_args():
    ap = argparse.ArgumentParser("Segmentation server (async)")
    ap.add_argument("--mode", choices=["pair","single"], default="pair")
    ap.add_argument("--host", default=HOST) 
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--model", default=MODEL_PATH)
    ap.add_argument("--threads-per-worker", type=int, default=INFERENCE_THREADS_PER_WORKER)
    ap.add_argument("--jpeg-quality", type=int, default=JPEG_QUALITY)
    ap.add_argument("--max-inflight", type=int, default=MAX_INFLIGHT)
    ap.add_argument("--pair-queue-max", type=int, default=PAIR_QUEUE_MAX)
    ap.add_argument("--blend-alpha", type=float, default=0.5, help="Transparency for overlay (0.0-1.0, default=0.5)")
    # Offline mode args
    ap.add_argument("--process-ids", action="store_true",
                    help="Run offline: read a video, produce class-ID masks, save as a single .npy array (no socket).")
    ap.add_argument("--input-video", default=None,
                    help="Path to input video file (required with --process-ids).")
    ap.add_argument("--output-npy", default=None,
                    help="Path to output .npy file (required with --process-ids).")
    ap.add_argument("--payload", choices=["jpg","ids","viz"], default="jpg",
                    help="What to send to the client. 'jpg' overlay, 'ids' label maps, or 'viz' final combined visualization.")
    ap.add_argument("--process-ids-stream", action="store_true",
                    help="With --process-ids, stream per-frame ID maps to one socket client instead of saving a single .npy.")
    ap.add_argument("--flower-road-length", type=int, default=130, help="Length of detected road required to be considered valid street")
    ap.add_argument("--flower-road-thickness", type=int, default=5, help="Thickness of detected road required to be considered valid street")
    ap.add_argument("--flower-collision-length", type=int, default=60, help="Length of free space required to be considered valid for movement")
    ap.add_argument("--flower-collision-thickness", type=int, default=5, help="Thickness of free space required to be considered valid for movement")
    ap.add_argument("--log-videos", action="store_true",
                    help="If set, saves original, processed (class), and postprocessed videos returned from client.")
    return ap.parse_args()
# ------------------------------ Offline Processor ------------------------------
def process_video_ids(model_path: str, input_video: str, output_npy: str, threads_per_worker: int):
    """
    Offline pipeline:
      - Load model on CPU
      - Read frames from input_video
      - For each frame, run segmentation -> class IDs (uint8)
      - Resize IDs back to original frame size
      - Stack into (T, H, W) and save to output_npy
    """
    if not os.path.isfile(input_video):
        raise FileNotFoundError(f"Input video not found: {input_video}")

    # Light thread/env pinning (same spirit as workers)
    try:
        cv2.setNumThreads(1)
    except Exception:
        pass
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    logging.info(f"[IDS] Loading model: {model_path}")
    core = Core()
    model = core.read_model(model_path)
    compiled = core.compile_model(
        model, "CPU",
        {"PERFORMANCE_HINT":"LATENCY", "NUM_STREAMS":"1", "INFERENCE_NUM_THREADS": int(threads_per_worker)}
    )
    inp = compiled.input(0); outp = compiled.output(0)
    in_h, in_w = int(inp.shape[2]), int(inp.shape[3])

    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_video}")

    ids_frames = []
    frame_idx = 0
    t_global0 = time.perf_counter()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        h, w = frame.shape[:2]

        t0 = time.perf_counter()
        if w != in_w or h != in_h:
            frame_resized = cv2.resize(frame, (in_w, in_h), interpolation=cv2.INTER_LINEAR)
        else:
            frame_resized = frame

        # NCHW uint8 (model expects 8-bit)
        blob = frame_resized.transpose(2, 0, 1)[None].astype(np.uint8)
        t1 = time.perf_counter()

        result = compiled([blob])[outp]  # (1, H, W) or (H, W)
        seg_map = result.squeeze().astype(np.uint8)  # (in_h, in_w) class IDs
        t2 = time.perf_counter()

        # Resize IDs back to original frame size (NEAREST to preserve labels)
        seg_ids = cv2.resize(seg_map, (w, h), interpolation=cv2.INTER_NEAREST)
        ids_frames.append(seg_ids)

        if frame_idx % LOG_EVERY_N_FRAMES == 0:
            logging.info(f"[IDS {frame_idx}] pre {(t1-t0):.3f}s | infer {(t2-t1):.3f}s")

        frame_idx += 1

    cap.release()
    if not ids_frames:
        raise RuntimeError("No frames were read from the video.")

    # Stack to (T, H, W) uint8 and save
    arr = np.stack(ids_frames, axis=0).astype(np.uint8)
    np.save(output_npy, arr, allow_pickle=False)
    t_global1 = time.perf_counter()

    logging.info(f"[IDS] Saved {arr.shape} uint8 to: {output_npy} | total time {(t_global1 - t_global0):.2f}s")

def main():
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    args = parse_args()
    # Offline mode: detach from client, process video -> npy
    if args.process_ids:
        if not args.input_video or not args.output_npy:
            raise SystemExit("--process-ids requires --input-video and --output-npy")
        logging.info("Running in process-ids (offline) mode: no socket server will be started.")
        process_video_ids(
            model_path=args.model,
            input_video=args.input_video,
            output_npy=args.output_npy,
            threads_per_worker=args.threads_per_worker
        )
        return

    # Online mode: original socket server
    global BLEND_ALPHA
    BLEND_ALPHA = args.blend_alpha
    srv = SegmentationServer(
        args.model, args.host, args.port, mode=args.mode,
        max_inflight=args.max_inflight, pair_queue_max=args.pair_queue_max,
        threads_per_worker=args.threads_per_worker, jpeg_quality=args.jpeg_quality,
        payload=args.payload,
        flower_road_length=args.flower_road_length,
        flower_road_thickness=args.flower_road_thickness,
        flower_collision_length=args.flower_collision_length,
        flower_collision_thickness=args.flower_collision_thickness,
        log_videos=args.log_videos
    )
    try:
        srv.serve()
    finally:
        srv.stop()

if __name__ == "__main__":
    main()
