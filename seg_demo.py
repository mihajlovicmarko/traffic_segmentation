# server_async.py
import os, time, socket, struct, logging, argparse, multiprocessing as mp, threading, queue
import cv2, numpy as np
from openvino.runtime import Core  # keep as-is (deprecation warning ok)

# ----------------------------- Defaults -----------------------------
HOST = "127.0.0.1"
PORT = 5000
MODEL_PATH = "intel/semantic-segmentation-adas-0001/FP16-INT8/semantic-segmentation-adas-0001.xml"

INFERENCE_THREADS_PER_WORKER = 3   # overridable via CLI
JPEG_QUALITY = 75
IN_QUEUE_MAX = 4                   # per-worker queue
OUT_QUEUE_MAX = 16                 # shared out queue (results)
PAIR_QUEUE_MAX = 16                # decoded frames waiting to be dispatched
MAX_INFLIGHT = 12                  # “window” of pairs/singles dispatched but not yet replied

LOG_EVERY_N_FRAMES = 20
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

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
def worker_loop(model_path, in_q, out_q, worker_id, core_set, threads_per_worker, jpeg_quality):
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
            break
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

        seg_map = result.squeeze().astype(np.uint8)
        seg_overlay = color_map[seg_map]
        seg_overlay = cv2.resize(seg_overlay, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
        blended = cv2.addWeighted(frame, 0.5, seg_overlay, 0.5, 0)
        t3 = time.perf_counter()

        ok, jpg = cv2.imencode(".jpg", blended, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
        jpg_bytes = jpg.tobytes() if ok else b""
        t4 = time.perf_counter()

        pre, infer, post, enc = (t1-t0, t2-t1, t3-t2, t4-t3)
        if frame_idx % LOG_EVERY_N_FRAMES == 0:
            logging.info(f"[W{worker_id} {frame_idx}] pre {pre:.3f}s | infer {infer:.3f}s | post {post:.3f}s | enc {enc:.3f}s")

        out_q.put((frame_idx, worker_id, jpg_bytes, pre, infer, post, enc))

# --------------------------- Server ---------------------------
class SegmentationServer:
    def __init__(self, model_path, host, port, mode="pair", max_inflight=MAX_INFLIGHT,
                 pair_queue_max=PAIR_QUEUE_MAX, threads_per_worker=INFERENCE_THREADS_PER_WORKER,
                 jpeg_quality=JPEG_QUALITY):
        self.host, self.port, self.mode = host, port, mode
        self.max_inflight = int(max_inflight)
        self.pair_queue_max = int(pair_queue_max)
        self.threads_per_worker = int(threads_per_worker)
        self.jpeg_quality = int(jpeg_quality)
        logging.info(f"Server mode: {self.mode} | max_inflight={self.max_inflight}")

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
                             self.threads_per_worker, self.jpeg_quality),
                       daemon=True)
            for i in range(self.num_workers)
        ]
        for p in self.procs:
            p.start()
        logging.info(f"Started {self.num_workers} worker process(es)")

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
            logging.info(f"Socket server listening on {self.host}:{self.port} ({self.mode} mode)")
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
        dispatch_time = {}   # idx -> t_dispatch

        # queue of decoded pairs waiting to be dispatched to workers
        pending_q: "queue.Queue[tuple[int,np.ndarray,np.ndarray]]" = queue.Queue(maxsize=self.pair_queue_max)

        def recv_thread():
            try:
                while True:
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
                    # send as soon as both ready
                    conn.sendall(struct.pack("!I", idx))
                    conn.sendall(struct.pack("!I", len(pair[0]))); conn.sendall(pair[0])
                    conn.sendall(struct.pack("!I", len(pair[1]))); conn.sendall(pair[1])

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

    # ----------------- ASYNC single handler -----------------
    def _handle_single_async(self, conn: socket.socket):
        stop_evt = threading.Event()
        inflight_sem = threading.Semaphore(self.max_inflight)
        inflight_count = 0
        inflight_lock = threading.Lock()
        dispatch_time = {}

        pending_q: "queue.Queue[tuple[int,np.ndarray]]" = queue.Queue(maxsize=self.pair_queue_max)

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
                    pending_q.put((idx, f))
            except ConnectionError:
                pass
            finally:
                stop_evt.set()

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
                # send immediately
                conn.sendall(struct.pack("!I", idx))
                conn.sendall(struct.pack("!I", len(data))); conn.sendall(data)

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

# ------------------------------ Main ------------------------------
def parse_args():
    ap = argparse.ArgumentParser("Segmentation server (async)")
    ap.add_argument("--mode", choices=["pair","single"], default="pair")
    ap.add_argument("--host", default=HOST); ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--model", default=MODEL_PATH)
    ap.add_argument("--threads-per-worker", type=int, default=INFERENCE_THREADS_PER_WORKER)
    ap.add_argument("--jpeg-quality", type=int, default=JPEG_QUALITY)
    ap.add_argument("--max-inflight", type=int, default=MAX_INFLIGHT)
    ap.add_argument("--pair-queue-max", type=int, default=PAIR_QUEUE_MAX)
    return ap.parse_args()

def main():
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    args = parse_args()
    srv = SegmentationServer(args.model, args.host, args.port, mode=args.mode,
                             max_inflight=args.max_inflight, pair_queue_max=args.pair_queue_max,
                             threads_per_worker=args.threads_per_worker, jpeg_quality=args.jpeg_quality)
    try:
        srv.serve()
    finally:
        srv.stop()

if __name__ == "__main__":
    main()
