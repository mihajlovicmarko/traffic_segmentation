import os
import time
import socket
import struct
import logging
import argparse
import multiprocessing as mp

import cv2
import numpy as np
from openvino.runtime import Core

# ----------------------------- Defaults -----------------------------
HOST = "127.0.0.1"
PORT = 5000
MODEL_PATH = "intel/semantic-segmentation-adas-0001/FP16-INT8/semantic-segmentation-adas-0001.xml"

# Worker performance knobs (overridable via CLI)
INFERENCE_THREADS_PER_WORKER = 3     # try 3–4 depending on CPU
JPEG_QUALITY = 75                    # 70–80 is a good speed/quality trade-off
IN_QUEUE_MAX = 2                     # small buffers to avoid latency build-up
OUT_QUEUE_MAX = 4

LOG_EVERY_N_FRAMES = 20

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")


# ----------------------------- Utils -----------------------------
def recv_exact(sock: socket.socket, n: int) -> bytes:
    """Receive exactly n bytes or raise ConnectionError."""
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("Socket closed while receiving data")
        buf += chunk
    return buf


# ------------------------- Worker process -------------------------
def worker_loop(model_path, in_q, out_q, worker_id, core_set):
    """
    Each worker lives in its own process with its own OpenVINO Core/model.
    Receives (frame_idx:int, frame:np.ndarray BGR) and returns (frame_idx, worker_id, jpg_bytes:bytes, pre, infer, post, enc).
    """
    # Keep library thread counts small inside the worker
    try:
        cv2.setNumThreads(1)
    except Exception:
        pass
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    # Pin the worker to the provided set of cores (best-effort)
    try:
        os.sched_setaffinity(0, core_set)
        logging.info(f"[W{worker_id}] pinned to cores: {sorted(core_set)}")
    except Exception:
        pass

    core = Core()
    model = core.read_model(model_path)
    compiled = core.compile_model(
        model,
        "CPU",
        {
            "PERFORMANCE_HINT": "LATENCY",
            "NUM_STREAMS": "1",                             # one stream per worker
            "INFERENCE_NUM_THREADS": INFERENCE_THREADS_PER_WORKER,
        },
    )
    inp = compiled.input(0)
    outp = compiled.output(0)
    in_h, in_w = int(inp.shape[2]), int(inp.shape[3])

    # Static color map for label → color
    np.random.seed(42)
    color_map = np.random.randint(0, 255, size=(256, 3), dtype=np.uint8)

    while True:
        item = in_q.get()
        if item is None:  # shutdown
            break

        frame_idx, frame = item

        # --- Preprocess (resize to model input, NHWC → NCHW uint8) ---
        t0 = time.perf_counter()
        if frame.shape[1] != in_w or frame.shape[0] != in_h:
            frame_resized = cv2.resize(frame, (in_w, in_h), interpolation=cv2.INTER_LINEAR)
        else:
            frame_resized = frame
        blob = frame_resized.transpose(2, 0, 1)[None].astype(np.uint8)
        t1 = time.perf_counter()

        # --- Inference ---
        result = compiled([blob])[outp]
        t2 = time.perf_counter()

        # --- Postprocess (colorize, resize back, blend) ---
        seg_map = result.squeeze().astype(np.uint8)  # (H,W)
        seg_overlay = color_map[seg_map]             # (H,W,3)
        seg_overlay = cv2.resize(
            seg_overlay, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST
        )
        blended = cv2.addWeighted(frame, 0.5, seg_overlay, 0.5, 0)
        t3 = time.perf_counter()

        # --- Encode JPEG ---
        ok, jpg = cv2.imencode(".jpg", blended, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
        jpg_bytes = jpg.tobytes() if ok else b""
        t4 = time.perf_counter()

        pre = t1 - t0
        infer = t2 - t1
        post = t3 - t2
        enc = t4 - t3

        if frame_idx % LOG_EVERY_N_FRAMES == 0:
            logging.info(
                f"[W{worker_id} {frame_idx}] pre {pre:.3f}s | "
                f"infer {infer:.3f}s | post {post:.3f}s | enc {enc:.3f}s"
            )

        out_q.put((frame_idx, worker_id, jpg_bytes, pre, infer, post, enc))


# --------------------------- Server (parent) ---------------------------
class SegmentationServer:
    def __init__(self, model_path: str, host: str, port: int, mode: str = "pair") -> None:
        self.host = host
        self.port = port
        self.mode = mode  # "pair" or "single"
        logging.info(f"Server mode: {self.mode}")

        # Workers: 2 in pair mode, 1 in single mode
        num_workers = 2 if self.mode == "pair" else 1

        def _core_sets(num_workers):
            n = os.cpu_count() or 8
            if num_workers == 1:
                return [set(range(n))]                      # use ALL cores in single mode
            h = n // 2
            return [set(range(0, h)), set(range(h, n))]     # split in pair mode

        core_sets = _core_sets(num_workers)

        # One input queue per worker, one shared output queue
        self.in_queues = [mp.Queue(maxsize=IN_QUEUE_MAX) for _ in range(num_workers)]
        self.out_queue = mp.Queue(maxsize=OUT_QUEUE_MAX)

        # Worker processes
        self.procs = [
            mp.Process(target=worker_loop,
                       args=(model_path, self.in_queues[i], self.out_queue, i, core_sets[i]),
                       daemon=True)
            for i in range(num_workers)
        ]
        for p in self.procs:
            p.start()
        logging.info(f"Started {num_workers} worker process(es)")

    def stop(self) -> None:
        # Graceful shutdown for workers
        for q in self.in_queues:
            q.put(None)
        for p in self.procs:
            p.join(timeout=2.0)
        for p in self.procs:
            if p.is_alive():
                p.terminate()

    def serve(self) -> None:
        if self.mode == "pair":
            self.serve_pairs()
        else:
            self.serve_single()

    def serve_pairs(self) -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind((self.host, self.port))
            server.listen(1)
            logging.info(f"Socket server listening on {self.host}:{self.port} (pair mode)")

            while True:
                conn, addr = server.accept()
                logging.info(f"Connected by {addr}")
                try:
                    self.handle_client_pair(conn)
                except ConnectionError:
                    logging.info("Client disconnected")
                except Exception as e:
                    logging.exception(f"Error handling client: {e}")
                finally:
                    conn.close()
                    logging.info(f"Connection closed for {addr}")

    def serve_single(self) -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind((self.host, self.port))
            server.listen(1)
            logging.info(f"Socket server listening on {self.host}:{self.port} (single mode)")

            while True:
                conn, addr = server.accept()
                logging.info(f"Connected by {addr}")
                try:
                    self.handle_client_single(conn)
                except ConnectionError:
                    logging.info("Client disconnected")
                except Exception as e:
                    logging.exception(f"Error handling client: {e}")
                finally:
                    conn.close()
                    logging.info(f"Connection closed for {addr}")

    # ---------- handlers ----------
    def handle_client_pair(self, conn: socket.socket) -> None:
        while True:
            t0 = time.perf_counter()

            # receive paired request
            idx_bytes = recv_exact(conn, 4)
            frame_idx = struct.unpack("!I", idx_bytes)[0]

            img1_size = struct.unpack("!I", recv_exact(conn, 4))[0]
            img1_bytes = recv_exact(conn, img1_size)

            img2_size = struct.unpack("!I", recv_exact(conn, 4))[0]
            img2_bytes = recv_exact(conn, img2_size)

            t_after_recv = time.perf_counter()

            # decode
            frame1 = cv2.imdecode(np.frombuffer(img1_bytes, np.uint8), cv2.IMREAD_COLOR)
            frame2 = cv2.imdecode(np.frombuffer(img2_bytes, np.uint8), cv2.IMREAD_COLOR)
            if frame1 is None or frame2 is None:
                logging.warning("Decode failed for one of the frames")
                break

            t_after_decode = time.perf_counter()

            # dispatch to workers 0 and 1
            self.in_queues[0].put((frame_idx, frame1))
            self.in_queues[1].put((frame_idx, frame2))

            t_after_dispatch = time.perf_counter()

            # gather both results for this frame_idx
            res = {0: None, 1: None}
            arrive = {}
            timings = {}
            while res[0] is None or res[1] is None:
                item = self.out_queue.get()
                if len(item) == 7:
                    idx, worker_id, jpg_bytes, pre, infer, post, enc = item
                else:
                    idx, worker_id, jpg_bytes = item; pre=infer=post=enc=0.0
                now = time.perf_counter()
                if idx == frame_idx and worker_id in (0, 1) and res[worker_id] is None:
                    res[worker_id] = jpg_bytes
                    arrive[worker_id] = now
                    timings[worker_id] = (pre, infer, post, enc)

            t_after_wait = max(arrive.values())

            res1_bytes = res[0]
            res2_bytes = res[1]
            if not res1_bytes or not res2_bytes:
                logging.error("One of the workers failed to encode result")
                break

            # send both results together
            conn.sendall(struct.pack("!I", frame_idx))
            conn.sendall(struct.pack("!I", len(res1_bytes))); conn.sendall(res1_bytes)
            conn.sendall(struct.pack("!I", len(res2_bytes))); conn.sendall(res2_bytes)

            t_after_send = time.perf_counter()

            # timings
            d_recv     = t_after_recv     - t0
            d_decode   = t_after_decode   - t_after_recv
            d_dispatch = t_after_dispatch - t_after_decode
            d_wait     = t_after_wait     - t_after_dispatch
            d_send     = t_after_send     - t_after_wait
            d_total    = t_after_send     - t0
            w0_wait    = arrive.get(0, t_after_wait) - t_after_dispatch
            w1_wait    = arrive.get(1, t_after_wait) - t_after_dispatch
            w0_pre, w0_infer, w0_post, w0_enc = timings.get(0, (0.0, 0.0, 0.0, 0.0))
            w1_pre, w1_infer, w1_post, w1_enc = timings.get(1, (0.0, 0.0, 0.0, 0.0))

            logging.info(
                f"pair {frame_idx}: total {d_total:.3f}s | recv {d_recv:.3f} | "
                f"decode {d_decode:.3f} | dispatch {d_dispatch:.3f} | "
                f"wait {d_wait:.3f} [w0 {w0_wait:.3f}, w1 {w1_wait:.3f}] | "
                f"worker0 [pre {w0_pre:.3f} | infer {w0_infer:.3f} | post {w0_post:.3f} | enc {w0_enc:.3f}] | "
                f"worker1 [pre {w1_pre:.3f} | infer {w1_infer:.3f} | post {w1_post:.3f} | enc {w1_enc:.3f}] | "
                f"send {d_send:.3f}"
            )

    def handle_client_single(self, conn: socket.socket) -> None:
        """Single-image protocol: idx, size, bytes -> idx, size, bytes."""
        while True:
            t0 = time.perf_counter()

            # receive single request
            idx_bytes = recv_exact(conn, 4)
            frame_idx = struct.unpack("!I", idx_bytes)[0]

            img_size = struct.unpack("!I", recv_exact(conn, 4))[0]
            img_bytes = recv_exact(conn, img_size)

            t_after_recv = time.perf_counter()

            # decode
            frame = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                logging.warning("Decode failed for the frame")
                break

            t_after_decode = time.perf_counter()

            # dispatch to worker 0
            self.in_queues[0].put((frame_idx, frame))

            t_after_dispatch = time.perf_counter()

            # gather result for this frame_idx
            res_bytes = None
            pre = infer = post = enc = 0.0
            while res_bytes is None:
                item = self.out_queue.get()
                if len(item) == 7:
                    idx, worker_id, jpg_bytes, pre, infer, post, enc = item
                else:
                    idx, worker_id, jpg_bytes = item; pre=infer=post=enc=0.0
                if idx == frame_idx:
                    res_bytes = jpg_bytes

            t_after_wait = time.perf_counter()

            # send result
            conn.sendall(struct.pack("!I", frame_idx))
            conn.sendall(struct.pack("!I", len(res_bytes))); conn.sendall(res_bytes)

            t_after_send = time.perf_counter()

            # timings
            d_recv     = t_after_recv     - t0
            d_decode   = t_after_decode   - t_after_recv
            d_dispatch = t_after_dispatch - t_after_decode
            d_wait     = t_after_wait     - t_after_dispatch
            d_send     = t_after_send     - t_after_wait
            d_total    = t_after_send     - t0

            logging.info(
                f"single {frame_idx}: total {d_total:.3f}s | recv {d_recv:.3f} | decode {d_decode:.3f} | "
                f"dispatch {d_dispatch:.3f} | wait {d_wait:.3f} | "
                f"worker [pre {pre:.3f} | infer {infer:.3f} | post {post:.3f} | enc {enc:.3f}] | "
                f"send {d_send:.3f}"
            )


# ------------------------------ Main ------------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="Segmentation server (single or paired frames).")
    ap.add_argument("--mode", choices=["pair", "single"], default="pair",
                    help="Protocol mode: 'pair' expects two images per request; 'single' expects one.")
    ap.add_argument("--host", default=HOST)
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--model", default=MODEL_PATH)
    ap.add_argument("--threads-per-worker", type=int, default=INFERENCE_THREADS_PER_WORKER)
    ap.add_argument("--jpeg-quality", type=int, default=JPEG_QUALITY)
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    # propagate CLI overrides to module-level knobs used in workers
    global INFERENCE_THREADS_PER_WORKER, JPEG_QUALITY
    INFERENCE_THREADS_PER_WORKER = args.threads_per_worker
    JPEG_QUALITY = args.jpeg_quality

    # 'spawn' is safest across Docker/WSL/Windows
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    server = SegmentationServer(args.model, args.host, args.port, mode=args.mode)
    try:
        server.serve()
    finally:
        server.stop()


if __name__ == "__main__":
    main()
