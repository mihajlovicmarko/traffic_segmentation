# client_async.py
import os, time, socket, struct, logging, argparse
import cv2, numpy as np
from concurrent.futures import ThreadPoolExecutor
import threading

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

def recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk: raise ConnectionError("Socket closed while receiving data")
        buf += chunk
    return buf

def open_cap(path: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(path, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")
    return cap

def make_writer(path: str, fps: float, w: int, h: int) -> cv2.VideoWriter:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    return cv2.VideoWriter(path, fourcc, fps, (w, h))

def encode_jpg(img: np.ndarray, quality: int) -> bytes:
    ok, buf = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok: raise RuntimeError("JPEG encode failed")
    return buf.tobytes()

# ----------------- async PAIR -----------------
def run_pair(args):
    cap1 = open_cap(args.video1); cap2 = open_cap(args.video2)
    fps = cap1.get(cv2.CAP_PROP_FPS) or 25.0
    w1,h1 = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w2,h2 = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))
    wri1 = make_writer(args.out1, fps, w1, h1); wri2 = make_writer(args.out2, fps, w2, h2)

    inflight_sem = threading.Semaphore(args.max_inflight)
    sender_done = threading.Event()
    frame_sent = 0
    start = time.time()

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s, ThreadPoolExecutor(max_workers=2) as pool:
        s.connect((args.host, args.port))
        logging.info(f"Connected to server {args.host}:{args.port} (pair mode, max_inflight={args.max_inflight})")

        def sender():
            nonlocal frame_sent
            try:
                while True:
                    ret1,f1 = cap1.read(); ret2,f2 = cap2.read()
                    if not ret1 or not ret2: break
                    inflight_sem.acquire()  # throttle
                    frame_sent += 1
                    idx = frame_sent

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
            try:
                while True:
                    try:
                        resp_idx = struct.unpack('!I', recv_exact(s, 4))[0]
                    except ConnectionError:
                        break

                    sz1 = struct.unpack('!I', recv_exact(s, 4))[0]
                    rb1 = recv_exact(s, sz1)
                    sz2 = struct.unpack('!I', recv_exact(s, 4))[0]
                    rb2 = recv_exact(s, sz2)

                    img1 = cv2.imdecode(np.frombuffer(rb1, np.uint8), cv2.IMREAD_COLOR)
                    img2 = cv2.imdecode(np.frombuffer(rb2, np.uint8), cv2.IMREAD_COLOR)
                    if img1 is None or img2 is None:
                        logging.error("Result decode failed"); break
                    if (img1.shape[1],img1.shape[0])!=(w1,h1): img1=cv2.resize(img1,(w1,h1))
                    if (img2.shape[1],img2.shape[0])!=(w2,h2): img2=cv2.resize(img2,(w2,h2))
                    wri1.write(img1); wri2.write(img2)

                    inflight_sem.release()
                    frames_done += 1
                    if frames_done % 20 == 0:
                        elapsed = time.time()
                        logging.info(f"RX wrote {frames_done} pairs, avg FPS: {20.0/(elapsed-last_time):.2f}")
                        last_time = time.time()
                        
                    # exit when sender finished AND all inflight drained

                    if sender_done.is_set() and inflight_sem._value == args.max_inflight:
                        break
            except Exception as e:
                logging.error(f"receiver error: {e}")

        t_s = threading.Thread(target=sender, daemon=True)
        t_r = threading.Thread(target=receiver, daemon=True)
        t_s.start(); t_r.start()
        t_s.join(); t_r.join()

    cap1.release(); cap2.release(); wri1.release(); wri2.release()
    logging.info("Done pair.")

# ----------------- async SINGLE -----------------
def run_single(args):
    vid = args.video1 if args.single_source==1 else args.video2
    out = args.out1 if args.single_source==1 else args.out2
    cap = open_cap(vid)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w,h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    wri = make_writer(out, fps, w, h)

    inflight_sem = threading.Semaphore(args.max_inflight)
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
                    sz = struct.unpack('!I', recv_exact(s, 4))[0]
                    rb = recv_exact(s, sz)
                    img = cv2.imdecode(np.frombuffer(rb, np.uint8), cv2.IMREAD_COLOR)
                    if img is None: break
                    if (img.shape[1],img.shape[0])!=(w,h): img=cv2.resize(img,(w,h))
                    wri.write(img)
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

    cap.release(); wri.release()
    logging.info("Done single.")

def parse_args():
    ap = argparse.ArgumentParser("Async client")
    ap.add_argument("--mode", choices=["pair","single"], default="pair")
    ap.add_argument("--single-source", type=int, choices=[1,2], default=1)
    ap.add_argument("--host", default="127.0.0.1"); ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--video1", default="test_videos/test_video_5.mp4")
    ap.add_argument("--video2", default="test_videos/test_video_6.mp4")
    ap.add_argument("--out1",   default="test_results/segmented_result_5.avi")
    ap.add_argument("--out2",   default="test_results/segmented_result_6.avi")
    ap.add_argument("--jpeg-quality", type=int, default=75)
    ap.add_argument("--max-inflight", type=int, default=12)  # window size
    return ap.parse_args()

def main():
    args = parse_args()
    if args.mode=="pair":
        run_pair(args)
    else:
        run_single(args)

if __name__ == "__main__":
    main()
