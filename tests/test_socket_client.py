import os
import time
import socket
import struct
import logging
import argparse
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')


def recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = b''
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("Socket closed while receiving data")
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
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return buf.tobytes()


def run_pair(args):
    cap1 = open_cap(args.video1)
    cap2 = open_cap(args.video2)

    fps = cap1.get(cv2.CAP_PROP_FPS) or 25.0  # pace from stream 1
    w1, h1 = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w2, h2 = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer1 = make_writer(args.out1, fps, w1, h1)
    writer2 = make_writer(args.out2, fps, w2, h2)

    frame_idx = 0
    start_time = time.time()

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((args.host, args.port))
        logging.info(f"Connected to server at {args.host}:{args.port} (pair mode)")

        pool = ThreadPoolExecutor(max_workers=2)

        while True:
            ret1, frame1 = cap1.read()
            ret2, frame2 = cap2.read()
            if not ret1 or not ret2:
                break

            frame_idx += 1

            # Encode both frames in parallel for a little speedup
            f1 = pool.submit(encode_jpg, frame1, args.jpeg_quality)
            f2 = pool.submit(encode_jpg, frame2, args.jpeg_quality)
            img1_bytes, img2_bytes = f1.result(), f2.result()

            # ---- send pair ----
            s.sendall(struct.pack('!I', frame_idx))
            s.sendall(struct.pack('!I', len(img1_bytes))); s.sendall(img1_bytes)
            s.sendall(struct.pack('!I', len(img2_bytes))); s.sendall(img2_bytes)

            # ---- receive paired results ----
            resp_idx = struct.unpack('!I', recv_exact(s, 4))[0]
            if resp_idx != frame_idx:
                logging.warning(f"Frame index mismatch: sent {frame_idx}, got {resp_idx}")

            res1_size = struct.unpack('!I', recv_exact(s, 4))[0]
            res1_bytes = recv_exact(s, res1_size)

            res2_size = struct.unpack('!I', recv_exact(s, 4))[0]
            res2_bytes = recv_exact(s, res2_size)

            res1_img = cv2.imdecode(np.frombuffer(res1_bytes, np.uint8), cv2.IMREAD_COLOR)
            res2_img = cv2.imdecode(np.frombuffer(res2_bytes, np.uint8), cv2.IMREAD_COLOR)
            if res1_img is None or res2_img is None:
                logging.error("Failed to decode result images")
                break

            if (res1_img.shape[1], res1_img.shape[0]) != (w1, h1):
                res1_img = cv2.resize(res1_img, (w1, h1), interpolation=cv2.INTER_LINEAR)
            if (res2_img.shape[1], res2_img.shape[0]) != (w2, h2):
                res2_img = cv2.resize(res2_img, (w2, h2), interpolation=cv2.INTER_LINEAR)

            writer1.write(res1_img)
            writer2.write(res2_img)

            if frame_idx % 10 == 0:
                elapsed = time.time() - start_time
                cur_fps = frame_idx / elapsed if elapsed > 0 else 0
                logging.info(f"Processed {frame_idx} paired frames, avg FPS: {cur_fps:.2f}")

    cap1.release(); cap2.release()
    writer1.release(); writer2.release()
    logging.info(f"Done. Total paired frames: {frame_idx}.\n  {args.out1}\n  {args.out2}")


def run_single(args):
    # Pick source/out by --single-source
    if args.single_source == 1:
        vid_path, out_path = args.video1, args.out1
    else:
        vid_path, out_path = args.video2, args.out2

    cap = open_cap(vid_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = make_writer(out_path, fps, w, h)

    frame_idx = 0
    start_time = time.time()

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((args.host, args.port))
        logging.info(f"Connected to server at {args.host}:{args.port} (single mode, source {args.single_source})")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1

            img_bytes = encode_jpg(frame, args.jpeg_quality)

            # ---- send single ----
            s.sendall(struct.pack('!I', frame_idx))
            s.sendall(struct.pack('!I', len(img_bytes))); s.sendall(img_bytes)

            # ---- receive single result ----
            resp_idx = struct.unpack('!I', recv_exact(s, 4))[0]
            if resp_idx != frame_idx:
                logging.warning(f"Frame index mismatch: sent {frame_idx}, got {resp_idx}")

            res_size = struct.unpack('!I', recv_exact(s, 4))[0]
            res_bytes = recv_exact(s, res_size)
            res_img = cv2.imdecode(np.frombuffer(res_bytes, np.uint8), cv2.IMREAD_COLOR)
            if res_img is None:
                logging.error("Failed to decode result image")
                break

            if (res_img.shape[1], res_img.shape[0]) != (w, h):
                res_img = cv2.resize(res_img, (w, h), interpolation=cv2.INTER_LINEAR)

            writer.write(res_img)

            if frame_idx % 10 == 0:
                elapsed = time.time() - start_time
                cur_fps = frame_idx / elapsed if elapsed > 0 else 0
                logging.info(f"Processed {frame_idx} frames (single), avg FPS: {cur_fps:.2f}")

    cap.release()
    writer.release()
    logging.info(f"Done. Total frames: {frame_idx}.\n  {out_path}")


def parse_args():
    ap = argparse.ArgumentParser(description="Client for segmentation server (pair or single).")
    ap.add_argument("--mode", choices=["pair", "single"], default="pair",
                    help="Protocol mode to match the server.")
    ap.add_argument("--single-source", type=int, choices=[1, 2], default=1,
                    help="Which video to use in --mode single.")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5000)

    ap.add_argument("--video1", default="test_videos/test_video_5.mp4")
    ap.add_argument("--video2", default="test_videos/test_video_6.mp4")
    ap.add_argument("--out1",   default="test_results/segmented_result_5.avi")
    ap.add_argument("--out2",   default="test_results/segmented_result_6.avi")

    ap.add_argument("--jpeg-quality", type=int, default=75)
    return ap.parse_args()


def main():
    args = parse_args()
    if args.mode == "pair":
        run_pair(args)
    else:
        run_single(args)


if __name__ == "__main__":
    main()
