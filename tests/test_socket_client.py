import socket
import struct
import cv2
import numpy as np
import logging
import time
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

HOST = '127.0.0.1'
PORT = 5000

VIDEO_PATH_1 = 'test_videos/test_video_5.mp4'
VIDEO_PATH_2 = 'test_videos/test_video_6.mp4'
OUT_PATH_1   = 'test_results/segmented_result_5.avi'
OUT_PATH_2   = 'test_results/segmented_result_6.avi'

def open_cap(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        logging.error(f"Cannot open video: {path}")
        raise FileNotFoundError(path)
    return cap

def make_writer(path, fps, w, h):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"XVID"), fps, (w, h))

def recv_exact(sock, n):
    buf = b''
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("Socket closed while receiving data")
        buf += chunk
    return buf

def main():
    cap1 = open_cap(VIDEO_PATH_1)
    cap2 = open_cap(VIDEO_PATH_2)

    fps = cap1.get(cv2.CAP_PROP_FPS) or 25.0  # use first video's fps for output pacing
    w1, h1 = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w2, h2 = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer1 = make_writer(OUT_PATH_1, fps, w1, h1)
    writer2 = make_writer(OUT_PATH_2, fps, w2, h2)

    frame_idx = 0
    start_time = time.time()

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))
        logging.info(f"Connected to server at {HOST}:{PORT}")

        while True:
            ret1, frame1 = cap1.read()
            ret2, frame2 = cap2.read()
            if not ret1 or not ret2:
                break  # stop when either video ends

            frame_idx += 1

            ok1, img1_bytes = cv2.imencode('.jpg', frame1)
            ok2, img2_bytes = cv2.imencode('.jpg', frame2)
            if not ok1 or not ok2:
                logging.error("JPEG encode failed")
                break
            img1_bytes = img1_bytes.tobytes()
            img2_bytes = img2_bytes.tobytes()

            # ==== send pair (n-th frames together) ====
            header = struct.pack('!I', frame_idx)
            s.sendall(header)
            s.sendall(struct.pack('!I', len(img1_bytes))); s.sendall(img1_bytes)
            s.sendall(struct.pack('!I', len(img2_bytes)));  s.sendall(img2_bytes)

            # ==== receive paired results ====
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

            # To match original sizes per stream, resize if needed
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
    logging.info(f"Done. Total paired frames: {frame_idx}. Saved to:\n  {OUT_PATH_1}\n  {OUT_PATH_2}")

if __name__ == "__main__":
    main()
