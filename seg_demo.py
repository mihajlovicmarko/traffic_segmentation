import cv2, numpy as np, time, queue, threading, os
from collections import deque
from openvino.runtime import Core

VIDEO_PATH = os.path.join(os.path.dirname(__file__), "test", "test_video_6.mp4")
MODEL_PATH = "intel/semantic-segmentation-adas-0001/FP16-INT8/semantic-segmentation-adas-0001.xml"
NUM_REQS = 3
WRITE_OUTPUT = True

# ---- OpenVINO setup ----
t0 = time.time()
core = Core()
model = core.read_model(MODEL_PATH)
compiled = core.compile_model(model, "CPU", {
    "PERFORMANCE_HINT": "LATENCY",
    "NUM_STREAMS": "1",
    "INFERENCE_NUM_THREADS": os.cpu_count() or 8,
})
inp = compiled.input(0); outp = compiled.output(0)
in_h, in_w = int(inp.shape[2]), int(inp.shape[3])
print(f"[Init] Model load & compile: {time.time() - t0:.4f}s")

# ---- Video IO ----
t0 = time.time()
cap = cv2.VideoCapture(VIDEO_PATH, cv2.CAP_FFMPEG)
if not cap.isOpened():
    raise RuntimeError(f"Cannot open {VIDEO_PATH} with FFmpeg backend.")
fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
writer = cv2.VideoWriter("segmented_video.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H)) if WRITE_OUTPUT else None
print(f"[Init] Video open: {time.time() - t0:.4f}s")

# ---- Color map ----
np.random.seed(42)
color_map = np.random.randint(0, 255, size=(256, 3), dtype=np.uint8)

# ---- Postproc worker ----
done_q = queue.Queue(maxsize=NUM_REQS * 2)
STOP = object()

def postproc_worker():
    while True:
        item = done_q.get()
        if item is STOP:
            break
        t_start = time.time()
        frame_idx, frame_orig, result = item
        seg_map = result.squeeze().astype(np.uint8)
        seg_overlay = color_map[seg_map]
        seg_overlay_resized = cv2.resize(seg_overlay, (W, H), interpolation=cv2.INTER_NEAREST)
        blended = cv2.addWeighted(frame_orig, 0.5, seg_overlay_resized, 0.5, 0)
        if writer:
            writer.write(blended)
        print(f"[Postproc] Frame {frame_idx}: {time.time() - t_start:.4f}s")

post_thread = threading.Thread(target=postproc_worker, daemon=True)
post_thread.start()

# ---- Create N infer requests ----
requests = [compiled.create_infer_request() for _ in range(NUM_REQS)]
userdata  = [None] * NUM_REQS

frame_idx = 0
start_all = time.time()

while True:
    t_frame_start = time.time()

    # --- Read frame ---
    t0 = time.time()
    ret, frame = cap.read()
    t_read = time.time() - t0
    if not ret:
        break
    frame_idx += 1

    # --- Preprocess ---
    t0 = time.time()
    if frame.shape[1] != in_w or frame.shape[0] != in_h:
        frame_resized = cv2.resize(frame, (in_w, in_h), interpolation=cv2.INTER_LINEAR)
    else:
        frame_resized = frame
    blob = frame_resized.transpose(2, 0, 1)[None].astype(np.uint8)
    t_preproc = time.time() - t0

    slot = (frame_idx - 1) % NUM_REQS
    req = requests[slot]

    # --- Wait & get previous result ---
    t0 = time.time()
    if userdata[slot] is not None:
        req.wait()
        result = req.get_output_tensor(outp.index).data[:]
        done_q.put((userdata[slot][0], userdata[slot][1], result))
    t_wait_get = time.time() - t0

    # --- Start async inference ---
    t0 = time.time()
    userdata[slot] = (frame_idx, frame.copy())
    req.start_async({inp: blob})
    t_infer_start = time.time() - t0

    print(f"[Frame {frame_idx}] Read: {t_read:.4f}s | Preproc: {t_preproc:.4f}s | Wait+Get: {t_wait_get:.4f}s | StartInfer: {t_infer_start:.4f}s | Total so far: {time.time() - t_frame_start:.4f}s")

# Flush remaining
for slot, req in enumerate(requests):
    if userdata[slot] is not None:
        req.wait()
        result = req.get_output_tensor(outp.index).data[:]
        done_q.put((userdata[slot][0], userdata[slot][1], result))
        userdata[slot] = None

done_q.put(STOP)
post_thread.join()
cap.release()
if writer:
    writer.release()

total_time = time.time() - start_all
print(f"Done {frame_idx} frames in {total_time:.2f}s  →  {frame_idx/total_time:.2f} FPS (end-to-end)")
