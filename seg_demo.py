import cv2, numpy as np, time, queue, threading, os
from collections import deque
from openvino.runtime import Core

VIDEO_PATH = os.path.join(os.path.dirname(__file__), "test", "test_video_6.mp4")
MODEL_PATH = "intel/semantic-segmentation-adas-0001/FP16-INT8/semantic-segmentation-adas-0001.xml"
NUM_REQS = 3          # try 2 or 3
WRITE_OUTPUT = True

# ---- OpenVINO setup ----
core = Core()
model = core.read_model(MODEL_PATH)
compiled = core.compile_model(model, "CPU", {
    "PERFORMANCE_HINT": "LATENCY",
    "NUM_STREAMS": "1",
    "INFERENCE_NUM_THREADS": os.cpu_count() or 8,
})
inp = compiled.input(0); outp = compiled.output(0)
in_h, in_w = int(inp.shape[2]), int(inp.shape[3])

# ---- Video IO ----
cap = cv2.VideoCapture(VIDEO_PATH, cv2.CAP_FFMPEG)
if not cap.isOpened():
    raise RuntimeError(f"Cannot open {VIDEO_PATH} with FFmpeg backend.")
fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
writer = cv2.VideoWriter("segmented_video.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H)) if WRITE_OUTPUT else None

# ---- Color map ----
np.random.seed(42)
color_map = np.random.randint(0, 255, size=(256, 3), dtype=np.uint8)

# ---- Postproc worker ----
done_q = queue.Queue(maxsize=NUM_REQS * 2)
STOP = object()

def postproc_worker():
    while True:
        item = done_q.get()
        if item is STOP: break
        frame_idx, frame_orig, result = item
        seg_map = result.squeeze().astype(np.uint8)                # (1024,2048)
        seg_overlay = color_map[seg_map]
        seg_overlay_resized = cv2.resize(seg_overlay, (W, H), interpolation=cv2.INTER_NEAREST)
        blended = cv2.addWeighted(frame_orig, 0.5, seg_overlay_resized, 0.5, 0)
        if writer: writer.write(blended)

post_thread = threading.Thread(target=postproc_worker, daemon=True)
post_thread.start()

# ---- Create N infer requests and per-slot userdata ----
requests = [compiled.create_infer_request() for _ in range(NUM_REQS)]
userdata  = [None] * NUM_REQS   # stores (frame_idx, frame_copy) for each slot

frame_idx = 0
start_all = time.time()

while True:
    ret, frame = cap.read()
    if not ret: break
    frame_idx += 1

    # Prepare input (resize to model size, then NCHW uint8)
    if frame.shape[1] != in_w or frame.shape[0] != in_h:
        frame_resized = cv2.resize(frame, (in_w, in_h), interpolation=cv2.INTER_LINEAR)
    else:
        frame_resized = frame
    blob = frame_resized.transpose(2, 0, 1)[None].astype(np.uint8)

    slot = (frame_idx - 1) % NUM_REQS
    req = requests[slot]

    # If this slot was used before, wait for it and dispatch its result to postproc
    if userdata[slot] is not None:
        req.wait()  # finish previous inference on this slot
        result = req.get_output_tensor(outp.index).data[:]  # copy numpy
        done_q.put((userdata[slot][0], userdata[slot][1], result))  # (idx, frame, result)

    # Start new async inference in this slot
    userdata[slot] = (frame_idx, frame.copy())
    req.start_async({inp: blob})

# Flush remaining in-flight requests
for slot, req in enumerate(requests):
    if userdata[slot] is not None:
        req.wait()
        result = req.get_output_tensor(outp.index).data[:]
        done_q.put((userdata[slot][0], userdata[slot][1], result))
        userdata[slot] = None

# Stop worker & cleanup
done_q.put(STOP)
post_thread.join()
cap.release()
if writer: writer.release()

total_time = time.time() - start_all
print(f"Done {frame_idx} frames in {total_time:.2f}s  →  {frame_idx/total_time:.2f} FPS (end-to-end)")
