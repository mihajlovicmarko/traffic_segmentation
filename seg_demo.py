import cv2, numpy as np, time, queue, threading, os
from openvino.runtime import Core
import socket
import struct

class SegmentationModel:
    def __init__(self, model_path, num_reqs=3):
        self.core = Core()
        self.model = self.core.read_model(model_path)
        self.compiled = self.core.compile_model(self.model, "CPU", {
            "PERFORMANCE_HINT": "LATENCY",
            "NUM_STREAMS": "1",
            "INFERENCE_NUM_THREADS": os.cpu_count() or 8,
        })
        self.inp = self.compiled.input(0)
        self.outp = self.compiled.output(0)
        self.in_h, self.in_w = int(self.inp.shape[2]), int(self.inp.shape[3])
        self.num_reqs = num_reqs
        self.requests = [self.compiled.create_infer_request() for _ in range(num_reqs)]
        self.userdata = [None] * num_reqs

    def preprocess(self, frame):
        if frame.shape[1] != self.in_w or frame.shape[0] != self.in_h:
            frame_resized = cv2.resize(frame, (self.in_w, self.in_h), interpolation=cv2.INTER_LINEAR)
        else:
            frame_resized = frame
        blob = frame_resized.transpose(2, 0, 1)[None].astype(np.uint8)
        return blob

    def infer_async(self, slot, blob, frame_idx, frame):
        req = self.requests[slot]
        self.userdata[slot] = (frame_idx, frame.copy())
        req.start_async({self.inp: blob})

    def get_result(self, slot):
        req = self.requests[slot]
        req.wait()
        result = req.get_output_tensor(self.outp.index).data[:]
        frame_idx, frame = self.userdata[slot]
        self.userdata[slot] = None
        return frame_idx, frame, result

    def flush(self):
        results = []
        for slot, req in enumerate(self.requests):
            if self.userdata[slot] is not None:
                req.wait()
                result = req.get_output_tensor(self.outp.index).data[:]
                frame_idx, frame = self.userdata[slot]
                results.append((frame_idx, frame, result))
                self.userdata[slot] = None
        return results

class SegmentationVideoProcessor:
    def __init__(self, video_path, model_path, write_output=True, num_reqs=3):
        self.video_path = video_path
        self.model_path = model_path
        self.write_output = write_output
        self.num_reqs = num_reqs
        self.cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open {video_path} with FFmpeg backend.")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.W = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.H = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.writer = cv2.VideoWriter("segmented_video.mp4", cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (self.W, self.H)) if write_output else None
        np.random.seed(42)
        self.color_map = np.random.randint(0, 255, size=(256, 3), dtype=np.uint8)
        self.done_q = queue.Queue(maxsize=num_reqs * 2)
        self.STOP = object()
        self.model = SegmentationModel(model_path, num_reqs)
        self.post_thread = threading.Thread(target=self.postproc_worker, daemon=True)
        self.post_thread.start()

    def postproc_worker(self):
        while True:
            item = self.done_q.get()
            if item is self.STOP: break
            frame_idx, frame_orig, result = item
            seg_map = result.squeeze().astype(np.uint8)
            seg_overlay = self.color_map[seg_map]
            seg_overlay_resized = cv2.resize(seg_overlay, (self.W, self.H), interpolation=cv2.INTER_NEAREST)
            blended = cv2.addWeighted(frame_orig, 0.5, seg_overlay_resized, 0.5, 0)
            if self.writer: self.writer.write(blended)

    def process(self):
        frame_idx = 0
        start_all = time.time()
        while True:
            ret, frame = self.cap.read()
            if not ret: break
            frame_idx += 1
            blob = self.model.preprocess(frame)
            slot = (frame_idx - 1) % self.num_reqs
            if self.model.userdata[slot] is not None:
                result_tuple = self.model.get_result(slot)
                self.done_q.put(result_tuple)
            self.model.infer_async(slot, blob, frame_idx, frame)
        # Flush remaining
        for result_tuple in self.model.flush():
            self.done_q.put(result_tuple)
        self.done_q.put(self.STOP)
        self.post_thread.join()
        self.cap.release()
        if self.writer: self.writer.release()
        total_time = time.time() - start_all
        print(f"Done {frame_idx} frames in {total_time:.2f}s  →  {frame_idx/total_time:.2f} FPS (end-to-end)")

    def process_image(self, image):
        """
        Process a single image and return the blended segmentation result.
        Accepts either a numpy array or a file path.
        """
        if isinstance(image, str):
            frame = cv2.imread(image)
            if frame is None:
                raise ValueError(f"Cannot read image from path: {image}")
        else:
            frame = image

        blob = self.model.preprocess(frame)
        self.model.infer_async(0, blob, 0, frame)
        frame_idx, frame_orig, result = self.model.get_result(0)
        seg_map = result.squeeze().astype(np.uint8)
        seg_overlay = self.color_map[seg_map]
        seg_overlay_resized = cv2.resize(seg_overlay, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
        blended = cv2.addWeighted(frame_orig, 0.5, seg_overlay_resized, 0.5, 0)
        return blended

    def serve_socket(self, host='127.0.0.1', port=5000):
        """
        Start a socket server that waits for image bytes, processes them,
        and returns the segmented image bytes to the client.
        """
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
            server.bind((host, port))
            server.listen(1)
            print(f"Socket server listening on {host}:{port}")
            while True:
                conn, addr = server.accept()
                with conn:
                    print(f"Connected by {addr}")
                    # Receive image size (4 bytes, unsigned int)
                    size_data = conn.recv(4)
                    if not size_data:
                        continue
                    img_size = struct.unpack('!I', size_data)[0]
                    # Receive image bytes
                    img_bytes = b''
                    while len(img_bytes) < img_size:
                        chunk = conn.recv(img_size - len(img_bytes))
                        if not chunk:
                            break
                        img_bytes += chunk
                    # Decode image
                    nparr = np.frombuffer(img_bytes, np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    if frame is None:
                        print("Failed to decode image")
                        continue
                    # Process image
                    result_img = self.process_image(frame)
                    # Encode result as JPEG
                    _, result_bytes = cv2.imencode('.jpg', result_img)
                    result_bytes = result_bytes.tobytes()
                    # Send result size and bytes
                    conn.sendall(struct.pack('!I', len(result_bytes)))
                    conn.sendall(result_bytes)
                    print(f"Processed and sent result to {addr}")

def main():
    VIDEO_PATH = os.path.join(os.path.dirname(__file__), "test", "test_video_6.mp4")
    MODEL_PATH = "intel/semantic-segmentation-adas-0001/FP16-INT8/semantic-segmentation-adas-0001.xml"
    NUM_REQS = 3
    WRITE_OUTPUT = True
    processor = SegmentationVideoProcessor(VIDEO_PATH, MODEL_PATH, WRITE_OUTPUT, NUM_REQS)
    # Uncomment below to run socket server instead of video processing
    # processor.serve_socket(host='127.0.0.1', port=5000)
    processor.process()

if __name__ == "__main__":
    main()
