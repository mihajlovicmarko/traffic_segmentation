import cv2
import os
import time
import argparse
import subprocess

def set_camera_controls(device, exposure=100, brightness=-20, contrast=60):
    """Set camera parameters using v4l2-ctl."""
    print(f"🎛 Reducing ISO for {device}")
    subprocess.run(["v4l2-ctl", "-d", device, "--set-ctrl", "exposure_auto=1"])  # Manual exposure
    subprocess.run(["v4l2-ctl", "-d", device, "--set-ctrl", f"exposure_absolute={exposure}"])
    subprocess.run(["v4l2-ctl", "-d", device, "--set-ctrl", f"brightness={brightness}"])
    subprocess.run(["v4l2-ctl", "-d", device, "--set-ctrl", f"contrast={contrast}"])


def record_dual_cameras(cam1_path="/dev/video0", cam2_path="/dev/video2",
                        save_dir="recordings", fps=15, chunk_duration=20, codec="mp4v"):
    """
    Records from two cameras simultaneously at given FPS,
    splitting recordings into chunks of `chunk_duration` seconds.
    """
    os.makedirs(save_dir, exist_ok=True)
    set_camera_controls(cam2_path, exposure=10, brightness=-50, contrast=60)

    cam1 = cv2.VideoCapture(cam1_path, cv2.CAP_V4L2)
    cam2 = cv2.VideoCapture(cam2_path, cv2.CAP_V4L2)

    # Set resolution
    width, height = 640, 480
    cam1.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cam1.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cam2.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cam2.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cam1.set(cv2.CAP_PROP_FPS, fps)
    cam2.set(cv2.CAP_PROP_FPS, fps)

    if not cam1.isOpened() or not cam2.isOpened():
        print("❌ Error: Could not open one or both cameras.")
        return

    print(f"✅ Recording started at {fps} FPS, chunks: {chunk_duration}s each.")
    print(f"Files will be saved in: {save_dir}")
    print("Press Ctrl+C to stop.")

    # Define codec (H.264 or H.265 recommended for small file size)
    fourcc = cv2.VideoWriter_fourcc(*codec)

    try:
        while True:
            timestamp = int(time.time())
            out1_path = os.path.join(save_dir, f"cam1_{timestamp}.mp4")
            out2_path = os.path.join(save_dir, f"cam2_{timestamp}.mp4")

            out1 = cv2.VideoWriter(out1_path, fourcc, fps, (width, height))
            out2 = cv2.VideoWriter(out2_path, fourcc, fps, (width, height))

            start_time = time.time()
            frame_interval = 1.0 / fps

            while time.time() - start_time < chunk_duration:
                frame_start = time.time()

                ret1, frame1 = cam1.read()
                ret2, frame2 = cam2.read()

                if not ret1 or not ret2:
                    print("❌ Frame read error. Stopping.")
                    return

                out1.write(frame1)
                out2.write(frame2)

                # Maintain FPS timing
                elapsed = time.time() - frame_start
                if elapsed < frame_interval:
                    time.sleep(frame_interval - elapsed)

            out1.release()
            out2.release()
            print(f"💾 Saved: {out1_path} & {out2_path}")

    except KeyboardInterrupt:
        print("\n🛑 Recording stopped by user.")

    finally:
        cam1.release()
        cam2.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dual USB Camera Recorder with segmentation")
    parser.add_argument("--cam1", default="/dev/video0", help="Path to first camera (default: /dev/video0)")
    parser.add_argument("--cam2", default="/dev/video2", help="Path to second camera (default: /dev/video2)")
    parser.add_argument("--dir", default="recordings", help="Directory to save recordings")
    parser.add_argument("--fps", type=int, default=15, help="Frames per second (default: 15)")
    parser.add_argument("--chunk", type=int, default=20, help="Chunk duration in seconds (default: 20)")
    parser.add_argument("--codec", default="mp4v", help="Codec: mp4v (H.264), H264, XVID (default: mp4v)")
    args = parser.parse_args()

    record_dual_cameras(args.cam1, args.cam2, args.dir, args.fps, args.chunk, args.codec)
