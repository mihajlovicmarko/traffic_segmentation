import numpy as np
import cv2


def dzida():
    print("dzida")


def tensor_to_video(
    tensor: np.ndarray,
    out_path: str = "output.mp4",
    fps: int = 12,
    normalize: bool = True,
    codec: str = "mp4v",
    value_map: dict | None = None,
):
    """
    Write a tensor as an MP4 video.

    Args:
        tensor: (H, W, N) grayscale or (H, W, 3, N) RGB.
                Dtype can be float or uint8.
        out_path: Output file path, e.g. "out.mp4".
        fps: Frames per second.
        normalize: If True, linearly scale by a single global max to [0,255].
                   If False, assume values already in [0,255] when uint8.
        codec: FourCC codec (e.g., "mp4v", "avc1").
        value_map: Optional dict for discrete values -> RGB mapping.
                   Example for 0/1/2:
                     {0:(0,0,0), 1:(255,255,255), 2:(0,255,255)}
                   Mapping is assumed in RGB; will be converted to BGR for OpenCV.
    """
    assert tensor.ndim in (3, 4), "Tensor must be (H,W,N) or (H,W,3,N)"
    if tensor.ndim == 3:
        H, W, N = tensor.shape
        C = 1
    else:
        H, W, C, N = tensor.shape
        assert C == 3, "Color tensor must be (H,W,3,N)"

    # Prepare scaling once (avoid per-frame flicker)
    tensor_f = tensor.astype(np.float32, copy=False)
    if normalize:
        tmax = float(tensor_f.max())
        scale = 255.0 / tmax if tmax > 0 else 0.0
    else:
        scale = 1.0

    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(out_path, fourcc, fps, (W, H), isColor=True)
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter for '{out_path}'. Try a different codec or path.")

    # Prebuild BGR palette if value_map is provided
    bgr_map = None
    if value_map is not None:
        bgr_map = {k: (v[2], v[1], v[0]) for k, v in value_map.items()}  # RGB -> BGR

    for i in range(N):
        if C == 1:
            frame = tensor_f[:, :, i]

            if bgr_map is not None:
                # Map discrete values to BGR colors
                frame_bgr = np.zeros((H, W, 3), dtype=np.uint8)
                for val, color_bgr in bgr_map.items():
                    mask = (frame == val)
                    frame_bgr[mask] = color_bgr
            else:
                # Grayscale -> 3 channel
                if normalize:
                    fr = (frame * scale).clip(0, 255).astype(np.uint8)
                else:
                    fr = frame.astype(np.uint8)
                frame_bgr = cv2.merge([fr, fr, fr])

        else:
            # Color path: assume RGB input, convert to BGR
            frame_rgb = tensor_f[:, :, :, i]
            if normalize:
                fr = (frame_rgb * scale).clip(0, 255).astype(np.uint8)
            else:
                fr = frame_rgb.astype(np.uint8)
            frame_bgr = cv2.cvtColor(fr, cv2.COLOR_RGB2BGR)

        writer.write(np.ascontiguousarray(frame_bgr))

    writer.release()




def make_rotating_rect_tensor(
    H=236, W=330, n=12, *,
    length=120,
    thickness=12
) -> np.ndarray:
    """
    Create a tensor of shape (H, W, n) with one filled rectangle per slice.
    The rectangle is anchored at the bottom-center and rotates from left to right.
    """
    assert n >= 2, "Need at least two layers to interpolate angles."
    T = np.zeros((H, W, n), dtype=np.uint8)

    x0, y0 = W // 2, H - 1  # Anchor at bottom-center

    for i in range(n):
        theta_deg = 180.0 - i * (180.0 / (n - 1))  # from 180° to 0°
        theta = np.deg2rad(theta_deg)

        # direction vector
        dx = np.cos(theta)
        dy = -np.sin(theta)  # image coords: y down → invert sin

        x1 = x0 + length * dx
        y1 = y0 + length * dy

        # normal vector for thickness
        nx = -dy
        ny = dx

        half_t = thickness / 2.0

        p0 = np.array([x0, y0], dtype=np.float32)
        p1 = np.array([x1, y1], dtype=np.float32)
        nvec = np.array([nx, ny], dtype=np.float32)

        polygon = np.stack([
            p0 + nvec * half_t,
            p0 - nvec * half_t,
            p1 - nvec * half_t,
            p1 + nvec * half_t
        ]).astype(np.int32)

        canvas = np.zeros((H, W), dtype=np.uint8)
        cv2.fillPoly(canvas, [polygon], 1)
        T[:, :, i] = canvas

    return T
