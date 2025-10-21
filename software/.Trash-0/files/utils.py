import numpy as np
import cv2

def tensor_to_video(
    tensor: np.ndarray,
    out_path: str = "output.mp4",
    fps: int = 12,
    scale_to_255: bool = True,
    is_color: bool = False,
):
    """
    Save a 3D (H, W, N) or 4D (H, W, 3, N) tensor as an MP4 video.

    Parameters:
        tensor (np.ndarray): Tensor with shape (H, W, N) for grayscale or (H, W, 3, N) for RGB.
        out_path (str): Output video path.
        fps (int): Frames per second.
        scale_to_255 (bool): Whether to rescale input values to [0, 255].
        is_color (bool): Whether the input is RGB. If False, video is grayscale.
    """
    assert tensor.ndim in [3, 4], "Tensor must be 3D (H,W,N) or 4D (H,W,3,N)"
    if tensor.ndim == 3:
        H, W, N = tensor.shape
    else:
        H, W, C, N = tensor.shape
        assert C == 3, "Color video must have 3 channels"

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_path, fourcc, fps, (W, H), isColor=is_color)

    for i in range(N):
        if tensor.ndim == 3:
            frame = tensor[:, :, i]
        else:
            frame = tensor[:, :, :, i]
            frame = np.transpose(frame, (1, 0, 2)) if frame.shape[0] != H else frame

        if scale_to_255:
            frame = (255 * (frame.astype(np.float32) / frame.max())).astype(np.uint8)
        else:
            frame = frame.astype(np.uint8)

        if not is_color:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        writer.write(frame)

    writer.release()
    print(f"Video saved to: {out_path}")

