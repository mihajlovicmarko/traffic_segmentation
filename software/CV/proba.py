import numpy as np
import io
import cv2
from dataclasses import dataclass
import matplotlib.pyplot as plt
import os

@dataclass
class BEVGrid:
    """Defines the bird's‑eye grid in world meters."""
    meters_per_pixel: float = 0.05
    forward_range_m: tuple[float, float] = (0.0, 30.0)   # (y_min, y_max) forward
    lateral_range_m: tuple[float, float] = (-5.0, 5.0)   # (x_min, x_max) left/right

    @property
    def size(self) -> tuple[int, int]:
        """(W_out, H_out) in pixels for cv2.warpPerspective"""
        x_min, x_max = self.lateral_range_m
        y_min, y_max = self.forward_range_m
        W_out = int(np.round((x_max - x_min) / self.meters_per_pixel))
        H_out = int(np.round((y_max - y_min) / self.meters_per_pixel))
        return (W_out, H_out)

    def scaling_matrix(self) -> np.ndarray:
        """
        S maps destination pixel coords (u,v,1)^T to ground meters (x,y,1)^T.
        y grows forward; we place y forward downward in the BEV image so rows increase with distance.
        """
        x_min, _x_max = self.lateral_range_m
        _y_min, y_max = self.forward_range_m
        mpp = self.meters_per_pixel
        S = np.array([[ mpp,  0.0, x_min],
                      [ 0.0, -mpp, y_max],
                      [ 0.0,  0.0, 1.0 ]], dtype=np.float64)
        return S

class BEVProjector:
    """
    Bird's‑eye view projector.

    Supports:
      1) Calibrated plane homography (ground z=0) using intrinsics + pose.
      2) Trapezoid→rectangle homography (no calibration).

    Conventions:
      World: X right, Y forward, Z up.
      Camera (OpenCV): x right, y down, z forward.
      roll/pitch/yaw are applied as world->camera, ZYX order (yaw, pitch, roll).
    """

    def __init__(self, K: np.ndarray | None = None):
        self.K = None if K is None else self._validate_K(K)
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0
        self.height_m = 1.0
        self._H_pg = None  # ground->image homography (if calibrated set)

    # ---------- Public API ----------

    def set_intrinsics(self, K: np.ndarray) -> "BEVProjector":
        self.K = self._validate_K(K)
        self._H_pg = None
        return self

    def set_pose(self, height_m: float, roll: float = 0.0, pitch: float = 0.0, yaw: float = 0.0) -> "BEVProjector":
        self.height_m = float(height_m)
        self.roll = float(roll)
        self.pitch = float(pitch)
        self.yaw = float(yaw)
        self._H_pg = None
        return self

    def warp_calibrated(self, mask01: np.ndarray, grid: BEVGrid) -> np.ndarray:
        """
        Warp a binary mask (1/0, bool or uint8) to BEV using calibration.
        Returns uint8 0/255 mask of shape (H_out, W_out).
        """
        assert self.K is not None, "Call set_intrinsics(K) first."
        H_pg = self._ground_to_image_homography()  # 3x3
        self.H_pg = H_pg
        
        S = grid.scaling_matrix()
        self.S = S
        

        # We need M such that dst <- src: p_dst = S^{-1} * H_pg^{-1} * p_img
        M = np.linalg.inv(S) @ np.linalg.inv(H_pg)
        self.M = M

        
        src = self._to_uint8_255(mask01)
        W_out, H_out = grid.size
        
        bev = cv2.warpPerspective(
            src, M, (W_out, H_out),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0
        )
        
        return bev

    def warp_trapezoid(self, mask01: np.ndarray, src_quad: np.ndarray, out_size: tuple[int, int]) -> np.ndarray:
        """
        Warp a perspective trapezoid to a rectangle (no calibration needed).
        src_quad: 4x2 float32 in order [TL, TR, BR, BL]
        out_size: (W_out, H_out)
        """
        src = self._to_uint8_255(mask01)
        W_out, H_out = out_size
        dst_quad = np.array([[0, 0],
                             [W_out - 1, 0],
                             [W_out - 1, H_out - 1],
                             [0, H_out - 1]], dtype=np.float32)
        M = cv2.getPerspectiveTransform(src_quad.astype(np.float32), dst_quad)
        bev = cv2.warpPerspective(
            src, M, (W_out, H_out),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0
        )
        return bev

    def get_calibrated_homography(self, grid: BEVGrid) -> np.ndarray:
        """
        Returns the overall SRC->DST warp matrix M for calibrated BEV:
            p_dst = M * p_img
        so you can reuse it for many frames.
        """
        assert self.K is not None, "Call set_intrinsics(K) first."
        H_pg = self._ground_to_image_homography()
        S = grid.scaling_matrix()
        M = np.linalg.inv(S) @ np.linalg.inv(H_pg)
        return M

    # ---------- Internals ----------

    @staticmethod
    def _validate_K(K: np.ndarray) -> np.ndarray:
        K = np.asarray(K, dtype=np.float64)
        if K.shape != (3, 3) or not np.isfinite(K).all():
            raise ValueError("K must be 3x3 finite matrix")
        return K

    @staticmethod
    def _rot_from_rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
        """
        World:  X right, Y forward, Z up
        Camera: x right, y down,   z forward (OpenCV)
        Rotate in WORLD (yaw->pitch->roll), then align world->camera.
        """
        # Base alignment: Xw->xc, Yw->zc, Zw->-yc
        R_align = np.array([
            [1.0, 0.0,  0.0],  # Xw -> +xc
            [0.0, 0.0, -1.0],  # Zw -> -yc
            [0.0, 1.0,  0.0],  # Yw -> +zc
        ], dtype=np.float64)
    
        cz, sz = np.cos(yaw),   np.sin(yaw)
        cy, sy = np.cos(pitch), np.sin(pitch)
        cx, sx = np.cos(roll),  np.sin(roll)
    
        Rz = np.array([[ cz, -sz, 0],
                       [ sz,  cz, 0],
                       [  0,   0, 1]], dtype=np.float64)
        Ry = np.array([[ cy, 0, sy],
                       [  0, 1,  0],
                       [-sy, 0, cy]], dtype=np.float64)
        Rx = np.array([[1,  0,   0],
                       [0, cx, -sx],
                       [0, sx,  cx]], dtype=np.float64)
    
        R_world = Rx @ Ry @ Rz          # world -> rotated-world
        return R_align @ R_world        # world -> camera

    def _ground_to_image_homography(self) -> np.ndarray:
        """
        Builds H_pg such that p_img ~ H_pg * p_ground (z=0 plane).
        H_pg = K [r1 r2 t], where t = -R*C and C=(0,0,h)^T (camera center in world).
        """
        if self._H_pg is not None:
            return self._H_pg

        R_wc = self._rot_from_rpy(self.roll, self.pitch, self.yaw)  # world->camera
        t_wc = np.array([[0.0], [0.0], [self.height_m]], dtype=np.float64)  # camera at (0,0,h) in world

        r1 = R_wc[:, 0:1]
        r2 = R_wc[:, 1:2]
        t = R_wc @ (-t_wc)  # -R*C
        H_pg = self.K @ np.hstack([r1, r2, t])
        self._H_pg = H_pg
        return H_pg

    @staticmethod
    def _to_uint8_255(mask01: np.ndarray) -> np.ndarray:
        """Ensure input is uint8 in {0,255} for nearest-neighbor warps."""
        if mask01.dtype != np.uint8:
            src = (mask01.astype(np.uint8) * 255)
        else:
            src = mask01.copy()
            if src.max() == 1:
                src *= 255
        return src






def _build_intrinsics(W, H, K0=None, K0_size=None, fx=1100.0, fy=None):
    if K0 is not None and K0_size is not None:
        W0, H0 = K0_size
        sx, sy = W / float(W0), H / float(H0)
        return np.array([
            [K0[0,0]*sx, 0.0,           K0[0,2]*sx],
            [0.0,        K0[1,1]*sy,    K0[1,2]*sy],
            [0.0,        0.0,           1.0]
        ], dtype=np.float64)
    
    if fy is None:
        fy = fx
    cx, cy = W / 2.0, H / 2.0
    return np.array([
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)
# REPLACE the strict converters with robust ones

def indices_to_onehot_safe(
    idx2d: np.ndarray,
    valid_ids: list[int] | range = range(1, 13),  # labels you want to keep
    out_num_classes: int = 12,                     # K channels in output
    id_map: dict[int, int] | None = None,         # optional raw->target map (1..K)
    background_ids: tuple[int, ...] = (0,),       # labels to ignore
    dtype=np.uint8
) -> np.ndarray:
    """
    Robust 2D indices -> one-hot (H,W,K):
      - Keeps only valid_ids (or those present in id_map)
      - Ignores labels in background_ids and anything not in valid_ids
      - If id_map is provided, it can merge multiple raw labels into one channel
    """
    if idx2d.ndim != 2:
        raise ValueError(f"idx2d must be 2D, got shape {idx2d.shape}")

    h, w = idx2d.shape
    onehot = np.zeros((h, w, out_num_classes), dtype=dtype)

    if id_map:
        for raw_id, class_id in id_map.items():
            if not (1 <= class_id <= out_num_classes):
                raise ValueError(f"class_id {class_id} out of 1..{out_num_classes}")
            m = (idx2d == raw_id)
            if m.any():
                onehot[..., class_id - 1] |= m.astype(dtype)
        return onehot

    keep = set(valid_ids)
    ignore = set(background_ids)
    # Only light up channels for raw ids we care about (1..K by default)
    for raw_id in np.unique(idx2d):
        if raw_id in ignore:
            continue
        if raw_id in keep and 1 <= raw_id <= out_num_classes:
            m = (idx2d == raw_id)
            if m.any():
                onehot[..., raw_id - 1] |= m.astype(dtype)
        # else: silently ignore stray labels (e.g., 13+)
    return onehot


def stack_indices_to_onehot_safe(
    idx_stack: np.ndarray | list[np.ndarray],
    valid_ids: list[int] | range = range(1, 13),
    out_num_classes: int = 12,
    id_map: dict[int, int] | None = None,
    background_ids: tuple[int, ...] = (0,),
    dtype=np.uint8
) -> np.ndarray:
    """
    Robust (T,H,W) or list[(H,W)] -> (T,H,W,K) one-hot, ignoring stray labels.
    """
    if isinstance(idx_stack, np.ndarray):
        if idx_stack.ndim != 3:
            raise ValueError(f"idx_stack must be (T,H,W), got {idx_stack.shape}")
        frames = [idx_stack[t] for t in range(idx_stack.shape[0])]
    else:
        frames = idx_stack
        if not frames:
            raise ValueError("idx_stack is empty")

    oh_list = [
        indices_to_onehot_safe(
            f,
            valid_ids=valid_ids,
            out_num_classes=out_num_classes,
            id_map=id_map,
            background_ids=background_ids,
            dtype=dtype,
        )
        for f in frames
    ]
    return np.stack(oh_list, axis=0)  # (T,H,W,K)

def video_to_bev_tensor(
    video_tensor,
    segmented_video,
    n_frames=None,
    *,
    height_m=1.4,
    pitch_deg_down=6.0,
    roll_deg=0.0,
    yaw_deg=0.0,
    K0=None,
    K0_size=None,
    fx=1100.0,
    fy=None,
    meters_per_pixel=0.10,
    forward_range=(0.5, 40.0),
    lateral_range=(-8.0, 8.0),
    debug=False
):
    N_all = min(len(video_tensor), len(segmented_video))
    N = N_all if n_frames is None else min(n_frames, N_all)

    H, W = segmented_video[0].shape[:2]
    K = _build_intrinsics(W, H, K0=K0, K0_size=K0_size, fx=fx, fy=fy)

    roll = np.deg2rad(roll_deg)
    pitch = -np.deg2rad(pitch_deg_down)
    yaw = np.deg2rad(yaw_deg)

    projector = BEVProjector().set_intrinsics(K).set_pose(
        height_m=height_m,
        roll=roll,
        pitch=pitch,
        yaw=yaw
    )

    grid = BEVGrid(
        meters_per_pixel=meters_per_pixel,
        forward_range_m=forward_range,
        lateral_range_m=lateral_range
    )

    W_out, H_out = grid.size
    bev_tensor = np.zeros((H_out, W_out, N), dtype=np.uint8)

    for i in range(N):
        mask01 = (segmented_video[i] == 0).astype(np.uint8)
        bev = projector.warp_calibrated(mask01=mask01, grid=grid)
        bev_tensor[:, :, i] = bev

        if debug and (i == 0 or (i + 1) == N):
            H_pg, S = projector.H_pg, grid.scaling_matrix()
            A = H_pg @ S
            dst = np.array([[0,0,1],[W_out-1,0,1],[W_out-1,H_out-1,1],[0,H_out-1,1]], dtype=np.float64).T
            img_corners = (A @ dst); img_corners = (img_corners[:2]/img_corners[2]).T
            print(f"[DEBUG frame {i}] BEV corners -> image:\n", img_corners)
            print(f"[DEBUG frame {i}] bev.sum() = {bev.sum()}")

    return bev_tensor, projector, grid


def indices_to_onehot(idx2d: np.ndarray,
                      num_classes: int = 12,
                      class_offset: int = 1,
                      dtype=np.uint8) -> np.ndarray:
    """
    Convert a 2D index image with class IDs in [class_offset, class_offset+num_classes-1]
    into a one-hot tensor of shape (H, W, num_classes).
    """
    if idx2d.ndim != 2:
        raise ValueError(f"idx2d must be 2D, got shape {idx2d.shape}")
    h, w = idx2d.shape
    idx_norm = (idx2d.astype(np.int32) - class_offset)
    if (idx_norm.min() < 0) or (idx_norm.max() >= num_classes):
        raise ValueError(f"indices out of range after offset: min={idx_norm.min()}, max={idx_norm.max()}, K={num_classes}")
    classes = np.arange(num_classes, dtype=np.int32).reshape(1, 1, num_classes)
    idx_exp = idx_norm[..., None]
    onehot_bool = (idx_exp == classes)
    return onehot_bool.astype(dtype)

def stack_indices_to_onehot(idx_list,
                            num_classes: int = 12,
                            class_offset: int = 1,
                            dtype=np.uint8) -> np.ndarray:
    """
    Convert a list/iterator of 2D index frames into a 4D one-hot video tensor.
    """
    onehots = [indices_to_onehot(f, num_classes=num_classes, class_offset=class_offset, dtype=dtype)
               for f in idx_list]
    if not onehots:
        raise ValueError("idx_list is empty")
    return np.stack(onehots, axis=0)

def onehot_to_npy_bytes(onehot: np.ndarray) -> bytes:
    """
    Serialize a (H,W,K) or (T,H,W,K) one-hot tensor to .npy bytes.
    """
    buf = io.BytesIO()
    np.save(buf, onehot.astype(np.uint8, copy=False), allow_pickle=False)
    return buf.getvalue()

def npy_bytes_to_onehot(b: bytes) -> np.ndarray:
    """
    Load one-hot tensor from .npy bytes.
    Returns (H,W,K) or (T,H,W,K) depending on what was saved.
    """
    return np.load(io.BytesIO(b), allow_pickle=False)

def onehot_channel(onehot: np.ndarray,
                   class_index_original: int,
                   class_offset: int = 1,
                   squeeze_last: bool = True,
                   dtype=np.uint8) -> np.ndarray:
    """
    Extract one channel from one-hot where `class_index_original` is the original label id (e.g., 1..12).
    Accepts either (H,W,K) or (T,H,W,K). Returns (H,W) or (T,H,W) mask.
    """
    if onehot.ndim == 3:
        ch = class_index_original - class_offset
        if ch < 0 or ch >= onehot.shape[-1]:
            raise ValueError("class_index_original out of range")
        mask = onehot[..., ch]
        return mask.astype(dtype, copy=False) if squeeze_last else mask[..., None].astype(dtype, copy=False)
    elif onehot.ndim == 4:
        ch = class_index_original - class_offset
        if ch < 0 or ch >= onehot.shape[-1]:
            raise ValueError("class_index_original out of range")
        mask = onehot[..., ch]
        return mask.astype(dtype, copy=False) if squeeze_last else mask[..., None].astype(dtype, copy=False)
    else:
        raise ValueError(f"onehot must be 3D or 4D, got {onehot.shape}")

def onehot_to_binary_for_bev(onehot_frame: np.ndarray,
                             road_class_index_original: int,
                             class_offset: int = 1) -> np.ndarray:
    """
    Convert a (H,W,K) one-hot frame into a 2D binary mask (0/1) for the 'road' class.
    """
    mask = onehot_channel(onehot_frame, road_class_index_original, class_offset=class_offset,
                          squeeze_last=True, dtype=np.uint8)
    return mask

def npy_to_bev_tensor(
    path,
    n_frames=None,
    *,
    pitch_deg_down=6.0,
    height_m=1.4,
    fx=1100.0,
    fy=530,
    meters_per_pixel=0.10,
    forward_range=(0.5, 40.0),
    lateral_range=(-8.0, 8.0),
    debug=False,
    class_index_original=1,   # which raw label is “road”
    num_classes=12,
    # class_offset is no longer used; kept for call-compatibility only
    class_offset=1
):
    """
    Loads `.npy` labels (either (T,H,W) with raw ids, or (T,H,W,K) one-hot).
    Robustly converts raw ids into one-hot (ignoring junk labels), extracts the
    desired class channel (default: road=1), projects to BEV over T frames.
    """
    assert os.path.exists(path), f"File not found: {path}"
    segmented_video = np.load(path)

    # Convert to one-hot robustly if needed
    if segmented_video.ndim == 3:
        # (T,H,W) of raw indices; ignore 0 and >num_classes by default
        onehot_video = stack_indices_to_onehot_safe(
            segmented_video,
            valid_ids=range(1, num_classes + 1),
            out_num_classes=num_classes,
            id_map=None,           # supply a dict if you need to merge remap classes
            background_ids=(0,),   # ignore label 0
            dtype=np.uint8
        )
    elif segmented_video.ndim == 4:
        # Already (T,H,W,K)
        onehot_video = segmented_video
        if onehot_video.shape[-1] < num_classes:
            # You can either raise or pad; we’ll raise to avoid silent misuse
            raise ValueError(
                f"Provided one-hot has K={onehot_video.shape[-1]} < num_classes={num_classes}"
            )
    else:
        raise ValueError(f"Unexpected shape for input npy: {segmented_video.shape}")

    T, H, W, K = onehot_video.shape
    if not (1 <= class_index_original <= K):
        raise ValueError(f"class_index_original={class_index_original} not in 1..{K}")

    # Extract desired class channel (H,W) per frame: (T,H,W) 0/1
    mask_video = onehot_video[..., class_index_original - 1].astype(np.uint8)

    # Dummy RGB video (unused by BEV math, but keeps API consistent)
    dummy_video = np.zeros((T, H, W, 3), dtype=np.uint8)

    return video_to_bev_tensor(
        video_tensor=dummy_video,
        segmented_video=mask_video,     # (T,H,W) 0/1
        n_frames=n_frames,
        height_m=height_m,
        pitch_deg_down=pitch_deg_down,
        fx=fx,
        fy=fy,
        meters_per_pixel=meters_per_pixel,
        forward_range=forward_range,
        lateral_range=lateral_range,
        debug=debug
    )


# Process directly from file path:





def rotate_shift_sum(img1, img2, angle_deg=10.0, hor_shift_px=10):
    """
    Rotate img1 by +angle and shift LEFT, rotate img2 by -angle and shift RIGHT.
    Sum the two results on the smallest canvas that fits both, using 0-padding.

    Returns:
        output: np.ndarray of dtype float32 or uint8 (based on input),
                shape (H_out, W_out, C) or (H_out, W_out)
    """
    assert img1.shape == img2.shape, "Both images must be same size"
    H, W = img1.shape[:2]
    is_color = img1.ndim == 3
    C = img1.shape[2] if is_color else 1

    # Convert to float32 for accumulation
    img1_f = img1.astype(np.float32)
    img2_f = img2.astype(np.float32)

    # Rotation matrices around center
    center = (W / 2.0, H / 2.0)
    M1 = cv2.getRotationMatrix2D(center, +angle_deg, 1.0)
    M2 = cv2.getRotationMatrix2D(center, -angle_deg, 1.0)

    # Horizontal shift: img1 left, img2 right
    M1[0, 2] -= hor_shift_px
    M2[0, 2] += hor_shift_px

    # Compute canvas size
    def transform_corners(M):
        corners = np.array([[0,0],[W,0],[W,H],[0,H]], dtype=np.float32)
        ones = np.ones((4,1), dtype=np.float32)
        corners_h = np.hstack([corners, ones])
        transformed = (M @ corners_h.T).T
        return transformed

    corners1 = transform_corners(M1)
    corners2 = transform_corners(M2)
    all_corners = np.vstack([corners1, corners2])

    min_xy = np.floor(all_corners.min(axis=0)).astype(int)
    max_xy = np.ceil(all_corners.max(axis=0)).astype(int)
    out_W, out_H = (max_xy - min_xy)[0], (max_xy - min_xy)[1]

    shift_x, shift_y = -min_xy[0], -min_xy[1]
    M1[:, 2] += shift_x
    M2[:, 2] += shift_x
    M1[1, 2] += shift_y
    M2[1, 2] += shift_y

    # Warp both images
    dsize = (out_W, out_H)
    border_val = 0 if not is_color else (0,) * C
    warp1 = cv2.warpAffine(img1_f, M1, dsize, flags=cv2.INTER_LINEAR, borderValue=border_val)
    warp2 = cv2.warpAffine(img2_f, M2, dsize, flags=cv2.INTER_LINEAR, borderValue=border_val)

    # Sum the results
    summed = warp1 + warp2

    # Return in same dtype as input (clip if needed)
    if np.issubdtype(img1.dtype, np.integer):
        summed = np.clip(summed, 0, 255).astype(img1.dtype)
    return summed




def list_to_numpy_tensor(frames, axis=-1, dtype=None):
    """
    Convert a list of NumPy 2D arrays into a single 3D array (a 'tensor').
    axis=-1  -> shape (H, W, N)   # channels-last style
    axis=0   -> shape (N, H, W)   # batch-first style
    """
    if not frames:
        raise ValueError("frames list is empty")

    base_shape = frames[0].shape
    for i, f in enumerate(frames):
        if f.shape != base_shape:
            raise ValueError(f"All frames must have the same shape: "
                             f"frame 0 is {base_shape}, frame {i} is {f.shape}")
    if dtype is None:
        dtype = np.result_type(*[np.asarray(f).dtype for f in frames])

    stacked = np.stack([np.asarray(f, dtype=dtype) for f in frames], axis=axis)
    return stacked





current_image_path = "1754410876"

#"/workspace/test_results/cam2_1754590241.npy

path = "/workspace/test_results/cam1_"+current_image_path+".npy"
bev_tensor1, projector1, grid1 = npy_to_bev_tensor(
    path, 
    n_frames=300, 
    debug=True,     
    pitch_deg_down=0.0,
    height_m=1,
    fx=300,
    fy=int(530/380*200),
    meters_per_pixel=0.20,
    forward_range=(0.5, 30.0),
    lateral_range=(-20.0, 20.0),
)
path = "/workspace/test_results/cam2_"+current_image_path+".npy"
bev_tensor2, projector2, grid2 = npy_to_bev_tensor(
    path, 
    n_frames=300, 
    debug=True,     
    pitch_deg_down=0.0,
    height_m=1,
    fx=380.0,
    fy=530,
    meters_per_pixel=0.20,
    forward_range=(0.5, 30.0),
    lateral_range=(-20.0, 20.0),
)
bev_cal_tensor = 0
# Shape: (H_bev, W_bev, N)
fps = 12  # Set your desired frame rate
N = 300  # Number of frames

# Generate one combined frame to determine size
# Create the video writer

# Write each frame
combined_list = []
for i in range(N):
    combined = rotate_shift_sum(np.int8(bev_tensor2[:, :, i] / 255), np.int8(bev_tensor1[:, :, i] / 255), angle_deg = -35, hor_shift_px=-40)
    combined_list.append(combined)


tensor_hwn = list_to_numpy_tensor(combined_list, axis=-1, dtype=np.uint8)
import numpy as np
import cv2

def dilate_erode_blur_slice(
    image: np.ndarray, 
    kernel_size: int = 5,     
    blur_ksize: int = 7,
    blur_sigma: float = 1.0
    ) -> np.ndarray:
    """
    Performs dilation followed by erosion (closing) on the slice tensor_hwn[:, :, 50].

    Parameters:
        tensor_hwn (np.ndarray): Input tensor of shape (H, W, N)
        kernel_size (int): Size of the square kernel (default: 5)

    Returns:
        np.ndarray: Closed (dilated + eroded) version of slice [:, :, 50]
    """

    img = image.astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))

    
    closed = cv2.erode(img, kernel)
    dilated = cv2.dilate(closed, kernel)

    
    blurred = cv2.GaussianBlur(dilated, (blur_ksize, blur_ksize), blur_sigma)

    

    return blurred

denoised = dilate_erode_blur_slice(tensor_hwn, kernel_size=4, blur_sigma=6, blur_ksize=11)
#help()
plt.imshow(dilate_erode_blur_slice(tensor_hwn, kernel_size=4, blur_sigma=6, blur_ksize=11)[:, :, 10])

def bev_project_onehot_video(
    onehot_video: np.ndarray,        # (T, H, W, K), values in {0,1} or {0,255}
    *,
    height_m: float = 1.4,
    pitch_deg_down: float = 6.0,
    roll_deg: float = 0.0,
    yaw_deg: float = 0.0,
    K0=None,
    K0_size=None,
    fx: float = 1100.0,
    fy: float | None = None,
    meters_per_pixel: float = 0.10,
    forward_range: tuple[float, float] = (0.5, 40.0),
    lateral_range: tuple[float, float] = (-8.0, 8.0),
    binarize_threshold: int = 127,   # used if input channels are 0/255
) -> tuple[np.ndarray, BEVProjector, BEVGrid]:
    """
    Returns (bev_onehot, projector, grid) where bev_onehot is (T, H_bev, W_bev, K) in {0,1}.
    """
    assert onehot_video.ndim == 4, f"expected (T,H,W,K), got {onehot_video.shape}"
    T, H, W, K = onehot_video.shape

    # Build intrinsics
    Kcam = _build_intrinsics(W, H, K0=K0, K0_size=K0_size, fx=fx, fy=fy)

    # Pose (convert degrees to radians; pitch "down" is negative)
    roll  = np.deg2rad(roll_deg)
    pitch = -np.deg2rad(pitch_deg_down)
    yaw   = np.deg2rad(yaw_deg)

    # Projector & grid
    projector = BEVProjector().set_intrinsics(Kcam).set_pose(
        height_m=height_m, roll=roll, pitch=pitch, yaw=yaw
    )
    grid = BEVGrid(
        meters_per_pixel=meters_per_pixel,
        forward_range_m=forward_range,
        lateral_range_m=lateral_range,
    )

    # Precompute a single warp matrix M for SRC->DST to reuse for all frames/channels
    H_pg = projector._ground_to_image_homography()
    S = grid.scaling_matrix()
    M = np.linalg.inv(S) @ np.linalg.inv(H_pg)  # p_dst = M * p_img

    W_out, H_out = grid.size
    bev = np.zeros((T, H_out, W_out, K), dtype=np.uint8)

    # Project every class channel for every frame
    for t in range(T):
        frame_oh = onehot_video[t]                   # (H,W,K)
        # Ensure channel data are 0/255 uint8 for cv2.warpPerspective NN
        # (we'll convert back to 0/1 at the end)
        if frame_oh.dtype != np.uint8:
            frame_oh_u8 = (frame_oh.astype(np.uint8) * 255)
        else:
            # if already 0/1, scale to 0/255 to be robust to NN warps
            mx = frame_oh.max()
            frame_oh_u8 = frame_oh * (255 if mx <= 1 else 1)

        for k in range(K):
            src = frame_oh_u8[..., k]
            dst = cv2.warpPerspective(
                src, M, (W_out, H_out),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT, borderValue=0
            )
            # back to 0/1
            bev[t, ..., k] = (dst > binarize_threshold).astype(np.uint8)

    return bev, projector, grid


def npy_to_bev_onehot(
    path: str,
    n_frames: int | None = None,
    *,
    pitch_deg_down: float = 6.0,
    height_m: float = 1.4,
    fx: float = 1100.0,
    fy: float = 530.0,
    meters_per_pixel: float = 0.10,
    forward_range: tuple[float, float] = (0.5, 40.0),
    lateral_range: tuple[float, float] = (-8.0, 8.0),
    num_classes: int = 12,
) -> tuple[np.ndarray, BEVProjector, BEVGrid]:
    """
    Loads labels from `.npy`:
      - If shape (T,H,W) with raw ids: converts robustly to one-hot (T,H,W,K)
      - If shape (T,H,W,K): uses as-is (validates K)
    Returns BEV one-hot video (T, H_bev, W_bev, K) with all classes.
    """
    assert os.path.exists(path), f"File not found: {path}"
    arr = np.load(path)

    # to one-hot video (T,H,W,K)
    if arr.ndim == 3:
        onehot_video = stack_indices_to_onehot_safe(
            arr,
            valid_ids=range(1, num_classes + 1),
            out_num_classes=num_classes,
            id_map=None,
            background_ids=(0,),
            dtype=np.uint8,
        )
    elif arr.ndim == 4:
        if arr.shape[-1] < num_classes:
            raise ValueError(f"input K={arr.shape[-1]} < num_classes={num_classes}")
        onehot_video = arr.astype(np.uint8)
    else:
        raise ValueError(f"Unexpected array shape: {arr.shape}")

    T = onehot_video.shape[0]
    if n_frames is not None:
        T = min(T, int(n_frames))
        onehot_video = onehot_video[:T]

    bev_oh, projector, grid = bev_project_onehot_video(
        onehot_video,
        height_m=height_m,
        pitch_deg_down=pitch_deg_down,
        roll_deg=0.0,
        yaw_deg=0.0,
        fx=fx,
        fy=fy,
        meters_per_pixel=meters_per_pixel,
        forward_range=forward_range,
        lateral_range=lateral_range,
    )
    return bev_oh, projector, grid   # (T, H_bev, W_bev, K)

# --- Example usage for two cameras (returns 4-D) ---

path1 = f"/workspace/test_results/cam1_{current_image_path}.npy"
path2 = f"/workspace/test_results/cam2_{current_image_path}.npy"

bev1, proj1, grid1 = npy_to_bev_onehot(
    path1, n_frames=300,
    pitch_deg_down=0.0, height_m=1.0, fx=300, fy=int(530/380*200),
    meters_per_pixel=0.20, forward_range=(0.5, 30.0), lateral_range=(-20.0, 20.0),
)   # bev1: (T, H_bev, W_bev, K)

bev2, proj2, grid2 = npy_to_bev_onehot(
    path2, n_frames=300,
    pitch_deg_down=0.0, height_m=1.0, fx=380.0, fy=530,
    meters_per_pixel=0.20, forward_range=(0.5, 30.0), lateral_range=(-20.0, 20.0),
)   # bev2: (T, H_bev, W_bev, K)

# Combine both cameras per class

def rotate_shift_sum_binary(mask_left: np.ndarray, mask_right: np.ndarray,
                            angle_deg: float, hor_shift_px: int) -> np.ndarray:
    # masks are 0/1 uint8, convert to 0/255 for warp sum, then back to 0/1
    out = rotate_shift_sum((mask_left*255).astype(np.uint8),
                           (mask_right*255).astype(np.uint8),
                           angle_deg=angle_deg, hor_shift_px=hor_shift_px)
    return (out > 127).astype(np.uint8)

T, Hb, Wb, K = bev1.shape
combined = np.zeros_like(bev1, dtype=np.uint8)  # (T,Hb,Wb,K)
for t in range(T):
    for k in range(K):
        combined[t, ..., k] = rotate_shift_sum_binary(
            bev2[t, ..., k], bev1[t, ..., k],
            angle_deg=-35, hor_shift_px=-40
        )

plt.imshow(combined[0, ..., 0], cmap='gray')
plt.title("Combined BEV (first frame, first class)")
plt.axis("off")
plt.show()
