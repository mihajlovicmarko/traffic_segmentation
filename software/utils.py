import numpy as np
import cv2
from scipy.ndimage import gaussian_filter1d
from dataclasses import dataclass
import logging

from typing import Optional, Tuple
# ------------------ BEV utils ------------------

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


def make_projector_and_grid(
    frame_shape: Tuple[int, int],          # (H, W) of your input npy frame
    *,
    fx: float = 1100.0,
    fy: Optional[float] = None,
    roll_deg: float = 0.0,
    pitch_deg_down: float = 6.0,
    yaw_deg: float = 0.0,
    height_m: float = 1.4,
    meters_per_pixel: float = 0.10,
    forward_range: Tuple[float, float] = (0.5, 40.0),
    lateral_range: Tuple[float, float] = (-8.0, 8.0),
) -> tuple:
    """Build and return (projector, grid) once, then reuse them for all frames."""
    H, W = frame_shape
    fy_loc = fx if fy is None else fy
    K = np.array([[fx,    0.0,  W / 2.0],
                  [0.0,  fy_loc, H / 2.0],
                  [0.0,   0.0,   1.0   ]], dtype=np.float64)

    roll  = np.deg2rad(roll_deg)
    pitch = -np.deg2rad(pitch_deg_down)   # down is negative in this convention
    yaw   = np.deg2rad(yaw_deg)

    projector = BEVProjector().set_intrinsics(K).set_pose(
        height_m=height_m, roll=roll, pitch=pitch, yaw=yaw
    )
    grid = BEVGrid(
        meters_per_pixel=meters_per_pixel,
        forward_range_m=forward_range,
        lateral_range_m=lateral_range
    )
    return projector, grid

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




def decreasing_positive_quadratic(n: int, k: float = 0.6) -> np.ndarray:
    """
    Center-heavy bias in [~(1-k), 1]. k in [0,1]. Peak at center.
    """
    t = np.linspace(-1.0, 1.0, max(1, int(n)), dtype=np.float32)
    bias = 1.0 - k * (t ** 2)
    return np.clip(bias, 0.0, 1.0)

def rotate_shift_sum(img1, img2, angle_deg=-35.0, hor_shift_px=-40):
    """
    Rotate img1 by +angle and shift LEFT (negative), rotate img2 by -angle and shift RIGHT (positive).
    Sum results on a canvas that fits both.
    Inputs can be 0/1 or 0..255 (H,W) or (H,W,1/3).
    Returns: float32 or uint8, shape (H_out, W_out)
    """
    assert img1.shape == img2.shape, "Frames must have same size"
    H, W = img1.shape[:2]
    is_color = (img1.ndim == 3)
    C = img1.shape[2] if is_color else 1

    img1_f = img1.astype(np.float32)
    img2_f = img2.astype(np.float32)

    center = (W / 2.0, H / 2.0)
    M1 = cv2.getRotationMatrix2D(center, +angle_deg, 1.0)
    M2 = cv2.getRotationMatrix2D(center, -angle_deg, 1.0)

    M1[0, 2] += hor_shift_px  # left if negative
    M2[0, 2] -= hor_shift_px  # right if negative above

    def corners_after(M):
        cs = np.array([[0,0],[W,0],[W,H],[0,H]], dtype=np.float32)
        ones = np.ones((4,1), dtype=np.float32)
        cs_h = np.hstack([cs, ones])
        return (M @ cs_h.T).T

    c1 = corners_after(M1)
    c2 = corners_after(M2)
    allc = np.vstack([c1, c2])
    mn = np.floor(allc.min(axis=0)).astype(int)
    mx = np.ceil(allc.max(axis=0)).astype(int)
    out_W, out_H = (mx - mn)[0], (mx - mn)[1]

    shift_x, shift_y = -mn[0], -mn[1]
    M1[:, 2] += shift_x; M2[:, 2] += shift_x
    M1[1, 2] += shift_y; M2[1, 2] += shift_y

    dsize = (out_W, out_H)
    border_val = 0 if not is_color else (0,) * C
    w1 = cv2.warpAffine(img1_f, M1, dsize, flags=cv2.INTER_LINEAR, borderValue=border_val)
    w2 = cv2.warpAffine(img2_f, M2, dsize, flags=cv2.INTER_LINEAR, borderValue=border_val)
    S = w1 + w2
    return np.clip(S, 0, 255).astype(img1.dtype) if np.issubdtype(img1.dtype, np.integer) else S

def denoise_frame(frame_01: np.ndarray, kernel_size=4, blur_ksize=11, blur_sigma=6.0) -> np.ndarray:
    """
    Light morph + blur on a single 2D binary-ish frame (0/1 or 0/255). Returns uint8 in {0,255}.
    (Uses erode->dilate then Gaussian blur.)
    """
    img = (frame_01 > 0).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    er = cv2.erode(img, k)
    di = cv2.dilate(er, k)
    bl = cv2.GaussianBlur(di, (blur_ksize, blur_ksize), blur_sigma)
    return bl


def npy_frame_to_bev(
    seg_frame: np.ndarray,                 # (H, W) labels or binary
    projector: "BEVProjector",
    grid: "BEVGrid",
    *,
    road_label: int = 0
) -> np.ndarray:
    """
    Use your BEVProjector/BEVGrid to warp a single npy frame to BEV.
    Returns uint8 BEV mask (H_out, W_out) with values {0,255}.
    """
    # Build 1/0 mask (road=1). If already binary uint8, just threshold.
    if seg_frame.dtype == np.uint8 and seg_frame.max() in (1, 255):
        mask01 = (seg_frame > 0).astype(np.uint8)
    else:
        mask01 = (seg_frame == road_label).astype(np.uint8)

    bev = projector.warp_calibrated(mask01=mask01, grid=grid)  # returns uint8 {0,255}
    return bev


def _group_contiguous(idxs: np.ndarray):
    if idxs.size == 0: return []
    idxs = np.sort(idxs)
    groups = []
    start = prev = idxs[0]
    for x in idxs[1:]:
        if x == prev + 1:
            prev = x
        else:
            groups.append(np.arange(start, prev + 1))
            start = prev = x
    groups.append(np.arange(start, prev + 1))
    return groups

def _resample_scores(scores: np.ndarray, out_len: int) -> np.ndarray:
    """Linear resample scores from len M_road to M_small to build a same-length bias."""
    M = len(scores)
    if M == out_len: return scores.copy()
    x_in = np.linspace(0, M - 1, M, dtype=np.float32)
    x_out = np.linspace(0, M - 1, out_len, dtype=np.float32)
    return np.interp(x_out, x_in, scores).astype(np.float32)

# ------------------ streaming state ------------------

@dataclass
class ProcessingState:
    """Carry temporal state between calls."""
    sel_last_road: np.ndarray | None = None   # (M_road,) one-hot smoothed memory
    last_best_road_idx: int | None = None

# ------------------ per-frame processor ------------------

def process_two_bev_frames(
    cam1_bev: np.ndarray,                 # (H, W) 0/1 or 0/255
    cam2_bev: np.ndarray,                 # (H, W) same shape as cam1_bev
    flower_tensor_road: np.ndarray,       # (H, W, M_road) uint8/bool masks
    flower_tensor_collision: np.ndarray,  # (H, W, M_small)
    state: Optional[ProcessingState] = None,
    *,
    # combine params:
    combine_angle_deg: float = -35.0,
    combine_hshift_px: int = -40,
    # scoring/smoothing:
    centre_bias_coefficient: float = 0.6,
    sigma_scores: float = 4.0,
    remembrance: float = 8.0,
    residual_sigma: float = 16.0,
    convergent_with_road: float = 5.0,
    # thresholds:
    road_min_abs: float = 40.0,
    road_min_frac_of_max: float = 0.10,
    collision_min_abs: float = 300.0,
    collision_min_frac_of_max: float = 0.30,
    small_top_k: int = 1,
    # visualization colors:
    road_color=(100, 255, 100),
    selected_road_color=(0, 255, 255),
    small_color=(0, 0, 255),
    # denoise params:
    morph_kernel: int = 4,
    blur_ksize: int = 11,
    blur_sigma: float = 6.0
):
    """
    Returns:
      viz_img:     (Hc, Wc, 3) uint8 visualization for this frame
      denoised_u8: (Hc, Wc)    uint8 denoised (morph+blur) image of the combined frame
      new_state:   ProcessingState for next call
      result:      dict with fields (best_road_idx, best_small_idxs, kept_indices, combined_frame, road_scores, small_scores_biased)
    """
    H, W, M_road = flower_tensor_road.shape
    _, _, M_small = flower_tensor_collision.shape

    # 1) Combine two BEV frames into one canvas
    f1 = (cam1_bev > 0).astype(np.uint8)
    f2 = (cam2_bev > 0).astype(np.uint8)
    combined = rotate_shift_sum(f2, f1, angle_deg=combine_angle_deg, hor_shift_px=combine_hshift_px)
    combined01 = (combined > 0).astype(np.uint8)

    # 2) Denoise
    den = denoise_frame(combined01, kernel_size=morph_kernel, blur_ksize=blur_ksize, blur_sigma=blur_sigma)
    denoised_u8 = den  # <- will be returned
    frame = (den > 0).astype(np.float32)  # binary for scoring

    # 3) Prepare/restore state
    if state is None or state.sel_last_road is None or len(state.sel_last_road) != M_road:
        sel_last = np.zeros((M_road,), dtype=np.float32)
    else:
        sel_last = state.sel_last_road.astype(np.float32)

    # 4) ROAD scores
    road_scores = np.sum(frame[:, :, None] * flower_tensor_road.astype(np.float32), axis=(0, 1))  # (M_road,)
    road_scores *= decreasing_positive_quadratic(M_road, centre_bias_coefficient).astype(np.float32)
    road_scores = gaussian_filter1d(road_scores, sigma=sigma_scores)

    if remembrance > 0.0:
        residual_smoothed = gaussian_filter1d(sel_last, sigma=residual_sigma)
        road_scores = road_scores * (1.0 + remembrance * residual_smoothed)

    best_road_idx = int(np.argmax(road_scores))
    road_gate = max(road_min_abs, road_min_frac_of_max * (float(road_scores.max()) if road_scores.size else 0.0))
    draw_road = bool(road_scores[best_road_idx] >= road_gate)

    # update memory (one-hot)
    sel_last[:] = 0.0
    sel_last[best_road_idx] = 1.0

    # 5) COLLISION scores (independent) + bias toward road scores resampled to small grid
    small_scores = np.sum(frame[:, :, None] * flower_tensor_collision.astype(np.float32), axis=(0, 1))  # (M_small,)
    small_scores = gaussian_filter1d(small_scores, sigma=sigma_scores)
    road_scores_small = _resample_scores(road_scores, M_small)
    bias = 1.0 + (road_scores_small / (road_scores_small.sum() + 1e-8)) * convergent_with_road
    small_scores_biased = small_scores * bias

    coll_gate = max(collision_min_abs, collision_min_frac_of_max * (float(small_scores_biased.max()) if small_scores_biased.size else 0.0))
    valid_small = np.where(small_scores_biased >= coll_gate)[0]

    if valid_small.size > 0:
        k = int(max(1, min(small_top_k, valid_small.size)))
        valid_scores = small_scores_biased[valid_small]
        sel = np.argpartition(-valid_scores, kth=k-1)[:k]
        best_small_idxs = valid_small[sel]
        best_small_idxs = best_small_idxs[np.argsort(-small_scores_biased[best_small_idxs])]
    else:
        best_small_idxs = np.array([], dtype=int)

    # 6) Visualization (groups -> strongest angle, intensity ∝ group size)
    kept_indices = np.where(road_scores >= road_gate)[0]
    viz_img = np.zeros((*frame.shape, 3), dtype=np.uint8)

    if kept_indices.size > 0:
        groups = _group_contiguous(kept_indices)
        max_len = max(len(g) for g in groups) if groups else 1
        base_col = np.array(road_color, dtype=np.float32)
        for g in groups:
            g = np.array(g, dtype=int)
            g_best = g[np.argmax(road_scores[g])]
            best_mask = flower_tensor_road[:, :, g_best].astype(bool)
            factor = 0.5 + 0.5 * (len(g) / max_len)  # [0.5, 1.0]
            col = np.clip(np.round(base_col * factor), 0, 255).astype(np.uint8)
            viz_img[best_mask] = col

    if draw_road:
        sel_mask = flower_tensor_road[:, :, best_road_idx].astype(bool)
        viz_img[sel_mask] = np.array(selected_road_color, dtype=np.uint8)

    if best_small_idxs.size > 0:
        if best_small_idxs.size == 1:
            small_mask = flower_tensor_collision[:, :, best_small_idxs[0]].astype(bool)
            viz_img[small_mask] = np.array(small_color, dtype=np.uint8)
        else:
            small_mask_union = np.any(flower_tensor_collision[:, :, best_small_idxs].astype(bool), axis=2)
            viz_img[small_mask_union] = np.array(small_color, dtype=np.uint8)

    # 7) Pack results & state
    new_state = ProcessingState(sel_last_road=sel_last.copy(), last_best_road_idx=best_road_idx)
    result = {
        "best_road_idx": best_road_idx,
        "best_small_idxs": best_small_idxs,
        "kept_indices": kept_indices,
        "combined_frame": combined01,          # 0/1 combined
        "road_scores": road_scores,
        "small_scores_biased": small_scores_biased
    }

    # NEW: return denoised image as the second item
    return viz_img, denoised_u8, new_state, result


def combine_viz_and_denoised(viz_img: np.ndarray, denoised_frame: np.ndarray, alpha_viz: float = 0.85) -> np.ndarray:
    """
    Alpha-blend viz over denoised background only where viz has content.
    viz_img:       (H, W, 3) uint8
    denoised_frame:(H, W)    uint8
    returns:       (H, W, 3) uint8
    """
    bg = cv2.cvtColor(denoised_frame, cv2.COLOR_GRAY2BGR)
    # mask where the viz has any color
    mask = (viz_img.sum(axis=2) > 0)
    out = bg.copy()
    if mask.any():
        out[mask] = cv2.addWeighted(bg[mask], 1.0 - alpha_viz, viz_img[mask], alpha_viz, 0.0)
    return out
