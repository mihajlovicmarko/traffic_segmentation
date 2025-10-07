#!/usr/bin/env python3
"""
OpenVINO anonymizer for faces & license plates (blur/pixelate).
- Works on images (.jpg/.png/...) and videos (.mp4/.avi/...).
- Assumes OpenVINO IR models that output detections in the common format:
  [image_id, label, confidence, x_min, y_min, x_max, y_max] with coords in [0,1].
  (e.g., Intel's face/vehicle-license-plate models converted to IR.)

Usage:
  python anonymize.py \
      --input /path/to/folder_or_file \
      --out   /path/to/output \
      --face-model /models/face-detection-0200.xml \
      --plate-model /models/license-plate-recognition-barrier-0007.xml \
      --device CPU --conf 0.5 --blur pixelate

Notes:
- Writing via OpenCV strips EXIF/metadata from images.
- This script does NOT preserve audio when anonymizing videos.
"""

import os
import sys
import argparse
import glob
from pathlib import Path
import cv2
import numpy as np

try:
    from openvino.runtime import Core
except Exception as e:
    print("ERROR: OpenVINO not available. Install openvino-runtime and try again.", file=sys.stderr)
    raise

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".wmv"}

# ------------------------------- Util -------------------------------

def is_image(p: Path) -> bool:
    return p.suffix.lower() in IMAGE_EXTS

def is_video(p: Path) -> bool:
    return p.suffix.lower() in VIDEO_EXTS

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def list_inputs(input_path: Path):
    if input_path.is_file():
        return [input_path]
    files = []
    for ext in list(IMAGE_EXTS | VIDEO_EXTS):
        files.extend(input_path.rglob(f"*{ext}"))
    return sorted(files)

# --------------------------- Blurring/Pixels -------------------------

def clamp_box(x0, y0, x1, y1, w, h):
    x0 = max(0, min(w - 1, int(x0)))
    y0 = max(0, min(h - 1, int(y0)))
    x1 = max(0, min(w,     int(x1)))
    y1 = max(0, min(h,     int(y1)))
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1

def blur_rect(img, x0, y0, x1, y1, method="gauss", strength=25):
    roi = img[y0:y1, x0:x1]
    if roi.size == 0:
        return img
    if method == "gauss":
        # kernel must be odd; scale roughly by box size & strength
        k = max(3, int(strength // 2 * 2 + 1))
        kx = max(3, ( (x1-x0)//6*2 + 1))
        ky = max(3, ( (y1-y0)//6*2 + 1))
        kx = max(k, kx | 1)
        ky = max(k, ky | 1)
        roi_blur = cv2.GaussianBlur(roi, (kx, ky), 0)
    else:
        # pixelate: downsample then upsample (blocky)
        px = max(4, min(32, max((x1-x0)//strength, 8)))
        py = max(4, min(32, max((y1-y0)//strength, 8)))
        small = cv2.resize(roi, (px, py), interpolation=cv2.INTER_LINEAR)
        roi_blur = cv2.resize(small, (x1-x0, y1-y0), interpolation=cv2.INTER_NEAREST)
    img[y0:y1, x0:x1] = roi_blur
    return img

def draw_box(img, x0, y0, x1, y1, text="", color=(0,255,0)):
    x0 = max(0, min(img.shape[1]-1, int(x0)))
    y0 = max(0, min(img.shape[0]-1, int(y0)))
    x1 = max(0, min(img.shape[1]-1, int(x1)))
    y1 = max(0, min(img.shape[0]-1, int(y1)))
    cv2.rectangle(img, (x0,y0), (x1,y1), color, 2)
    if text:
        cv2.putText(img, text, (x0, max(0,y0-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return img

# ---------------------------- OpenVINO IO ----------------------------

class OVDetector:
    def __init__(self, core: "Core", model_xml: str, device: str = "CPU", conf_thr: float = 0.5, coords_mode: str = "normalized"):
        self.conf_thr = conf_thr
        self.coords_mode = coords_mode
        self.model = core.read_model(model=model_xml)
        self.compiled = core.compile_model(self.model, device)
        self.input = self.compiled.inputs[0]
        shape = list(self.input.shape)
        self.layout = None
        self.is_dynamic = any(s <= 0 for s in shape)
        if not self.is_dynamic and len(shape) == 4:
            if shape[1] == 3:
                self.layout = "NCHW"
                self.H = int(shape[2]); self.W = int(shape[3])
            elif shape[3] == 3:
                self.layout = "NHWC"
                self.H = int(shape[1]); self.W = int(shape[2])
            else:
                self.layout = "NCHW"
                self.H = int(shape[2]); self.W = int(shape[3])
        else:
            self.layout = "NCHW"
            self.H = None; self.W = None
        self.output = self.compiled.outputs[0]
        shp = list(self.input.shape)
        if len(shp) == 4 and all(isinstance(s, (int, np.integer)) and s > 0 for s in shp):
            if shp[1] == 3:
                self.inH, self.inW = int(shp[2]), int(shp[3])
            elif shp[3] == 3:
                self.inH, self.inW = int(shp[1]), int(shp[2])
            else:
                self.inH = self.H or 640; self.inW = self.W or 640
        else:
            self.inH = self.H or 640; self.inW = self.W or 640

    def preprocess(self, frame):
        h, w = frame.shape[:2]
        H = self.H or 640
        W = self.W or 640
        resized = cv2.resize(frame, (W, H))  # OpenCV is BGR, that’s fine

        if self.layout == "NHWC":
            # Model expects [1,H,W,3] (channels last)
            blob = resized[None, :, :, :].astype(np.float32)
        else:
            # Model expects [1,3,H,W] (channels first)
            blob = resized.transpose(2, 0, 1)[None, ...].astype(np.float32)

        return blob, (w, h), (W, H)

    def infer(self, frame):
        blob, (w, h), (W, H) = self.preprocess(frame)
        res = self.compiled([blob])[self.output]
        dets = self._parse(res)  # (label, conf, x0, y0, x1, y1)
        # --- AUTO-NORMALIZE if detections look like pixels in model space ---
        need_norm = any((d[2] > 1.5 or d[3] > 1.5 or d[4] > 1.5 or d[5] > 1.5) for d in dets) if dets else False
        shp = list(self.input.shape)
        if len(shp) == 4 and all(int(s) > 0 for s in shp):
            if shp[1] == 3:
                inH, inW = int(shp[2]), int(shp[3])
            elif shp[3] == 3:
                inH, inW = int(shp[1]), int(shp[2])
            else:
                inH, inW = H, W
        else:
            inH, inW = H, W
        abs_dets = []
        for label, conf, x0, y0, x1, y1 in dets:
            if need_norm:
                x0, x1 = x0 / inW, x1 / inW
                y0, y1 = y0 / inH, y1 / inH
            abs_dets.append((label, conf, x0 * w, y0 * h, x1 * w, y1 * h))
        return abs_dets

    def _parse(self, out):
        """
        Parses DetectionOutput-like tensors into (conf, x0, y0, x1, y1) in [0,1].
        """
        out = np.array(out)
        if out.ndim == 4 and out.shape[1] == 1:
            out = out[0, 0, :, :]
        elif out.ndim == 3:
            out = out[0, :, :]
        elif out.ndim == 2:
            pass
        else:
            out = out.reshape(-1, 7)

        keep = []
        for det in out:
            if det.shape[-1] != 7:
                continue
            _, label, conf, x0, y0, x1, y1 = det.astype(np.float32)
            label = int(label)
            if conf >= self.conf_thr:
                x0 = float(np.clip(x0, 0, 1))
                y0 = float(np.clip(y0, 0, 1))
                x1 = float(np.clip(x1, 0, 1))
                y1 = float(np.clip(y1, 0, 1))
                if x1 > x0 and y1 > y0:
                    keep.append((label, float(conf), x0, y0, x1, y1))
        return keep

# --- Tiling and grow ---
def infer_plate_with_tiling(frame, plate_det, tiles=2, conf_thr=0.30):
    H, W = frame.shape[:2]
    boxes = []
    th = H // tiles
    tw = W // tiles
    for r in range(tiles):
        for c in range(tiles):
            y0 = r * th
            x0 = c * tw
            y1 = H if r == tiles-1 else (r+1) * th
            x1 = W if c == tiles-1 else (c+1) * tw
            crop = frame[y0:y1, x0:x1]
            dets = plate_det.infer(crop)
            for (lbl, conf, X0, Y0, X1, Y1) in dets:
                if conf >= conf_thr:
                    boxes.append((lbl, conf, X0 + x0, Y0 + y0, X1 + x1, Y1 + y1))
    return boxes

def grow(x0, y0, x1, y1, scale=1.1, w=0, h=0):
    cx, cy = x0 + (x1-x0)*scale /2, y0 + (y1-y0)*scale/2
    ww, hh = (x1-x0)*scale, (y1-y0)*scale
    return clamp_box(cx-ww/2, cy-hh/2, cx+ww/2, cy+hh/2, w, h)

# --- Anonymization ---
def pick_plate(label, plate_mode, plate_label_cfg):
    if plate_mode == "plates":
        return (label == plate_label_cfg)
    if plate_mode == "vehicles":
        return (label != plate_label_cfg)
    if plate_mode == "all":
        return True
    return True

def is_plate_shape(x0, y0, x1, y1):
    w = max(1.0, (x1 - x0)); h = max(1.0, (y1 - y0))
    ar = w / h
    return 1.5 <= ar <= 8.0 and h >= 6

def anonymize_frame(frame, face_det=None, plate_det=None, blur_method="gauss", strength=25,
                    face_conf=0.5, plate_conf=0.30, plate_label=2, tiles=2, enlarge=1.0, plate_mode="auto"):
    h, w = frame.shape[:2]
    boxes = []
    if face_det is not None:
        for (label, conf, x0, y0, x1, y1) in face_det.infer(frame):
            if conf >= face_conf:
                cb = grow(x0, y0, x1, y1, scale=enlarge, w=w, h=h) if enlarge > 1.0 else clamp_box(x0, y0, x1, y1, w, h)
                if cb:
                    boxes.append(cb)
    raw_plate = infer_plate_with_tiling(frame, plate_det, tiles=tiles, conf_thr=plate_conf) if plate_det else []
    picked = []
    for (lbl, conf, x0, y0, x1, y1) in raw_plate:
        if plate_label > 0 and lbl != plate_label:
            continue
        picked.append((x0, y0, x1, y1))
    for (x0, y0, x1, y1) in picked:
        cb = grow(x0, y0, x1, y1, scale=enlarge, w=w, h=h) 
        if cb:
            boxes.append(cb)
    out = frame.copy()
    for (x0, y0, x1, y1) in boxes:
        out = blur_rect(out, x0, y0, x1, y1, method=blur_method, strength=strength)
    return out, boxes

def anonymize_image(in_path: Path, out_path: Path, face_det, plate_det, blur_method, strength,
                    face_conf=0.5, plate_conf=0.30, plate_label=2, tiles=2, enlarge=1.0, plate_mode="auto", debug=False):
    img = cv2.imread(str(in_path), cv2.IMREAD_COLOR)
    if img is None:
        print(f"[WARN] Could not read image: {in_path}")
        return
    out, boxes = anonymize_frame(img, face_det, plate_det, blur_method, strength,
                         face_conf=face_conf, plate_conf=plate_conf, plate_label=plate_label, tiles=tiles, enlarge=enlarge, plate_mode=plate_mode)
    ensure_dir(out_path.parent)
    # Save with OpenCV -> strips EXIF/metadata
    ok = cv2.imwrite(str(out_path), out, [cv2.IMWRITE_JPEG_QUALITY, 95] if out_path.suffix.lower() in [".jpg",".jpeg"] else [])
    if not ok:
        print(f"[WARN] Failed to write: {out_path}")
    if debug:
        dbg = img.copy()
        if face_det:
            for (lbl, conf, x0, y0, x1, y1) in face_det.infer(img):
                dbg = draw_box(dbg, x0, y0, x1, y1, f"F:{conf:.2f}", (0,255,0))
        raw_full = plate_det.infer(img) if plate_det else []
        for (lbl, conf, x0, y0, x1, y1) in raw_full:
            dbg = draw_box(dbg, x0, y0, x1, y1, f"Praw{lbl}:{conf:.2f}", (255,0,0))
        for (x0, y0, x1, y1) in boxes:
            dbg = draw_box(dbg, x0, y0, x1, y1, "BLUR", (0,255,255))
        vis = np.concatenate([img, out, dbg], axis=1)
        scale = min(1.0, 1920.0 / vis.shape[1])
        if scale < 1.0:
            vis = cv2.resize(vis, (int(vis.shape[1]*scale), int(vis.shape[0]*scale)))
        ensure_dir(out_path.parent / "debug")
        cv2.imwrite(str(out_path.parent / "debug" / f"{in_path.stem}_debug.jpg"), vis)

def anonymize_video(in_path: Path, out_path: Path, face_det, plate_det, blur_method, strength,
                    face_conf=0.5, plate_conf=0.30, plate_label=2, tiles=2, enlarge=1.0, plate_mode="auto", debug=False):
    cap = cv2.VideoCapture(str(in_path))
    if not cap.isOpened():
        print(f"[WARN] Could not open video: {in_path}")
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v") if out_path.suffix.lower() in [".mp4", ".m4v"] else cv2.VideoWriter_fourcc(*"XVID")
    ensure_dir(out_path.parent)
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
    if not writer.isOpened():
        print(f"[WARN] Could not open writer: {out_path}")
        cap.release()
        return
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out, boxes = anonymize_frame(frame, face_det, plate_det, blur_method, strength,
                             face_conf=face_conf, plate_conf=plate_conf, plate_label=plate_label, tiles=tiles, enlarge=enlarge, plate_mode=plate_mode)
        writer.write(out)
        if debug:
            dbg = frame.copy()
            if face_det:
                for (lbl, conf, x0, y0, x1, y1) in face_det.infer(frame):
                    dbg = draw_box(dbg, x0, y0, x1, y1, f"F:{conf:.2f}", (0,255,0))
            raw_full = plate_det.infer(frame) if plate_det else []
            for (lbl, conf, x0, y0, x1, y1) in raw_full:
                dbg = draw_box(dbg, x0, y0, x1, y1, f"Praw{lbl}:{conf:.2f}", (255,0,0))
            for (x0, y0, x1, y1) in boxes:
                dbg = draw_box(dbg, x0, y0, x1, y1, "BLUR", (0,255,255))
            vis = np.concatenate([frame, out, dbg], axis=1)
            scale = min(1.0, 1920.0 / vis.shape[1])
            if scale < 1.0:
                vis = cv2.resize(vis, (int(vis.shape[1]*scale), int(vis.shape[0]*scale)))
            if idx % 100 == 0:
                ensure_dir(Path(out_path).parent / "debug")
                cv2.imwrite(str((Path(out_path).parent / "debug" / f"{Path(in_path).stem}_{idx:06d}.jpg")), vis)
        idx += 1
        if idx % 100 == 0:
            print(f"[INFO] {in_path.name}: processed {idx} frames...")
    cap.release()
    writer.release()

# ------------------------------ CLI ---------------------------------

def main():
    ap = argparse.ArgumentParser(description="OpenVINO-based anonymizer for faces & license plates")
    ap.add_argument("--input", required=True, help="Path to image/video file or directory")
    ap.add_argument("--out", required=True, help="Output file or directory")
    ap.add_argument("--face-model", type=str, default=None, help="Path to OpenVINO IR face detector .xml")
    ap.add_argument("--plate-model", type=str, default=None, help="Path to OpenVINO IR license plate detector .xml")
    ap.add_argument("--device", type=str, default="CPU", help="OpenVINO device (CPU|GPU|AUTO...)")
    ap.add_argument("--conf", type=float, default=0.5, help="Confidence threshold (legacy, use --face-conf/--plate-conf)")
    ap.add_argument("--blur", type=str, default="gauss", choices=["gauss","pixelate"], help="Anonymization method")
    ap.add_argument("--strength", type=int, default=25, help="Blur/pixelate strength")
    ap.add_argument("--face-conf", type=float, default=0.5, help="Face confidence threshold")
    ap.add_argument("--plate-conf", type=float, default=0.30, help="Plate confidence threshold")
    ap.add_argument("--plate-label", type=int, default=2, help="Plate label (e.g. 2 for 0106)")
    ap.add_argument("--tiles", type=int, default=2, help="Number of tiles for plate detection")
    ap.add_argument("--enlarge", type=float, default=1.0, help="Box enlarge scale (e.g. 1.1 for +10%)")
    ap.add_argument("--debug", action="store_true", help="Write debug overlays next to outputs")
    ap.add_argument("--plate-mode", choices=["auto","plates","vehicles","all"], default="auto", help="What to keep from plate detector")
    ap.add_argument("--coords", choices=["normalized","absolute"], default="normalized", help="How the model outputs box coords (DetectionOutput is usually 'normalized')")
    args = ap.parse_args()

    in_p  = Path(args.input)
    out_p = Path(args.out)

    core = Core()
    face_det = OVDetector(core, args.face_model, args.device, args.face_conf, coords_mode=args.coords) if args.face_model else None
    plate_det = OVDetector(core, args.plate_model, args.device, args.plate_conf, coords_mode=args.coords) if args.plate_model else None
    if not face_det and not plate_det:
        print("ERROR: Provide at least one of --face-model or --plate-model")
        sys.exit(2)

    inputs = list_inputs(in_p)
    if not inputs:
        print(f"ERROR: No supported files found under {in_p}")
        sys.exit(1)

    debug = args.debug
    plate_mode = args.plate_mode
    coords_mode = args.coords
    # If single input file AND out is a file path, keep that
    if len(inputs) == 1 and out_p.suffix:
        if is_image(inputs[0]):
            anonymize_image(inputs[0], out_p, face_det, plate_det, args.blur, args.strength,
                            face_conf=args.face_conf, plate_conf=args.plate_conf, plate_label=args.plate_label, tiles=args.tiles, enlarge=args.enlarge, plate_mode=plate_mode, debug=debug)
        elif is_video(inputs[0]):
            anonymize_video(inputs[0], out_p, face_det, plate_det, args.blur, args.strength,
                            face_conf=args.face_conf, plate_conf=args.plate_conf, plate_label=args.plate_label, tiles=args.tiles, enlarge=args.enlarge, plate_mode=plate_mode, debug=debug)
        else:
            print(f"[WARN] Unsupported file type: {inputs[0]}")
        return
    # Otherwise treat --out as a directory
    ensure_dir(out_p)
    for src in inputs:
        rel = src.name if in_p.is_file() else src.relative_to(in_p)
        dst = out_p / rel
        ensure_dir(dst.parent)
        # Determine output file path for images/videos
        if is_image(src):
            out_file = dst.with_suffix(".jpg") if dst.suffix.lower() in (".png",".bmp",".tif",".tiff",".webp") else dst
        elif is_video(src):
            out_file = dst.with_suffix(".mp4")
        else:
            print(f"[SKIP] Unsupported file: {src}")
            continue
        # Skip if output file already exists
        if out_file.exists():
            print(f"[SKIP] Already processed: {out_file.name}")
            continue
        if is_image(src):
            anonymize_image(src, out_file, face_det, plate_det, args.blur, args.strength,
                            face_conf=args.face_conf, plate_conf=args.plate_conf, plate_label=args.plate_label, tiles=args.tiles, enlarge=args.enlarge, plate_mode=plate_mode, debug=debug)
        elif is_video(src):
            anonymize_video(src, out_file, face_det, plate_det, args.blur, args.strength,
                            face_conf=args.face_conf, plate_conf=args.plate_conf, plate_label=args.plate_label, tiles=args.tiles, enlarge=args.enlarge, plate_mode=plate_mode, debug=debug)
    print("[DONE] Anonymization complete.")

if __name__ == "__main__":
    main()
