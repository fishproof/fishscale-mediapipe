"""
FishScale.AI — MediaPipe Face Landmark Microservice

A lightweight Flask service that accepts an image (URL or base64) and returns
facial landmark pixel coordinates for human reference scaling.

Deployed as a standalone cloud service (Railway.app) and called by the
FishScale.AI production Node.js server via HTTP.

Endpoints:
  POST /detect  — Detect face landmarks in an image (requires API key)
  GET  /health  — Health check (public)

Returns key measurement pairs for px/inch calibration:
  - bizygomatic (temple-to-temple): landmarks 234 <-> 454
  - biocular (outer eye corners): landmarks 33 <-> 263
  - ipd_inner (inner eye corners): landmarks 133 <-> 362
  - ipd_iris (iris centers): landmarks 473 <-> 468
  - face_height (forehead to chin): landmarks 10 <-> 152

Multi-scale detection:
  MediaPipe struggles when the face occupies <2% of image area. In typical fishing
  photos (1080x1920), a face is ~150px wide = ~1.6% area. We solve this by:
  1. Trying full image first
  2. Trying progressively tighter crops of the upper image
  3. Upscaling small crops 2x to boost face pixel count
  All coordinates are mapped back to original image space.

Authentication:
  Set MEDIAPIPE_API_KEY env var. Requests must include:
    Authorization: Bearer <key>
  The /health endpoint is public (no auth required).
"""

import os
import sys
import math
import time
import base64
import tempfile
import logging
import functools
from io import BytesIO
from typing import Optional, Tuple, List

import numpy as np
import requests as http_requests
from flask import Flask, request, jsonify
from PIL import Image as PILImage

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# ─── Configuration ─────────────────────────────────────────────────────────

PORT = int(os.environ.get("PORT", os.environ.get("MEDIAPIPE_PORT", 8080)))
MODEL_PATH = os.path.join(os.path.dirname(__file__), "face_landmarker.task")
API_KEY = os.environ.get("MEDIAPIPE_API_KEY", "")

# Maximum dimension for crops sent to MediaPipe (larger = slower, smaller = miss big faces)
MAX_CROP_DIMENSION = 1050

# Minimum upscale target: if a crop is smaller than this, upscale it
MIN_UPSCALE_TARGET = 600

# ─── Logging ───────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="[MediaPipe] %(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("mediapipe-service")

# ─── Landmark index pairs for measurements ─────────────────────────────────

MEASUREMENTS = {
    "bizygomatic": (234, 454),       # Temple-to-temple (~5.5" adult avg)
    "biocular": (33, 263),           # Outer eye corners (~3.5-3.7")
    "ipd_inner": (133, 362),         # Inner eye corners (~2.5")
    "ipd_iris": (473, 468),          # Iris center-to-center (~2.5")
    "face_height": (10, 152),        # Forehead to chin (~7-8")
}

# Additional individual landmarks to return
KEY_LANDMARKS = {
    "right_eye_outer": 33,
    "right_eye_inner": 133,
    "left_eye_inner": 362,
    "left_eye_outer": 263,
    "right_iris_center": 473,
    "left_iris_center": 468,
    "nose_tip": 1,
    "chin": 152,
    "forehead_top": 10,
    "right_temple": 234,
    "left_temple": 454,
    "midway_between_eyes": 168,
    "right_cheek": 205,
    "left_cheek": 425,
}

# ─── Multi-scale crop strategies ──────────────────────────────────────────
# Each tuple: (x_start_pct, y_start_pct, x_end_pct, y_end_pct, description)
# Ordered from largest to smallest — we want the biggest crop that still detects.
CROP_STRATEGIES = [
    # Tier 1: Large crops (for bigger faces or closer photos)
    (0.0, 0.0, 1.0, 0.55, "upper 55%"),
    (0.0, 0.0, 1.0, 0.45, "upper 45%"),
    
    # Tier 2: Medium crops (for typical fishing photos)
    (0.10, 0.0, 0.90, 0.45, "center-upper 80%w"),
    (0.0, 0.0, 0.60, 0.45, "left-upper"),
    (0.40, 0.0, 1.0, 0.45, "right-upper"),
    
    # Tier 3: Tight crops (for small faces in large images)
    (0.15, 0.0, 0.85, 0.35, "tight center"),
    (0.0, 0.0, 0.50, 0.35, "tight left"),
    (0.50, 0.0, 1.0, 0.35, "tight right"),
    
    # Tier 4: Very tight crops with upscaling (last resort)
    (0.20, 0.02, 0.80, 0.30, "very tight center"),
    (0.05, 0.02, 0.55, 0.30, "very tight left"),
    (0.45, 0.02, 0.95, 0.30, "very tight right"),
]

# ─── Initialize MediaPipe ──────────────────────────────────────────────────

log.info(f"Loading face landmarker model from {MODEL_PATH}")
if not os.path.exists(MODEL_PATH):
    log.error(f"Model file not found: {MODEL_PATH}")
    sys.exit(1)

_options = mp_vision.FaceLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=mp_vision.RunningMode.IMAGE,
    num_faces=1,
    min_face_detection_confidence=0.3,
    min_face_presence_confidence=0.3,
)
_landmarker = mp_vision.FaceLandmarker.create_from_options(_options)
log.info("Face landmarker model loaded successfully")

# ─── Flask App ─────────────────────────────────────────────────────────────

app = Flask(__name__)


# ─── Auth Middleware ───────────────────────────────────────────────────────

def require_api_key(f):
    """Decorator to require API key for protected endpoints."""
    @functools.wraps(f)
    def decorated(*args, **kwargs):
        if not API_KEY:
            # No API key configured — allow all requests (dev mode)
            return f(*args, **kwargs)
        
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header[7:]
        else:
            token = auth_header
        
        if token != API_KEY:
            return jsonify({"success": False, "error": "Unauthorized — invalid API key"}), 401
        
        return f(*args, **kwargs)
    return decorated


# ─── Helper Functions ──────────────────────────────────────────────────────

def _download_image_bytes(image_url: Optional[str], image_base64: Optional[str]) -> Optional[bytes]:
    """Download image data from URL or decode from base64."""
    try:
        if image_url:
            resp = http_requests.get(image_url, timeout=15)
            resp.raise_for_status()
            return resp.content
        elif image_base64:
            if "," in image_base64:
                image_base64 = image_base64.split(",", 1)[1]
            return base64.b64decode(image_base64)
        return None
    except Exception as e:
        log.error(f"Failed to download image: {e}")
        return None


def _pil_to_mp_image(pil_img: PILImage.Image) -> Optional[mp.Image]:
    """Convert a PIL Image to a MediaPipe Image via temp file."""
    try:
        buf = BytesIO()
        pil_img.save(buf, format="JPEG", quality=90)
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(buf.getvalue())
            tmp_path = f.name
        mp_image = mp.Image.create_from_file(tmp_path)
        os.unlink(tmp_path)
        return mp_image
    except Exception as e:
        log.error(f"Failed to convert PIL to MediaPipe image: {e}")
        return None


def _bytes_to_mp_image(img_data: bytes) -> Optional[mp.Image]:
    """Convert raw image bytes to a MediaPipe Image."""
    try:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(img_data)
            tmp_path = f.name
        mp_image = mp.Image.create_from_file(tmp_path)
        os.unlink(tmp_path)
        return mp_image
    except Exception as e:
        log.error(f"Failed to convert bytes to MediaPipe image: {e}")
        return None


def _detect_face(mp_image: mp.Image):
    """Run face detection on a MediaPipe image. Returns result or None."""
    try:
        result = _landmarker.detect(mp_image)
        if result.face_landmarks and len(result.face_landmarks) > 0:
            return result
        return None
    except Exception as e:
        log.error(f"Face detection error: {e}")
        return None


def _distance(p1: tuple, p2: tuple) -> float:
    """Euclidean distance between two (x, y) points."""
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def _build_response(result, orig_width: int, orig_height: int,
                     crop_offset_x: int = 0, crop_offset_y: int = 0,
                     detect_width: int = 0, detect_height: int = 0,
                     scale_factor: float = 1.0,
                     crop_strategy: str = "full", start_time: float = 0):
    """
    Build the JSON response from a successful face detection result.
    
    All coordinates are mapped back to the original image space:
    1. MediaPipe returns normalized coords (0-1) relative to the detection image
    2. Multiply by detect_width/height to get pixel coords in the detection image
    3. Divide by scale_factor to undo any upscaling
    4. Add crop_offset to map back to original image
    5. Divide by orig_width/height for percentage coords
    """
    face = result.face_landmarks[0]
    num_landmarks = len(face)
    
    # If detect dimensions not set, use original
    if detect_width == 0:
        detect_width = orig_width
    if detect_height == 0:
        detect_height = orig_height
    
    def lm_px(idx: int) -> tuple:
        """Get pixel coordinates in ORIGINAL image space."""
        lm = face[idx]
        # Step 1: Normalized -> detection image pixels
        det_x = lm.x * detect_width
        det_y = lm.y * detect_height
        # Step 2: Undo upscaling
        crop_x = det_x / scale_factor
        crop_y = det_y / scale_factor
        # Step 3: Add crop offset
        orig_x = crop_x + crop_offset_x
        orig_y = crop_y + crop_offset_y
        return (orig_x, orig_y)
    
    def lm_pct(idx: int) -> dict:
        """Get percentage coordinates relative to ORIGINAL image."""
        px = lm_px(idx)
        return {"x_pct": (px[0] / orig_width) * 100, "y_pct": (px[1] / orig_height) * 100}
    
    # Compute measurement distances in pixels (in original image space)
    measurements = {}
    for name, (idx_a, idx_b) in MEASUREMENTS.items():
        if idx_a < num_landmarks and idx_b < num_landmarks:
            pa = lm_px(idx_a)
            pb = lm_px(idx_b)
            dist_px = _distance(pa, pb)
            measurements[name] = {
                "distance_px": round(dist_px, 1),
                "point_a": {"x_px": round(pa[0], 1), "y_px": round(pa[1], 1)},
                "point_b": {"x_px": round(pb[0], 1), "y_px": round(pb[1], 1)},
                "point_a_pct": lm_pct(idx_a),
                "point_b_pct": lm_pct(idx_b),
            }
    
    # Extract key individual landmarks (in original image space)
    landmarks = {}
    for name, idx in KEY_LANDMARKS.items():
        if idx < num_landmarks:
            px = lm_px(idx)
            landmarks[name] = {
                "x_px": round(px[0], 1),
                "y_px": round(px[1], 1),
                "x_pct": round((px[0] / orig_width) * 100, 2),
                "y_pct": round((px[1] / orig_height) * 100, 2),
            }
    
    # Face quality (z-coordinate variance)
    z_values = [face[i].z for i in range(min(468, num_landmarks))]
    z_std = float(np.std(z_values))
    
    elapsed = time.time() - start_time
    log.info(
        f"Face detected via '{crop_strategy}' (scale={scale_factor:.1f}x): {num_landmarks} landmarks, "
        f"bizygomatic={measurements.get('bizygomatic', {}).get('distance_px', 0)}px, "
        f"ipd_iris={measurements.get('ipd_iris', {}).get('distance_px', 0)}px, "
        f"face_height={measurements.get('face_height', {}).get('distance_px', 0)}px "
        f"({elapsed*1000:.0f}ms)"
    )
    
    return jsonify({
        "success": True,
        "image_width": orig_width,
        "image_height": orig_height,
        "num_landmarks": num_landmarks,
        "measurements": measurements,
        "landmarks": landmarks,
        "face_z_std": round(z_std, 4),
        "latency_ms": round(elapsed * 1000),
        "detection_strategy": crop_strategy,
    })


# ─── Routes ────────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    """Public health check endpoint."""
    return jsonify({"status": "ok", "model": "face_landmarker_v2"})


@app.route("/detect", methods=["POST"])
@require_api_key
def detect():
    """Detect face landmarks in an image. Requires API key."""
    start = time.time()

    data = request.get_json(force=True)
    image_url = data.get("image_url")
    image_base64 = data.get("image_base64")

    if not image_url and not image_base64:
        return jsonify({"success": False, "error": "Provide image_url or image_base64"}), 400

    # Download/decode image bytes once
    img_data = _download_image_bytes(image_url, image_base64)
    if img_data is None:
        return jsonify({"success": False, "error": "Failed to load image"}), 400

    # Get original image dimensions
    try:
        pil_orig = PILImage.open(BytesIO(img_data))
        orig_width, orig_height = pil_orig.size
    except Exception as e:
        return jsonify({"success": False, "error": f"Invalid image: {e}"}), 400

    # ─── Strategy 1: Try full image first ────────────────────────────────
    mp_image = _bytes_to_mp_image(img_data)
    if mp_image is not None:
        result = _detect_face(mp_image)
        if result is not None:
            return _build_response(
                result, orig_width, orig_height,
                detect_width=orig_width, detect_height=orig_height,
                crop_strategy="full",
                start_time=start,
            )
        log.info(f"Full image ({orig_width}x{orig_height}) -- no face, trying crops...")

    # ─── Strategy 2: Try progressive crops with optional upscaling ───────
    for (x1_pct, y1_pct, x2_pct, y2_pct, desc) in CROP_STRATEGIES:
        # Compute crop region in pixels
        cx1 = int(orig_width * x1_pct)
        cy1 = int(orig_height * y1_pct)
        cx2 = int(orig_width * x2_pct)
        cy2 = int(orig_height * y2_pct)
        cw = cx2 - cx1
        ch = cy2 - cy1
        
        if cw < 80 or ch < 80:
            continue
        
        try:
            cropped = pil_orig.crop((cx1, cy1, cx2, cy2))
        except Exception:
            continue
        
        # Determine if we need to upscale this crop
        scale = 1.0
        scaled_w, scaled_h = cw, ch
        
        if max(cw, ch) < MIN_UPSCALE_TARGET:
            scale = 2.0
            scaled_w = int(cw * scale)
            scaled_h = int(ch * scale)
            cropped = cropped.resize((scaled_w, scaled_h), PILImage.LANCZOS)
        elif max(cw, ch) > MAX_CROP_DIMENSION:
            pass
        
        crop_mp = _pil_to_mp_image(cropped)
        if crop_mp is None:
            continue
        
        result = _detect_face(crop_mp)
        if result is not None:
            log.info(f"Face found in crop '{desc}' ({cw}x{ch} at {cx1},{cy1}, scale={scale:.1f}x)")
            return _build_response(
                result, orig_width, orig_height,
                crop_offset_x=cx1, crop_offset_y=cy1,
                detect_width=scaled_w, detect_height=scaled_h,
                scale_factor=scale,
                crop_strategy=f"crop:{desc}",
                start_time=start,
            )

    # ─── Strategy 3: Last resort — upscale the full upper portion 2x ────
    if orig_width > 1500 or orig_height > 2000:
        upper_h = int(orig_height * 0.35)
        upper_crop = pil_orig.crop((0, 0, orig_width, upper_h))
        scale = 2.0
        up_w = int(orig_width * scale)
        up_h = int(upper_h * scale)
        if up_w > 2400:
            scale = 2400.0 / orig_width
            up_w = 2400
            up_h = int(upper_h * scale)
        upscaled = upper_crop.resize((up_w, up_h), PILImage.LANCZOS)
        
        crop_mp = _pil_to_mp_image(upscaled)
        if crop_mp is not None:
            result = _detect_face(crop_mp)
            if result is not None:
                log.info(f"Face found via upscaled upper 35% ({up_w}x{up_h}, scale={scale:.1f}x)")
                return _build_response(
                    result, orig_width, orig_height,
                    crop_offset_x=0, crop_offset_y=0,
                    detect_width=up_w, detect_height=up_h,
                    scale_factor=scale,
                    crop_strategy="upscaled:upper35%",
                    start_time=start,
                )

    # ─── All strategies failed ───────────────────────────────────────────
    elapsed = time.time() - start
    strategies_tried = ["full"] + [s[4] for s in CROP_STRATEGIES]
    if orig_width > 1500 or orig_height > 2000:
        strategies_tried.append("upscaled:upper35%")
    
    log.info(f"No face detected after {len(strategies_tried)} attempts ({elapsed*1000:.0f}ms)")
    return jsonify({
        "success": False,
        "error": "No face detected in image (tried full + crops + upscale)",
        "image_width": orig_width,
        "image_height": orig_height,
        "latency_ms": round(elapsed * 1000),
        "strategies_tried": strategies_tried,
    })


if __name__ == "__main__":
    log.info(f"Starting MediaPipe Face service on port {PORT}")
    app.run(host="0.0.0.0", port=PORT, debug=False)
