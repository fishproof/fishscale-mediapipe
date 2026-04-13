"""
MediaPipe Face + Hand Landmark Microservice

A lightweight Flask service that accepts an image (URL or base64) and returns
facial and/or hand landmark pixel coordinates for human reference scaling.

Endpoints:
  POST /detect       — Detect face landmarks in an image
  POST /detect-hand  — Detect hand landmarks in an image (21 keypoints per hand)
  GET  /health       — Health check

Face detection returns key measurement pairs for px/inch calibration:
  - bizygomatic (temple-to-temple): landmarks 234 ↔ 454
  - biocular (outer eye corners): landmarks 33 ↔ 263
  - ipd_inner (inner eye corners): landmarks 133 ↔ 362
  - ipd_iris (iris centers): landmarks 473 ↔ 468
  - face_height (forehead to chin): landmarks 10 ↔ 152

Hand detection returns 21 keypoints per hand with measurements:
  - index_finger_length: landmarks 5 → 8 (MCP to TIP)
  - middle_finger_length: landmarks 9 → 12 (MCP to TIP)
  - ring_finger_length: landmarks 13 → 16 (MCP to TIP)
  - palm_width: landmarks 5 → 17 (INDEX_MCP to PINKY_MCP)
  - hand_span: landmarks 4 → 20 (THUMB_TIP to PINKY_TIP)

Multi-scale detection:
  MediaPipe struggles when the target occupies <2% of image area. In typical fishing
  photos (1080x1920), a face is ~150px wide = ~1.6% area. We solve this by:
  1. Trying full image first
  2. Trying progressively tighter crops
  3. Upscaling small crops 2x to boost pixel count
  All coordinates are mapped back to original image space.
"""

import os
import sys
import math
import time
import base64
import tempfile
import logging
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

PORT = int(os.environ.get("MEDIAPIPE_PORT", 5055))
FACE_MODEL_PATH = os.path.join(os.path.dirname(__file__), "face_landmarker.task")
HAND_MODEL_PATH = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")

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

# ─── Face Landmark index pairs for measurements ──────────────────────────

MEASUREMENTS = {
    "bizygomatic": (234, 454),       # Temple-to-temple (~5.5" adult avg)
    "biocular": (33, 263),           # Outer eye corners (~3.5-3.7")
    "ipd_inner": (133, 362),         # Inner eye corners (~2.5")
    "ipd_iris": (473, 468),          # Iris center-to-center (~2.5")
    "face_height": (10, 152),        # Forehead to chin (~7-8")
}

# Additional individual face landmarks to return
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

# ─── Hand Landmark keypoint names (21 points) ────────────────────────────

HAND_LANDMARK_NAMES = [
    "WRIST",              # 0
    "THUMB_CMC",          # 1
    "THUMB_MCP",          # 2
    "THUMB_IP",           # 3
    "THUMB_TIP",          # 4
    "INDEX_FINGER_MCP",   # 5
    "INDEX_FINGER_PIP",   # 6
    "INDEX_FINGER_DIP",   # 7
    "INDEX_FINGER_TIP",   # 8
    "MIDDLE_FINGER_MCP",  # 9
    "MIDDLE_FINGER_PIP",  # 10
    "MIDDLE_FINGER_DIP",  # 11
    "MIDDLE_FINGER_TIP",  # 12
    "RING_FINGER_MCP",    # 13
    "RING_FINGER_PIP",    # 14
    "RING_FINGER_DIP",    # 15
    "RING_FINGER_TIP",    # 16
    "PINKY_MCP",          # 17
    "PINKY_PIP",          # 18
    "PINKY_DIP",          # 19
    "PINKY_TIP",          # 20
]

# Hand measurement pairs: (landmark_a, landmark_b, description)
HAND_MEASUREMENTS = {
    "index_finger_length": (5, 8),    # INDEX_MCP → INDEX_TIP
    "middle_finger_length": (9, 12),  # MIDDLE_MCP → MIDDLE_TIP
    "ring_finger_length": (13, 16),   # RING_MCP → RING_TIP
    "pinky_finger_length": (17, 20),  # PINKY_MCP → PINKY_TIP
    "thumb_length": (1, 4),           # THUMB_CMC → THUMB_TIP
    "palm_width": (5, 17),            # INDEX_MCP → PINKY_MCP
    "hand_span": (4, 20),             # THUMB_TIP → PINKY_TIP
    "wrist_to_middle_tip": (0, 12),   # WRIST → MIDDLE_TIP (full hand length)
    # Individual knuckle-to-knuckle segments (for partial palm visibility)
    "index_to_middle_mcp": (5, 9),    # INDEX_MCP → MIDDLE_MCP
    "middle_to_ring_mcp": (9, 13),    # MIDDLE_MCP → RING_MCP
    "ring_to_pinky_mcp": (13, 17),    # RING_MCP → PINKY_MCP
}

# ─── Multi-scale crop strategies (face) ──────────────────────────────────
# Each tuple: (x_start_pct, y_start_pct, x_end_pct, y_end_pct, description)
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

# ─── Multi-scale crop strategies (hand) ──────────────────────────────────
# Hands in fishing photos are typically in the middle/lower portion of the image.
# The person holds the fish at chest/waist level, so hands are usually 30-80% down.
HAND_CROP_STRATEGIES = [
    # Tier 1: Large crops covering hand region
    (0.0, 0.20, 1.0, 0.90, "middle 70%"),
    (0.0, 0.30, 1.0, 0.85, "lower-middle"),
    (0.0, 0.15, 1.0, 0.75, "upper-middle"),
    # Tier 2: Left/right halves (hand is often on one side)
    (0.0, 0.20, 0.60, 0.85, "left-middle"),
    (0.40, 0.20, 1.0, 0.85, "right-middle"),
    # Tier 3: Center focus
    (0.15, 0.25, 0.85, 0.80, "center-middle"),
    # Tier 4: Tight crops with upscaling
    (0.10, 0.30, 0.55, 0.75, "tight left-center"),
    (0.45, 0.30, 0.90, 0.75, "tight right-center"),
]

# ─── Initialize MediaPipe Models ──────────────────────────────────────────

# Face landmarker
log.info(f"Loading face landmarker model from {FACE_MODEL_PATH}")
if not os.path.exists(FACE_MODEL_PATH):
    log.error(f"Face model file not found: {FACE_MODEL_PATH}")
    sys.exit(1)

_face_options = mp_vision.FaceLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=FACE_MODEL_PATH),
    running_mode=mp_vision.RunningMode.IMAGE,
    num_faces=1,
    min_face_detection_confidence=0.3,
    min_face_presence_confidence=0.3,
)
_face_landmarker = mp_vision.FaceLandmarker.create_from_options(_face_options)
log.info("Face landmarker model loaded successfully")

# Hand landmarker
log.info(f"Loading hand landmarker model from {HAND_MODEL_PATH}")
_hand_landmarker = None
if os.path.exists(HAND_MODEL_PATH):
    try:
        _hand_options = mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=HAND_MODEL_PATH),
            running_mode=mp_vision.RunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=0.3,
            min_hand_presence_confidence=0.3,
        )
        _hand_landmarker = mp_vision.HandLandmarker.create_from_options(_hand_options)
        log.info("Hand landmarker model loaded successfully")
    except Exception as e:
        log.error(f"Failed to load hand landmarker: {e}")
else:
    log.warning(f"Hand model file not found: {HAND_MODEL_PATH} — hand detection disabled")

# ─── Flask App ─────────────────────────────────────────────────────────────

app = Flask(__name__)


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
        result = _face_landmarker.detect(mp_image)
        if result.face_landmarks and len(result.face_landmarks) > 0:
            return result
        return None
    except Exception as e:
        log.error(f"Face detection error: {e}")
        return None


def _detect_hand(mp_image: mp.Image):
    """Run hand detection on a MediaPipe image. Returns result or None."""
    if _hand_landmarker is None:
        return None
    try:
        result = _hand_landmarker.detect(mp_image)
        if result.hand_landmarks and len(result.hand_landmarks) > 0:
            return result
        return None
    except Exception as e:
        log.error(f"Hand detection error: {e}")
        return None


def _distance(p1: tuple, p2: tuple) -> float:
    """Euclidean distance between two (x, y) points."""
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


# ─── Face Detection Response Builder ─────────────────────────────────────

def _build_face_response(result, orig_width: int, orig_height: int,
                          crop_offset_x: int = 0, crop_offset_y: int = 0,
                          detect_width: int = 0, detect_height: int = 0,
                          scale_factor: float = 1.0,
                          crop_strategy: str = "full", start_time: float = 0):
    """Build the JSON response from a successful face detection result."""
    face = result.face_landmarks[0]
    num_landmarks = len(face)
    
    if detect_width == 0:
        detect_width = orig_width
    if detect_height == 0:
        detect_height = orig_height
    
    def lm_px(idx: int) -> tuple:
        lm = face[idx]
        det_x = lm.x * detect_width
        det_y = lm.y * detect_height
        crop_x = det_x / scale_factor
        crop_y = det_y / scale_factor
        orig_x = crop_x + crop_offset_x
        orig_y = crop_y + crop_offset_y
        return (orig_x, orig_y)
    
    def lm_pct(idx: int) -> dict:
        px = lm_px(idx)
        return {"x_pct": (px[0] / orig_width) * 100, "y_pct": (px[1] / orig_height) * 100}
    
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


# ─── Hand Detection Response Builder ─────────────────────────────────────

def _build_hand_response(result, orig_width: int, orig_height: int,
                          crop_offset_x: int = 0, crop_offset_y: int = 0,
                          detect_width: int = 0, detect_height: int = 0,
                          scale_factor: float = 1.0,
                          crop_strategy: str = "full", start_time: float = 0):
    """Build the JSON response from a successful hand detection result."""
    
    if detect_width == 0:
        detect_width = orig_width
    if detect_height == 0:
        detect_height = orig_height
    
    hands = []
    
    for hand_idx in range(len(result.hand_landmarks)):
        hand_lms = result.hand_landmarks[hand_idx]
        
        # Determine handedness
        handedness = "unknown"
        handedness_score = 0.0
        if result.handedness and hand_idx < len(result.handedness):
            h = result.handedness[hand_idx]
            if h and len(h) > 0:
                handedness = h[0].category_name  # "Left" or "Right"
                handedness_score = h[0].score
        
        def lm_px(idx: int) -> tuple:
            lm = hand_lms[idx]
            det_x = lm.x * detect_width
            det_y = lm.y * detect_height
            crop_x = det_x / scale_factor
            crop_y = det_y / scale_factor
            orig_x = crop_x + crop_offset_x
            orig_y = crop_y + crop_offset_y
            return (orig_x, orig_y)
        
        # Extract all 21 landmarks in original image space
        landmarks = {}
        for idx, name in enumerate(HAND_LANDMARK_NAMES):
            if idx < len(hand_lms):
                px = lm_px(idx)
                landmarks[name] = {
                    "index": idx,
                    "x_px": round(px[0], 1),
                    "y_px": round(px[1], 1),
                    "x_pct": round((px[0] / orig_width) * 100, 2),
                    "y_pct": round((px[1] / orig_height) * 100, 2),
                }
        
        # Compute hand measurements in pixels (original image space)
        measurements = {}
        for name, (idx_a, idx_b) in HAND_MEASUREMENTS.items():
            if idx_a < len(hand_lms) and idx_b < len(hand_lms):
                pa = lm_px(idx_a)
                pb = lm_px(idx_b)
                dist_px = _distance(pa, pb)
                measurements[name] = {
                    "distance_px": round(dist_px, 1),
                    "point_a": {"x_px": round(pa[0], 1), "y_px": round(pa[1], 1)},
                    "point_b": {"x_px": round(pb[0], 1), "y_px": round(pb[1], 1)},
                    "point_a_pct": {
                        "x_pct": round((pa[0] / orig_width) * 100, 2),
                        "y_pct": round((pa[1] / orig_height) * 100, 2),
                    },
                    "point_b_pct": {
                        "x_pct": round((pb[0] / orig_width) * 100, 2),
                        "y_pct": round((pb[1] / orig_height) * 100, 2),
                    },
                }
        
        # World landmarks (real-world 3D coordinates in meters)
        world_measurements = {}
        if result.hand_world_landmarks and hand_idx < len(result.hand_world_landmarks):
            world_lms = result.hand_world_landmarks[hand_idx]
            for mname, (idx_a, idx_b) in HAND_MEASUREMENTS.items():
                if idx_a < len(world_lms) and idx_b < len(world_lms):
                    wa = world_lms[idx_a]
                    wb = world_lms[idx_b]
                    dist_m = math.sqrt(
                        (wa.x - wb.x) ** 2 +
                        (wa.y - wb.y) ** 2 +
                        (wa.z - wb.z) ** 2
                    )
                    world_measurements[mname] = {
                        "distance_meters": round(dist_m, 4),
                        "distance_inches": round(dist_m * 39.3701, 2),
                    }
        
        hands.append({
            "handedness": handedness,
            "handedness_score": round(handedness_score, 3),
            "num_landmarks": len(hand_lms),
            "landmarks": landmarks,
            "measurements": measurements,
            "world_measurements": world_measurements,
        })
    
    elapsed = time.time() - start_time
    
    # Log summary
    hand_summary = ", ".join([
        f"{h['handedness']}(idx_len={h['measurements'].get('index_finger_length', {}).get('distance_px', 0)}px, "
        f"palm={h['measurements'].get('palm_width', {}).get('distance_px', 0)}px)"
        for h in hands
    ])
    log.info(
        f"Hand(s) detected via '{crop_strategy}' (scale={scale_factor:.1f}x): "
        f"{len(hands)} hand(s) — {hand_summary} ({elapsed*1000:.0f}ms)"
    )
    
    return jsonify({
        "success": True,
        "image_width": orig_width,
        "image_height": orig_height,
        "num_hands": len(hands),
        "hands": hands,
        "latency_ms": round(elapsed * 1000),
        "detection_strategy": crop_strategy,
    })


# ─── Health Endpoint ──────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "models": {
            "face": "face_landmarker_v2",
            "hand": "hand_landmarker" if _hand_landmarker else "not_loaded",
        },
    })


# ─── Face Detection Endpoint ─────────────────────────────────────────────

@app.route("/detect", methods=["POST"])
def detect():
    start = time.time()

    data = request.get_json(force=True)
    image_url = data.get("image_url")
    image_base64 = data.get("image_base64")

    if not image_url and not image_base64:
        return jsonify({"success": False, "error": "Provide image_url or image_base64"}), 400

    img_data = _download_image_bytes(image_url, image_base64)
    if img_data is None:
        return jsonify({"success": False, "error": "Failed to load image"}), 400

    try:
        pil_orig = PILImage.open(BytesIO(img_data))
        orig_width, orig_height = pil_orig.size
    except Exception as e:
        return jsonify({"success": False, "error": f"Invalid image: {e}"}), 400

    # Strategy 1: Try full image first
    mp_image = _bytes_to_mp_image(img_data)
    if mp_image is not None:
        result = _detect_face(mp_image)
        if result is not None:
            return _build_face_response(
                result, orig_width, orig_height,
                detect_width=orig_width, detect_height=orig_height,
                crop_strategy="full",
                start_time=start,
            )
        log.info(f"Full image ({orig_width}x{orig_height}) — no face, trying crops...")

    # Strategy 2: Try progressive crops with optional upscaling
    for (x1_pct, y1_pct, x2_pct, y2_pct, desc) in CROP_STRATEGIES:
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
        
        scale = 1.0
        scaled_w, scaled_h = cw, ch
        
        if max(cw, ch) < MIN_UPSCALE_TARGET:
            scale = 2.0
            scaled_w = int(cw * scale)
            scaled_h = int(ch * scale)
            cropped = cropped.resize((scaled_w, scaled_h), PILImage.LANCZOS)
        
        crop_mp = _pil_to_mp_image(cropped)
        if crop_mp is None:
            continue
        
        result = _detect_face(crop_mp)
        if result is not None:
            log.info(f"Face found in crop '{desc}' ({cw}x{ch} at {cx1},{cy1}, scale={scale:.1f}x)")
            return _build_face_response(
                result, orig_width, orig_height,
                crop_offset_x=cx1, crop_offset_y=cy1,
                detect_width=scaled_w, detect_height=scaled_h,
                scale_factor=scale,
                crop_strategy=f"crop:{desc}",
                start_time=start,
            )

    # Strategy 3: Last resort — upscale the full upper portion 2x
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
                return _build_face_response(
                    result, orig_width, orig_height,
                    crop_offset_x=0, crop_offset_y=0,
                    detect_width=up_w, detect_height=up_h,
                    scale_factor=scale,
                    crop_strategy="upscaled:upper35%",
                    start_time=start,
                )

    # All strategies failed
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


# ─── Hand Detection Endpoint ─────────────────────────────────────────────

@app.route("/detect-hand", methods=["POST"])
def detect_hand():
    start = time.time()

    if _hand_landmarker is None:
        return jsonify({
            "success": False,
            "error": "Hand landmarker model not loaded",
        }), 503

    data = request.get_json(force=True)
    image_url = data.get("image_url")
    image_base64 = data.get("image_base64")

    if not image_url and not image_base64:
        return jsonify({"success": False, "error": "Provide image_url or image_base64"}), 400

    img_data = _download_image_bytes(image_url, image_base64)
    if img_data is None:
        return jsonify({"success": False, "error": "Failed to load image"}), 400

    try:
        pil_orig = PILImage.open(BytesIO(img_data))
        orig_width, orig_height = pil_orig.size
    except Exception as e:
        return jsonify({"success": False, "error": f"Invalid image: {e}"}), 400

    # Strategy 1: Try full image first
    mp_image = _bytes_to_mp_image(img_data)
    if mp_image is not None:
        result = _detect_hand(mp_image)
        if result is not None:
            return _build_hand_response(
                result, orig_width, orig_height,
                detect_width=orig_width, detect_height=orig_height,
                crop_strategy="full",
                start_time=start,
            )
        log.info(f"Full image ({orig_width}x{orig_height}) — no hand, trying crops...")

    # Strategy 2: Try progressive crops focused on hand region
    for (x1_pct, y1_pct, x2_pct, y2_pct, desc) in HAND_CROP_STRATEGIES:
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
        
        scale = 1.0
        scaled_w, scaled_h = cw, ch
        
        if max(cw, ch) < MIN_UPSCALE_TARGET:
            scale = 2.0
            scaled_w = int(cw * scale)
            scaled_h = int(ch * scale)
            cropped = cropped.resize((scaled_w, scaled_h), PILImage.LANCZOS)
        
        crop_mp = _pil_to_mp_image(cropped)
        if crop_mp is None:
            continue
        
        result = _detect_hand(crop_mp)
        if result is not None:
            log.info(f"Hand found in crop '{desc}' ({cw}x{ch} at {cx1},{cy1}, scale={scale:.1f}x)")
            return _build_hand_response(
                result, orig_width, orig_height,
                crop_offset_x=cx1, crop_offset_y=cy1,
                detect_width=scaled_w, detect_height=scaled_h,
                scale_factor=scale,
                crop_strategy=f"crop:{desc}",
                start_time=start,
            )

    # Strategy 3: Last resort — upscale the middle portion 2x
    mid_top = int(orig_height * 0.25)
    mid_bot = int(orig_height * 0.80)
    mid_crop = pil_orig.crop((0, mid_top, orig_width, mid_bot))
    mid_h = mid_bot - mid_top
    scale = 2.0
    up_w = int(orig_width * scale)
    up_h = int(mid_h * scale)
    if up_w > 2400:
        scale = 2400.0 / orig_width
        up_w = 2400
        up_h = int(mid_h * scale)
    upscaled = mid_crop.resize((up_w, up_h), PILImage.LANCZOS)
    
    crop_mp = _pil_to_mp_image(upscaled)
    if crop_mp is not None:
        result = _detect_hand(crop_mp)
        if result is not None:
            log.info(f"Hand found via upscaled middle 55% ({up_w}x{up_h}, scale={scale:.1f}x)")
            return _build_hand_response(
                result, orig_width, orig_height,
                crop_offset_x=0, crop_offset_y=mid_top,
                detect_width=up_w, detect_height=up_h,
                scale_factor=scale,
                crop_strategy="upscaled:middle55%",
                start_time=start,
            )

    # All strategies failed
    elapsed = time.time() - start
    strategies_tried = ["full"] + [s[4] for s in HAND_CROP_STRATEGIES] + ["upscaled:middle55%"]
    
    log.info(f"No hand detected after {len(strategies_tried)} attempts ({elapsed*1000:.0f}ms)")
    return jsonify({
        "success": False,
        "error": "No hand detected in image (tried full + crops + upscale)",
        "image_width": orig_width,
        "image_height": orig_height,
        "latency_ms": round(elapsed * 1000),
        "strategies_tried": strategies_tried,
    })


if __name__ == "__main__":
    log.info(f"Starting MediaPipe Face+Hand service on port {PORT}")
    app.run(host="127.0.0.1", port=PORT, debug=False)
