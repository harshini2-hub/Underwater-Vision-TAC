"""
underwater_enhance.py  —  TAC 2026
Stub file — replace with your real calibration values and enhance logic.
"""

import cv2
import numpy as np

# ── Camera Matrices (replace with your real calibrated values) ──
CAMERA_MATRICES = {
    0: np.array([[600.,   0., 320.],
                 [  0., 600., 240.],
                 [  0.,   0.,   1.]], dtype=np.float32),

    1: np.array([[554.256,   0., 320.],
                 [  0., 554.256, 240.],
                 [  0.,       0.,  1.]], dtype=np.float32),
}

# ── Distortion Coefficients (replace with your real values) ──
DIST_COEFFS = {
    0: np.array([-0.30,  0.10, 0., 0.,  0.    ], dtype=np.float32),
    1: np.array([-0.38,  0.15, 0., 0., -0.035 ], dtype=np.float32),
}


def enhance_for_aruco(frame, cam_id, tac_mode=False):
    """
    Pre-process frame for better ArUco detection underwater.
    Replace this with your real enhancement logic if needed.
    """
    # Convert to grayscale-ish for ArUco — boost contrast with CLAHE
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(gray)

    if tac_mode:
        # Extra sharpening in TAC mode
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        enhanced_gray = cv2.filter2D(enhanced_gray, -1, kernel)

    # Return as BGR so the rest of the pipeline stays consistent
    return cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)
