"""
╔══════════════════════════════════════════════════════════════════════════════╗
║         ULTIMATE REAL-TIME FINGER COUNTER  —  YOLO26 Hand Keypoints         ║
║  Model: yolo26n_hand.pt  |  Dataset: hand-keypoints (21 kpts, MediaPipe)    ║
║  Counts: 0-5 per hand, 0-10 total  |  All orientations, all positions        ║
╚══════════════════════════════════════════════════════════════════════════════╝

Exact keypoint map (from hand-keypoints.yaml):
  0  = wrist
  1  = thumb_cmc      2  = thumb_mcp      3  = thumb_ip       4  = thumb_tip
  5  = index_mcp      6  = index_pip      7  = index_dip      8  = index_tip
  9  = middle_mcp    10  = middle_pip    11  = middle_dip    12  = middle_tip
 13  = ring_mcp      14  = ring_pip      15  = ring_dip      16  = ring_tip
 17  = pinky_mcp     18  = pinky_pip     19  = pinky_dip     20  = pinky_tip

Algorithm overview:
  ─ Uses result.keypoints.data → shape (N_hands, 21, 3)  [x, y, conf]
  ─ Filters by per-keypoint confidence (>= KPT_CONF_THRESH)
  ─ Computes a HAND AXIS VECTOR (wrist → middle_mcp) for palm direction
  ─ Projects each finger tip vector onto the axis to decide UP/DOWN
    (works for any rotation: pointing up, down, sideways, tilted)
  ─ Thumb uses a dedicated lateral axis (wrist → index_mcp)
  ─ Each finger also requires tip to be FURTHER from wrist than PIP
    (extra curl check via distance ratio)
  ─ Temporal smoothing (rolling window) to avoid flicker
  ─ Supports 0-10 fingers (two hands simultaneously)
"""

import cv2
import numpy as np
from collections import deque
from ultralytics import YOLO

# ═══════════════════════════════  CONFIGURATION  ═════════════════════════════

MODEL_PATH        = "yolo26n_hand.pt"   # path to your trained weights

# Detection thresholds
DETECT_CONF       = 0.40   # YOLO box confidence for hand detection
KPT_CONF_THRESH   = 0.20   # minimum per-keypoint confidence to use it

# Finger extension thresholds
AXIS_DOT_THRESH   = 0.10   # dot-product threshold on normalised axis (main check)
DIST_RATIO_THRESH = 1.10   # tip_dist / pip_dist must be > this (curl check)

# Temporal smoothing: majority-vote over last N frames per hand slot
SMOOTH_WINDOW     = 5      # frames

# Camera
CAM_INDEX         = 0
CAM_WIDTH         = 1280
CAM_HEIGHT        = 720

# ═════════════════════════════  KEYPOINT INDICES  ════════════════════════════

W    = 0                                         # wrist
# Thumb
TC, TM, TI, TT  = 1, 2, 3, 4                    # cmc mcp ip tip
# Index
IM, IP, ID, IT  = 5, 6, 7, 8                    # mcp pip dip tip
# Middle
MM, MP, MD, MT  = 9, 10, 11, 12
# Ring
RM, RP, RD, RT  = 13, 14, 15, 16
# Pinky
PM, PP, PD, PT  = 17, 18, 19, 20

# (name, tip_idx, pip_idx, mcp_idx)
FOUR_FINGERS = [
    ("Index",  IT, IP, IM),
    ("Middle", MT, MP, MM),
    ("Ring",   RT, RP, RM),
    ("Pinky",  PT, PP, PM),
]

FINGER_NAMES = ["Thumb", "Index", "Middle", "Ring", "Pinky"]

# ═══════════════════════════  COLOUR PALETTE  ════════════════════════════════

# Count colours (0 → red, 5 → bright green)
COUNT_COLOR = {
    0:  (0,   0,   230),
    1:  (0,   80,  255),
    2:  (0,   165, 255),
    3:  (0,   230, 230),
    4:  (60,  255, 60),
    5:  (0,   210, 0),
}
FINGER_UP_COLOR   = (50,  255, 50)
FINGER_DOWN_COLOR = (50,  50,  200)
HUD_BG            = (20,  20,  20)

# ═════════════════════════  CORE FINGER LOGIC  ═══════════════════════════════

def _valid(kp_data, idx, thresh=KPT_CONF_THRESH):
    """Return True if keypoint idx has confidence above threshold."""
    return float(kp_data[idx, 2]) >= thresh


def _pt(kp_data, idx):
    """Return (x, y) pixel coords of keypoint idx."""
    return np.array([kp_data[idx, 0], kp_data[idx, 1]], dtype=np.float64)


def _dist(a, b):
    return np.linalg.norm(b - a)


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-6 else v


def finger_extended_axis(kp_data, tip_idx, pip_idx, mcp_idx, axis_vec):
    """
    Generic finger extension test using hand axis projection.

    A finger is considered EXTENDED when:
      1. Its tip vector (wrist→tip) projected on the hand axis > AXIS_DOT_THRESH
         (the tip is pointing in the same direction the hand axis points)
      2. dist(wrist→tip) > dist(wrist→pip) * DIST_RATIO_THRESH
         (tip is further out than pip, i.e., not curled)

    Both conditions required.  Falls back gracefully if keypoints are occluded.
    """
    # Need at minimum: wrist, pip, tip
    if not (_valid(kp_data, W) and _valid(kp_data, tip_idx) and _valid(kp_data, pip_idx)):
        return False

    wrist   = _pt(kp_data, W)
    tip     = _pt(kp_data, tip_idx)
    pip     = _pt(kp_data, pip_idx)

    tip_vec  = tip - wrist
    pip_vec  = pip - wrist

    # 1. Axis dot-product check
    axis_unit = _unit(axis_vec)
    tip_dot   = np.dot(_unit(tip_vec), axis_unit)
    pip_dot   = np.dot(_unit(pip_vec), axis_unit)

    axis_check = (tip_dot > AXIS_DOT_THRESH) and (tip_dot > pip_dot - 0.05)

    # 2. Distance ratio check (tip further from wrist than pip)
    tip_dist = _dist(wrist, tip)
    pip_dist = _dist(wrist, pip)
    dist_check = (pip_dist < 1e-3) or (tip_dist / (pip_dist + 1e-6) > DIST_RATIO_THRESH)

    return axis_check and dist_check


def thumb_extended_axis(kp_data, axis_vec):
    """
    Thumb uses a lateral axis: wrist → index_mcp.
    The thumb opens sideways, NOT along the main hand axis.

    Extended if:
      1. Thumb tip projects significantly along the lateral axis
      2. Tip is further from wrist than thumb IP joint
    """
    if not (_valid(kp_data, W) and _valid(kp_data, TT) and _valid(kp_data, TI)):
        return False

    wrist       = _pt(kp_data, W)
    tip         = _pt(kp_data, TT)
    ip          = _pt(kp_data, TI)
    index_mcp   = _pt(kp_data, IM) if _valid(kp_data, IM) else None

    # Lateral axis = wrist → index_mcp (thumb opens away from index)
    if index_mcp is not None:
        lateral_vec = index_mcp - wrist
    else:
        # Fallback: axis perpendicular to hand_axis in 2D
        lateral_vec = np.array([-axis_vec[1], axis_vec[0]])

    lateral_unit = _unit(lateral_vec)

    tip_vec = tip - wrist
    ip_vec  = ip  - wrist

    tip_lat = np.dot(_unit(tip_vec), lateral_unit)
    ip_lat  = np.dot(_unit(ip_vec),  lateral_unit)

    # Thumb is "open" if its tip is significantly lateral compared to IP
    lat_check  = abs(tip_lat) > AXIS_DOT_THRESH
    dist_check = _dist(wrist, tip) > _dist(wrist, ip) * DIST_RATIO_THRESH

    # Additionally, the thumb should NOT point the same direction as the main axis
    # (that would mean the fist is pointing forward, not the thumb opening)
    thumb_along_axis = abs(np.dot(_unit(tip_vec), _unit(axis_vec)))
    not_straight_ahead = thumb_along_axis < 0.92

    return lat_check and dist_check and not_straight_ahead


def compute_hand_axis(kp_data):
    """
    Compute the primary hand extension direction: wrist → middle_mcp.
    This is the direction fingers extend when open.
    Falls back to wrist → index_mcp, then wrist → ring_mcp.
    """
    wrist = _pt(kp_data, W) if _valid(kp_data, W) else None
    if wrist is None:
        return None

    for mcp_idx in [MM, IM, RM, PM]:
        if _valid(kp_data, mcp_idx):
            mcp = _pt(kp_data, mcp_idx)
            v = mcp - wrist
            if np.linalg.norm(v) > 1e-3:
                return v   # raw vector (will be normalised inside functions)

    return None


def count_fingers_robust(kp_data):
    """
    Full robust finger count for one hand.

    kp_data : np.ndarray shape (21, 3)  — [x, y, conf] per keypoint

    Returns:
        count       : int  0-5
        finger_up   : list[bool]  [Thumb, Index, Middle, Ring, Pinky]
        valid       : bool  — False if hand keypoints are too sparse/unreliable
    """
    # Require wrist to be detected
    if not _valid(kp_data, W):
        return 0, [False] * 5, False

    # Count valid keypoints; if too few, skip
    valid_count = sum(1 for i in range(21) if float(kp_data[i, 2]) >= KPT_CONF_THRESH)
    if valid_count < 7:
        return 0, [False] * 5, False

    axis = compute_hand_axis(kp_data)
    if axis is None:
        return 0, [False] * 5, False

    results = []

    # ── THUMB ──────────────────────────────────────────────────────────────
    results.append(thumb_extended_axis(kp_data, axis))

    # ── 4 FINGERS ──────────────────────────────────────────────────────────
    for _name, tip, pip, mcp in FOUR_FINGERS:
        results.append(finger_extended_axis(kp_data, tip, pip, mcp, axis))

    count = sum(results)
    return count, results, True


# ═══════════════════════════  TEMPORAL SMOOTHER  ═════════════════════════════

class HandSmoother:
    """
    Keeps a rolling history of finger states for one hand slot.
    Returns the majority-vote result over the last SMOOTH_WINDOW frames.
    """
    def __init__(self, window=SMOOTH_WINDOW):
        self.window  = window
        self.history = [deque(maxlen=window) for _ in range(5)]   # 5 fingers

    def update(self, finger_up_list):
        for i, up in enumerate(finger_up_list):
            self.history[i].append(int(up))

    def get_smoothed(self):
        smoothed = []
        for dq in self.history:
            if len(dq) == 0:
                smoothed.append(False)
            else:
                smoothed.append(sum(dq) > len(dq) / 2)
        return smoothed, sum(smoothed)

    def reset(self):
        for dq in self.history:
            dq.clear()


# We keep up to 2 smoothers (one per hand slot)
smoothers = [HandSmoother(), HandSmoother()]


# ═══════════════════════════  DRAWING HELPERS  ═══════════════════════════════

def draw_rounded_rect(img, x1, y1, x2, y2, color, radius=12, thickness=-1):
    """Draw a rounded rectangle."""
    cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, thickness)
    cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, thickness)
    cv2.circle(img, (x1 + radius, y1 + radius), radius, color, thickness)
    cv2.circle(img, (x2 - radius, y1 + radius), radius, color, thickness)
    cv2.circle(img, (x1 + radius, y2 - radius), radius, color, thickness)
    cv2.circle(img, (x2 - radius, y2 - radius), radius, color, thickness)


def draw_finger_indicators(img, x1, y2, finger_up):
    """Draw 5 coloured circles below the bounding box for each finger state."""
    labels = ["T", "I", "M", "R", "P"]
    spacing = 28
    start_x = x1 + 10
    base_y  = y2 + 28
    for i, (lbl, up) in enumerate(zip(labels, finger_up)):
        cx = start_x + i * spacing
        color = FINGER_UP_COLOR if up else FINGER_DOWN_COLOR
        cv2.circle(img, (cx, base_y), 11, color, -1)
        cv2.circle(img, (cx, base_y), 11, (255, 255, 255), 1)
        cv2.putText(img, lbl, (cx - 5, base_y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 1, cv2.LINE_AA)


def draw_hand_label(img, x1, y1, count, finger_up, hand_label=""):
    """Draw finger count pill above the bounding box."""
    color = COUNT_COLOR.get(count, (200, 200, 200))
    label = f" {count} {'finger' if count == 1 else 'fingers'} {hand_label}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.85, 2)
    bx1 = x1
    by1 = max(0, y1 - th - 22)
    bx2 = x1 + tw + 14
    by2 = y1 - 2
    draw_rounded_rect(img, bx1, by1, bx2, by2, color, radius=8)
    cv2.putText(img, label, (bx1 + 6, by2 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 0), 2, cv2.LINE_AA)


def draw_hud(img, total, hand_counts):
    """Draw the top-left HUD showing total + per-hand count."""
    hud_color = COUNT_COLOR.get(min(total, 5), (200, 200, 200))

    # Main total panel
    draw_rounded_rect(img, 8, 8, 340, 90, HUD_BG, radius=12)
    draw_rounded_rect(img, 8, 8, 340, 90, hud_color, radius=12, thickness=2)
    cv2.putText(img,
                f"  Total fingers: {total}",
                (16, 68),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, hud_color, 3, cv2.LINE_AA)

    # Per-hand breakdown
    if len(hand_counts) > 1:
        txt = "  " + "   ".join(
            f"Hand {i+1}: {c}" for i, c in enumerate(hand_counts))
        cv2.putText(img, txt, (16, 108),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1, cv2.LINE_AA)


def draw_keypoints_manual(img, kp_data, color=(180, 180, 0)):
    """Draw raw keypoints as small dots (on top of YOLO drawing for debug)."""
    for i in range(21):
        if float(kp_data[i, 2]) >= KPT_CONF_THRESH:
            x = int(kp_data[i, 0])
            y = int(kp_data[i, 1])
            cv2.circle(img, (x, y), 3, color, -1)


# ═════════════════════════════  MAIN LOOP  ═══════════════════════════════════

def main():
    print("=" * 70)
    print("  YOLO26 Ultimate Finger Counter")
    print("  Model :", MODEL_PATH)
    print("  Press Q to quit | D to toggle debug keypoints")
    print("=" * 70)

    model = YOLO(MODEL_PATH)

    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)

    if not cap.isOpened():
        print("[ERROR] Cannot open webcam!")
        return

    debug_kpts = False    # toggle with D key
    frame_no   = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            print("[WARN] Frame grab failed, retrying...")
            continue

        frame_no += 1

        # ── YOLO Inference ────────────────────────────────────────────────
        results = model.predict(frame, conf=DETECT_CONF, verbose=False)
        annotated = results[0].plot()   # YOLO draws skeleton + boxes

        total_count  = 0
        hand_counts  = []

        result = results[0]

        if result.keypoints is not None and result.boxes is not None:
            # kp_all: shape (N_hands, 21, 3)  x, y, conf
            kp_all   = result.keypoints.data.cpu().numpy()
            boxes_all = result.boxes.xyxy.cpu().numpy()

            n_hands = len(kp_all)

            for hand_idx in range(n_hands):
                kp_data = kp_all[hand_idx]      # (21, 3)
                box     = boxes_all[hand_idx]
                x1, y1, x2, y2 = map(int, box)

                # ── Robust count ──────────────────────────────────────────
                count, finger_up, valid = count_fingers_robust(kp_data)

                if not valid:
                    # Hand detected but keypoints too sparse → skip smoothing
                    hand_counts.append(0)
                    continue

                # ── Temporal smoothing ────────────────────────────────────
                smoother_idx = min(hand_idx, 1)  # max 2 smoothers
                smoothers[smoother_idx].update(finger_up)
                finger_up_smooth, count_smooth = smoothers[smoother_idx].get_smoothed()

                total_count += count_smooth
                hand_counts.append(count_smooth)

                # ── Draw ──────────────────────────────────────────────────
                draw_hand_label(annotated, x1, y1, count_smooth, finger_up_smooth,
                                f"(H{hand_idx+1})" if n_hands > 1 else "")
                draw_finger_indicators(annotated, x1, y2, finger_up_smooth)

                if debug_kpts:
                    draw_keypoints_manual(annotated, kp_data)

                # Finger-name text (right side of box)
                for fi, (fname, fup) in enumerate(zip(FINGER_NAMES, finger_up_smooth)):
                    fc = FINGER_UP_COLOR if fup else FINGER_DOWN_COLOR
                    cv2.putText(annotated,
                                f"{fname}: {'UP' if fup else '--'}",
                                (x2 + 6, y1 + 20 + fi * 22),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, fc, 1, cv2.LINE_AA)

        # Reset smoothers for hands that disappeared
        n_detected = len(hand_counts)
        for si in range(n_detected, 2):
            smoothers[si].reset()

        # ── HUD ───────────────────────────────────────────────────────────
        draw_hud(annotated, total_count, hand_counts)

        # ── FPS overlay ───────────────────────────────────────────────────
        cv2.putText(annotated,
                    f"Hands: {len(hand_counts)}",
                    (CAM_WIDTH - 180, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2, cv2.LINE_AA)

        cv2.imshow("YOLO26 Finger Counter  [Q=quit | D=debug]", annotated)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("d"):
            debug_kpts = not debug_kpts
            print(f"[INFO] Debug keypoints: {'ON' if debug_kpts else 'OFF'}")

    cap.release()
    cv2.destroyAllWindows()
    print("Bye!")


if __name__ == "__main__":
    main()