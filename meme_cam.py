import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pyvirtualcam
import numpy as np
import time
import math
import os
import urllib.request

# ==========================================
# 1. CONFIGURATION & IMAGE LOADING
# ==========================================
MEME_DURATION = 2.0  # Seconds the image stays on screen
COOLDOWN_DURATION = 3.0  # Seconds the system ignores new gestures AFTER the meme disappears


def get_meme_image(filename, fallback_text):
    if os.path.exists(filename):
        return cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    else:
        img = np.zeros((300, 400, 4), dtype=np.uint8)
        img[..., 3] = 200
        img[..., 0] = 255
        cv2.putText(img, fallback_text, (30, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255, 255), 3)
        return img


meme_cinema = get_meme_image("cinema.png", "CINEMA!")
meme_sleeping = get_meme_image("sleeping.png", "SLEEPING!")
meme_double_fist = get_meme_image("double_fist.png", "DOUBLE FIST!")
meme_ok = get_meme_image("ok.png", "OK SIGN!")
meme_fist_mouth = get_meme_image("fist_mouth.png", "FIST MOUTH!")
meme_facepalm = get_meme_image("facepalm.png", "FACEPALM!")
meme_scream = get_meme_image("scream.png", "SCREAM!")
meme_scratch = get_meme_image("scratch.png", "SCRATCHING!")


# ==========================================
# 2. DOWNLOAD AI MODELS AUTOMATICALLY
# ==========================================
def download_model(url, filename):
    if not os.path.exists(filename):
        print(f"Downloading AI Model: {filename}...")
        urllib.request.urlretrieve(url, filename)


download_model(
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
    "hand_landmarker.task")
download_model(
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
    "pose_landmarker_lite.task")
download_model(
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
    "face_landmarker.task")


# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================
def calc_distance(p1, p2):
    return math.hypot(p1.x - p2.x, p1.y - p2.y)


def is_fist(hand_landmarks):
    fingers_curled = [
        hand_landmarks[8].y > hand_landmarks[6].y,
        hand_landmarks[12].y > hand_landmarks[10].y,
        hand_landmarks[16].y > hand_landmarks[14].y,
        hand_landmarks[20].y > hand_landmarks[18].y
    ]
    return all(fingers_curled)


def is_open_hand(hand_landmarks):
    fingers_extended = [
        hand_landmarks[8].y < hand_landmarks[6].y,
        hand_landmarks[12].y < hand_landmarks[10].y,
        hand_landmarks[16].y < hand_landmarks[14].y,
        hand_landmarks[20].y < hand_landmarks[18].y
    ]
    return all(fingers_extended)


def is_sideways_hand(hand_landmarks):
    dx = abs(hand_landmarks[9].x - hand_landmarks[0].x)
    dy = abs(hand_landmarks[9].y - hand_landmarks[0].y)
    return (dx > dy * 1.5) and is_open_hand(hand_landmarks)


def overlay_transparent(background, overlay):
    bg_h, bg_w, _ = background.shape
    ov_h, ov_w, _ = overlay.shape

    if ov_h > bg_h or ov_w > bg_w:
        scale = min(bg_w / ov_w, bg_h / ov_h) * 0.8
        ov_w, ov_h = int(ov_w * scale), int(ov_h * scale)
        overlay = cv2.resize(overlay, (ov_w, ov_h))

    x = (bg_w - ov_w) // 2
    y = (bg_h - ov_h) // 2

    overlay_img = overlay[..., :3]
    mask = overlay[..., 3:] / 255.0

    region = background[y:y + ov_h, x:x + ov_w]
    background[y:y + ov_h, x:x + ov_w] = (1.0 - mask) * region + mask * overlay_img
    return background


def draw_debug_points(frame, landmarks, color):
    h, w, _ = frame.shape
    for lm in landmarks:
        if hasattr(lm, 'visibility') and lm.visibility is not None:
            if lm.visibility < 0.5:
                continue
        x, y = int(lm.x * w), int(lm.y * h)
        cv2.circle(frame, (x, y), 2, color, -1)


# ==========================================
# 4. MAIN APPLICATION
# ==========================================
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or math.isnan(fps): fps = 30.0

    options_hand = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path='hand_landmarker.task'),
        num_hands=2, running_mode=vision.RunningMode.VIDEO)
    hand_landmarker = vision.HandLandmarker.create_from_options(options_hand)

    options_pose = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path='pose_landmarker_lite.task'),
        running_mode=vision.RunningMode.VIDEO)
    pose_landmarker = vision.PoseLandmarker.create_from_options(options_pose)

    options_face = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path='face_landmarker.task'),
        output_face_blendshapes=True, running_mode=vision.RunningMode.VIDEO)
    face_landmarker = vision.FaceLandmarker.create_from_options(options_face)

    # --- STATE VARIABLES ---
    active_meme = None
    meme_start_time = 0
    start_time = time.time()

    show_debug_spots = False
    last_toggle_time = 0

    # Universal Hold Timer Variables
    pending_meme = None
    gesture_hold_start_time = 0

    print("Starting Virtual Camera... (Press 'q' in the preview window to quit)")

    with pyvirtualcam.Camera(width=width, height=height, fps=fps) as cam:
        while True:
            ret, frame = cap.read()
            if not ret: break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            current_time = time.time()
            timestamp_ms = int((current_time - start_time) * 1000)

            hand_res = hand_landmarker.detect_for_video(mp_image, timestamp_ms)
            pose_res = pose_landmarker.detect_for_video(mp_image, timestamp_ms)
            face_res = face_landmarker.detect_for_video(mp_image, timestamp_ms)

            # --- DEBUG SPOTS TOGGLE ---
            if hand_res.hand_landmarks:
                for hand_landmarks in hand_res.hand_landmarks:
                    if is_sideways_hand(hand_landmarks):
                        if current_time - last_toggle_time > 2.0:
                            show_debug_spots = not show_debug_spots
                            last_toggle_time = current_time
                        break

            # --- GESTURE RECOGNITION LOGIC ---
            time_since_last_meme = current_time - meme_start_time

            # Clear the meme from the screen once its 2-second duration is up
            if active_meme is not None and time_since_last_meme > MEME_DURATION:
                active_meme = None

            # Only listen for new gestures if both duration and cooldown have passed
            if time_since_last_meme > (MEME_DURATION + COOLDOWN_DURATION):

                detected_meme = None

                # Extract Face Blendshapes
                jaw_open = 0.0
                look_up_left = 0.0
                look_up_right = 0.0

                if face_res.face_blendshapes:
                    blendshapes = face_res.face_blendshapes[0]
                    left_blink = next((item.score for item in blendshapes if item.category_name == 'eyeBlinkLeft'), 0)
                    right_blink = next((item.score for item in blendshapes if item.category_name == 'eyeBlinkRight'), 0)
                    jaw_open = next((item.score for item in blendshapes if item.category_name == 'jawOpen'), 0)
                    look_up_left = next((item.score for item in blendshapes if item.category_name == 'eyeLookUpLeft'),
                                        0)
                    look_up_right = next((item.score for item in blendshapes if item.category_name == 'eyeLookUpRight'),
                                         0)

                    # 1. SLEEPING
                    if left_blink > 0.5 and right_blink > 0.5:
                        detected_meme = meme_sleeping

                if pose_res.pose_landmarks and len(pose_res.pose_landmarks) > 0:
                    pose = pose_res.pose_landmarks[0]
                    nose = pose[0]
                    mouth = pose[10]
                    left_ear, right_ear = pose[7], pose[8]

                    if hand_res.hand_landmarks:
                        # 2-HAND GESTURES (Checked first for priority)
                        if len(hand_res.hand_landmarks) >= 2 and detected_meme is None:
                            hand1, hand2 = hand_res.hand_landmarks[0], hand_res.hand_landmarks[1]
                            wrist1, wrist2 = pose[15], pose[16]

                            # THE SCREAM
                            if jaw_open > 0.2:
                                if (calc_distance(wrist1, left_ear) < 0.2 or calc_distance(wrist1, right_ear) < 0.2) and \
                                        (calc_distance(wrist2, left_ear) < 0.2 or calc_distance(wrist2,
                                                                                                right_ear) < 0.2):
                                    detected_meme = meme_scream

                            # CINEMA / DOUBLE FIST (Only checks if Scream wasn't triggered)
                            wrists_raised = (wrist1.visibility > 0.5 and wrist1.y < pose[11].y) and \
                                            (wrist2.visibility > 0.5 and wrist2.y < pose[12].y)

                            if wrists_raised and detected_meme is None:
                                if is_open_hand(hand1) and is_open_hand(hand2):
                                    detected_meme = meme_cinema
                                elif is_fist(hand1) and is_fist(hand2):
                                    detected_meme = meme_double_fist

                        # 1-HAND GESTURES
                        if detected_meme is None:
                            for hand_landmarks in hand_res.hand_landmarks:
                                wrist = hand_landmarks[0]

                                # OK SIGN
                                if calc_distance(hand_landmarks[4], hand_landmarks[8]) < 0.05 and hand_landmarks[12].y < \
                                        hand_landmarks[10].y:
                                    detected_meme = meme_ok
                                    break

                                # FIST OVER MOUTH
                                if is_fist(hand_landmarks) and calc_distance(wrist, mouth) < 0.15:
                                    detected_meme = meme_fist_mouth
                                    break

                                # HEAD SCRATCHING
                                # Check if wrist is raised above the nose level
                                if wrist.y < nose.y:
                                    # Check if fingertips (8, 12) are pointing down (dangling)
                                    if hand_landmarks[8].y > hand_landmarks[5].y and hand_landmarks[12].y > \
                                            hand_landmarks[9].y:
                                        detected_meme = meme_scratch
                                        break

                                # FACEPALM
                                if calc_distance(wrist, nose) < 0.15 and wrist.y < nose.y:
                                    detected_meme = meme_facepalm
                                    break

                # --- UNIVERSAL HOLD TIMER LOGIC ---
                if detected_meme is not None:
                    # If we are holding the same pose we detected previously
                    if pending_meme is detected_meme:
                        # Sleep needs 1 second, everything else needs 0.5 seconds
                        required_hold_time = 1.0 if detected_meme is meme_sleeping else 0.5

                        if current_time - gesture_hold_start_time >= required_hold_time:
                            active_meme = pending_meme
                            meme_start_time = current_time
                            pending_meme = None  # Reset timer
                    else:
                        # A new pose was detected this frame
                        pending_meme = detected_meme
                        gesture_hold_start_time = current_time
                else:
                    # No pose detected, drop the timer
                    pending_meme = None
                    gesture_hold_start_time = 0

            else:
                # Inside Cooldown/Meme Duration, reset the hold timers so they don't buffer
                pending_meme = None
                gesture_hold_start_time = 0

            # --- DRAW VISUAL DEBUG INDICATORS ---
            if show_debug_spots:
                if face_res.face_landmarks:
                    for face_landmarks in face_res.face_landmarks:
                        draw_debug_points(frame, face_landmarks, (255, 255, 255))
                if pose_res.pose_landmarks:
                    for pose_landmarks in pose_res.pose_landmarks:
                        draw_debug_points(frame, pose_landmarks, (255, 0, 0))
                if hand_res.hand_landmarks:
                    for hand_landmarks in hand_res.hand_landmarks:
                        draw_debug_points(frame, hand_landmarks, (0, 255, 0))

            # --- RENDER OVERLAY ---
            if active_meme is not None:
                frame = overlay_transparent(frame, active_meme)

            # --- OUTPUT ---
            cv2.imshow("Webcam Preview (Press Q to exit)", frame)
            out_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cam.send(out_frame)
            cam.sleep_until_next_frame()

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()