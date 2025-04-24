import cv2
import mediapipe as mp
import pyautogui
import time
import numpy as np

# Initialize MediaPipe and camera
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)
cam = cv2.VideoCapture(0)
screen_w, screen_h = pyautogui.size()

# Eye landmarks for EAR calculation
LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]

# Cooldowns
last_left_click_time = 0
last_right_click_time = 0
CLICK_COOLDOWN = 1  # in seconds
EAR_THRESHOLD = 0.21  # tuned threshold for blink

# Function to calculate EAR
def calculate_ear(eye_landmarks):
    # eye_landmarks: list of 6 (x, y)
    A = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
    B = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
    C = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
    ear = (A + B) / (2.0 * C)
    return ear

while True:
    success, frame = cam.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    frame_h, frame_w, _ = frame.shape

    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]

        # Get 2D points of left and right eye
        left_eye = []
        right_eye = []
        for idx in LEFT_EYE_IDX:
            point = face_landmarks.landmark[idx]
            x, y = int(point.x * frame_w), int(point.y * frame_h)
            left_eye.append(np.array([x, y]))
            cv2.circle(frame, (x, y), 2, (255, 255, 0), -1)
        for idx in RIGHT_EYE_IDX:
            point = face_landmarks.landmark[idx]
            x, y = int(point.x * frame_w), int(point.y * frame_h)
            right_eye.append(np.array([x, y]))
            cv2.circle(frame, (x, y), 2, (0, 255, 255), -1)

        # Calculate EARs
        left_ear = calculate_ear(np.array(left_eye))
        right_ear = calculate_ear(np.array(right_eye))
        current_time = time.time()

        # Debug EAR display
        cv2.putText(frame, f'LEFT EAR: {left_ear:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(frame, f'RIGHT EAR: {right_ear:.2f}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Left Eye Blink = Left Click
        if left_ear < EAR_THRESHOLD and (current_time - last_left_click_time) > CLICK_COOLDOWN:
            pyautogui.click(button='left')
            last_left_click_time = current_time

        # Right Eye Blink = Right Click
        if right_ear < EAR_THRESHOLD and (current_time - last_right_click_time) > CLICK_COOLDOWN:
            pyautogui.click(button='right')
            last_right_click_time = current_time

        # Move cursor using iris landmark (id: 475)
        iris = face_landmarks.landmark[475]
        cursor_x = int(iris.x * screen_w)
        cursor_y = int(iris.y * screen_h)
        pyautogui.moveTo(cursor_x, cursor_y)

    # Show frame
    cv2.imshow("Eye Controlled Mouse", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cam.release()
cv2.destroyAllWindows()
