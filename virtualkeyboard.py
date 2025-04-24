import cv2
import mediapipe as mp
import time
import numpy as np
import pyautogui

# Initialize MediaPipe Hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# Initialize hand tracker
hands = mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# Define the virtual keyboard
ctime = time.time()  # Define ctime here

# Function to calculate distance between two points
def calculateIntDistance(pt1, pt2):
    return int(((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2) ** 0.5)

# Creating keys
class Key:
    def __init__(self, x, y, w, h, text, special_characters=None):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.text = text
        self.special_characters = special_characters
        self.start_time = 0
        self.pressed = False

    def drawKey(self, img, text_color=(255, 255, 255), bg_color=(0, 0, 0), pressed_color=(0, 255, 0),
                alpha=0.5, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, thickness=2):
        # Draw the key box
        bg_rec = img[self.y: self.y + self.h, self.x: self.x + self.w]
        if bg_rec.shape[0] == 0 or bg_rec.shape[1] == 0:
            return  # Skip if the region is empty

        if self.pressed:
            color = pressed_color
        else:
            color = bg_color

        white_rect = np.ones(bg_rec.shape, dtype=np.uint8)
        white_rect[:] = color
        res = cv2.addWeighted(bg_rec, alpha, white_rect, 1 - alpha, 1.0)

        # Putting the image back to its position
        img[self.y: self.y + self.h, self.x: self.x + self.w] = res

        # Put the letter
        text_size = cv2.getTextSize(self.text, fontFace, fontScale, thickness)
        text_pos = (int(self.x + self.w // 2 - text_size[0][0] // 2),
                    int(self.y + self.h // 2 + text_size[0][1] // 2))
        cv2.putText(img, self.text, text_pos, fontFace, fontScale, text_color, thickness)

        # Put special characters
        if self.special_characters:
            char_size = cv2.getTextSize(self.special_characters, fontFace, fontScale / 2, thickness)
            char_pos = (int(self.x + self.w // 2 - char_size[0][0] // 2),
                        int(self.y + self.h // 2 - text_size[0][1] // 2 - 5))
            cv2.putText(img, self.special_characters, char_pos, fontFace, fontScale / 2, text_color, thickness)

    def isOver(self, x, y):
        if (self.x + self.w > x > self.x) and (self.y + self.h > y > self.y):
            return True
        return False

# Create keys
w, h = 80, 60
startX, startY = 100, 100  # Adjusted for starting placement

# Rows of keys
rows = [
    "1234567890",
    "qwertyuiop",
    "asdfghjkl",
    "zxcvbnm",
    "",

    "!@#$%^&*()"
]

keys = []

for i, row in enumerate(rows):
    row_len = len(row)
    for j, char in enumerate(row):
        x = startX + (w + 5) * j
        y = startY + (h + 5) * i
        special_characters = None
        '''if i == 0 and j < 10:
            special_characters = "!@#$%^&*()"[j]'''
        keys.append(Key(x, y, w, h, char, special_characters))

#keys.append(Key(startX + 9 * w + 10, startY + 3 * h + 10, w, h, "Caps"))
keys.append(Key(startX + -1 * w + 15, startY + 4 * h + 10, 2 * w, h, "Win"))
keys.append(Key(startX+130, startY+4*h+15, 4*w, h, "Space"))
keys.append(Key(startX+10*w + 50, startY+2*h+10, w, h, "clr"))
keys.append(Key(startX+7*w+10, startY+4*h+15, 2*w, h, "<--"))

showKey = Key(300,5,80,50, 'Show')
exitKey = Key(300,65,80,50, 'Exit')
textBox = Key(startX, startY-h-5, 10*w+9*5, h,'')

# Open camera
cap = cv2.VideoCapture(0)
cap.set(3, 1920)  # Set width
cap.set(4, 1080)  # Set height

caps_lock_on = False

# Main loop
while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

            # Get the index fingertip location
            index_tip = (
                int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image.shape[1]),
                int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image.shape[0]))

            # Check if index is over a key
            for k in keys:
                if k.isOver(index_tip[0], index_tip[1]):
                    k.pressed = True
                    if k.start_time == 0:
                        k.start_time = time.time()
                    elif time.time() - k.start_time > 1:
                        print(f"Pressed key: {k.text}")
                        if k.text == '<--':
                            # Backspace functionality
                            if len(textBox.text) > 0:
                                textBox.text = textBox.text[:-1]
                        elif k.text == 'Space':
                            # Space functionality
                            textBox.text += " "
                        elif k.text == 'Caps':
                            # Caps Lock functionality
                            caps_lock_on = not caps_lock_on
                        elif k.text == 'Win':
                            # Windows key functionality (for example, open the Start menu)
                            pyautogui.hotkey('winleft')
                        else:
                            if caps_lock_on or k.text.isnumeric():
                                textBox.text += k.special_characters if k.special_characters else k.text
                                # Simulating the press of the actual keyboard
                                pyautogui.write(k.special_characters if k.special_characters else k.text.lower())
                            else:
                                textBox.text += k.text.lower()
                                # Simulating the press of the actual keyboard
                                pyautogui.write(k.text.lower())

                        k.start_time = 0
                else:
                    k.pressed = False
                    k.start_time = 0

    # Draw virtual keyboard
    for k in keys:
        k.drawKey(image)

    # Draw text box
    textBox.drawKey(image)

    # Draw Caps Lock status
    #caps_status = "ON" if caps_lock_on else "OFF"
    #cv2.putText(image, f"Caps Lock: {caps_status}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Flip the image horizontally for a selfie-view display.
    image = cv2.resize(image, (1920, 1080))
    cv2.imshow('Virtual Keyboard', image)

    # Check for key press to exit
    pressed_key = cv2.waitKey(1)
    if pressed_key == 27:  # 27 corresponds to the ESC key
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()

##################################################################################################

import mediapipe as mp
import cv2 as cv
import time


class HandDetector:
    """
    HandDetector class uses MediaPipe to detect and track hands in video frames.

    Attributes:
    - mode (bool): Static mode or dynamic mode for the hand detection.
    - maxHands (int): Maximum number of hands to detect.
    - complexity (int): Complexity level of the hand landmarks model.
    - detectionCon (float): Minimum detection confidence threshold.
    - trackCon (float): Minimum tracking confidence threshold.
    """

    def __init__(self, mode=False, maxHands=2, complexity=1, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.complexity = complexity
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        # Initialize MediaPipe hands and drawing utilities
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            self.mode, self.maxHands, self.complexity, self.detectionCon, self.trackCon
        )
        self.mpDraw = mp.solutions.drawing_utils
        self.tip_IDS = [4, 8, 12, 16, 20]  # Indices of fingertips

    def findHands(self, frame, draw=True):
        """
        Detects hands in the provided frame and draws landmarks if specified.

        Args:
        - frame (ndarray): The input image in BGR format.
        - draw (bool): Whether to draw the landmarks on the frame.

        Returns:
        - frame (ndarray): The processed frame with landmarks drawn if specified.
        """
        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)  # Convert BGR to RGB for MediaPipe
        self.results = self.hands.process(frame_rgb)  # Process the RGB frame to detect hands

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(frame, handLms, self.mpHands.HAND_CONNECTIONS)

        return frame

    def findPositions(self, frame, handNo=0, draw=False):
        """
        Finds and returns the dictionary of landmark positions for the specified hand.

        Args:
        - frame (ndarray): The input image.
        - handNo (int): The index of the hand (default is 0 for the first detected hand).
        - draw (bool): Whether to draw circles on landmarks.

        Returns:
        - landmarks (dict): Dictionary of landmark positions with landmark ID as keys and coordinates as values.
        """
        self.landmarks = {}
        if self.results.multi_hand_landmarks:
            myhand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myhand.landmark):
                h, w, _ = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.landmarks[id] = (cx, cy)

                if draw:
                    cv.circle(frame, (cx, cy), 10, (255, 0, 255), cv.FILLED)

        return self.landmarks

    def fingersUp(self):
        """
        Determines which fingers are up and returns their status.

        Returns:
        - fingers (list): List of integers (0 or 1) representing if the fingers are up (1) or down (0).
        """
        if len(self.landmarks) == 0:
            return []

        fingers = []

        # Determine if it is a left or right hand
        wrist_x = self.landmarks[0][0]  # Wrist x-coordinate
        thumb_x = self.landmarks[self.tip_IDS[0]][0]  # Thumb tip x-coordinate

        # Check thumb status
        if thumb_x > wrist_x:
            # Right hand (Thumb is to the right of the wrist)
            fingers.append(1 if self.landmarks[self.tip_IDS[0]][0] > self.landmarks[self.tip_IDS[0] - 2][0] else 0)
        else:
            # Left hand (Thumb is to the left of the wrist)
            fingers.append(1 if self.landmarks[self.tip_IDS[0]][0] < self.landmarks[self.tip_IDS[0] - 2][0] else 0)

        # Check the status of the other four fingers
        for ID in range(1, 5):
            fingers.append(1 if self.landmarks[self.tip_IDS[ID]][1] < self.landmarks[self.tip_IDS[ID] - 2][1] else 0)

        return fingers

    def release(self):
        """
        Releases the MediaPipe hand detection resources.
        """
        self.hands.close()


def main():
    pTime = 0
    cap = cv.VideoCapture(0)  # Capture video from the default camera
    detector = HandDetector()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = detector.findHands(frame)
        landmarks = detector.findPositions(frame)
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        frame = cv.flip(frame, 1)  # Flip frame horizontally
        cv.putText(frame, f'FPS: {int(fps)}', (10, 50), cv.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

        cv.imshow('Hands', frame)

        # Exit when 'p' key is pressed
        if cv.waitKey(1) & 0xFF == ord('p'):
            break

    # Release resources
    cap.release()
    detector.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()