import cv2 as cv
import numpy as np
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import HandTrackingModule as htm

# Initializing Audio Interface
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)
min_vol, max_vol = volume.GetVolumeRange()[:2]

# Initializing Webcam and Hand Detector
cap = cv.VideoCapture(0)
if not cap.isOpened():
    raise Exception("Couldn't open the webcam")

detector = htm.HandDetector(detectionCon=0.7, trackCon=0.7)
vol, vol_bar, vol_per = 0, 400, 0

while cap.isOpened():
    # Reading frame from webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Detecting hands and finding positions
    frame = detector.findHands(frame, draw=True)
    landmarks = detector.findPositions(frame)

    if landmarks:
        # Getting coordinates of thumb and index fingertips
        x1, y1 = landmarks[4][:2]
        x2, y2 = landmarks[8][:2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        # Visualizing landmarks and connection
        for (x, y) in [(x1, y1), (x2, y2), (cx, cy)]:
            # Drawing circles for fingertips and center
            cv.circle(frame, (x, y), 10, (255, 0, 255) if (x, y) != (cx, cy) else (0, 255, 0), cv.FILLED)
        # Drawing line between thumb and index finger
        cv.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)

        # Calculating distance between thumb and index finger and mapping it to volume range
        length = np.linalg.norm([x2 - x1, y2 - y1])
        vol = np.interp(length, [20, 200], [min_vol, max_vol])
        vol_bar = np.interp(length, [20, 200], [400, 150])
        vol_per = np.interp(length, [20, 200], [0, 100])

        # Setting system volume level based on hand distance
        volume.SetMasterVolumeLevel(vol, None)

    # Flipping frame horizontally for mirror effect
    frame = cv.flip(frame, 1)

    # Determining color for volume bar based on percentage
    bar_color = (0, 255, 0) if vol_per <= 70 else (0, 0, 255)

    # Drawing volume bar and displaying percentage
    cv.rectangle(frame, (50, 150), (85, 400), bar_color, 3)
    cv.rectangle(frame, (50, int(vol_bar)), (85, 400), bar_color, cv.FILLED)
    cv.putText(frame, f"{int(vol_per)} %", (45, 140), cv.FONT_HERSHEY_COMPLEX, 1.25, (255, 255, 255), 2)

    # Displaying frame
    cv.imshow("Hand", frame)

    # Breaking loop on pressing 'p'
    if cv.waitKey(1) & 0xFF == ord('p'):
        break

# Releasing webcam and closing windows
cap.release()
cv.destroyAllWindows()

##########################################################################

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
