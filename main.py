import screen_brightness_control as sbc
import mediapipe as mp
import cv2
import numpy as np
from math import hypot

cam_clip = cv2.VideoCapture(0)
my_Hands = mp.solutions.hands
hands = my_Hands.Hands()
Hand_straight_line_draw = mp.solutions.drawing_utils

while True:
    success, img = cam_clip.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    lmList = []
    if results.multi_hand_landmarks:
        for handlandmark in results.multi_hand_landmarks:
            for id, lm in enumerate(handlandmark.landmark):
                h, w, _ = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
            Hand_straight_line_draw.draw_landmarks(img, handlandmark, my_Hands.HAND_CONNECTIONS)

    if lmList != []:
        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]

        cv2.circle(img, (x1, y1), 4, (255, 0, 0), cv2.FILLED)
        cv2.circle(img, (x2, y2), 4, (255, 0, 0), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)

        length = hypot(x2 - x1, y2 - y1)

        bright = np.interp(length, [15, 220], [0, 100])
        print(bright, length)
        sbc.set_brightness(int(bright))

        # Hand range 15 - 220
        # Brightness range 0 - 100

    cv2.imshow('Image', img)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break

