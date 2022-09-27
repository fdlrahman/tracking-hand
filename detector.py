import cv2
import numpy as np
import os
import time
import HandTrackingModule as htm

cap = cv2.VideoCapture(0)

folderPath = 'FingerImages'
myList = os.listdir(folderPath)
myList.sort()

overlayList = []

for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')

    overlayList.append(image)

pTime = 0

detector = htm.handDetector(detectionCon=0.8)

tipIds = [4, 8, 12, 16, 20]

while True:
    success, img = cap.read()

    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)

    fingers = []

    if (len(lmList) != 0):

        if lmList[tipIds[0]][2] < (lmList[tipIds[0] - 1][2] - 10):
            fingers.append(1)
        else:
            fingers.append(0)

        for id in range(1, 5):
            if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        count = fingers.count(1)

        h, w, _ = overlayList[count].shape
        img[0:h, 0:w] = overlayList[count]

        cv2.rectangle(img, (20, 225), (170, 425), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, str(count), (45, 375), cv2.FONT_HERSHEY_PLAIN,
                    10, (255, 0, 0), 25)

    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime

    cv2.putText(img, f'FPS: {int(fps)}', (1700, 70),
                cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    cv2.imshow('Image', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
