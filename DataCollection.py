import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time
import os

def draw_hand(imgCrop, imgWhite, imgSize, offset=(0,0)):
    imgCropShape =imgCrop.shape
    aspectRatio = imgCropShape [0] / imgCropShape[1]

    if aspectRatio > 1:
            k = imgSize / imgCropShape[0]
            wCal = math.ceil(k * imgCropShape[1])
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap + offset[1]: wCal + wGap + offset[1]] = imgResize

        else:
            k = imgSize / imgCropShape[1]
            hCal = math.ceil(k * imgCropShape[0])
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap + offset[0]:hCal + hGap + offset[0], :] = imgResize

        return imgWhite


cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2) #detect up to 2 hands

offset = 20
imgSize = 300

folder = "Data/A"
counter = 0

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)

    if hands:
        imgWhite

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == ord("s"):
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
        print(counter)