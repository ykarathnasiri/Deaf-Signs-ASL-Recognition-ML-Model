import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math


def draw_hand(imgCrop, imgWhite, imgSize, offset=(0, 0)):
    imgCropShape = imgCrop.shape
    aspectRatio = imgCropShape[0] / imgCropShape[1]

    if aspectRatio > 1:
        k = imgSize / imgCropShape[0]
        wCal = math.ceil(k * imgCropShape[1])
        imgResize = cv2.resize(imgCrop, (wCal, imgSize))
        wGap = math.ceil((imgSize - wCal) / 2)
        imgWhite[:, wGap + offset[1]:wCal + wGap + offset[1]] = imgResize
        prediction, index = classifier.getPrediction(img)
        print(prediction, index)

    else:
        k = imgSize / imgCropShape[1]
        hCal = math.ceil(k * imgCropShape[0])
        imgResize = cv2.resize(imgCrop, (imgSize, hCal))
        hGap = math.ceil((imgSize - hCal) / 2)
        imgWhite[hGap + offset[0]:hCal + hGap + offset[0], :] = imgResize

    return imgWhite


cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2)  # Detect up to 2 hands
classifier = Classifier("Model\keras_model.h5", "Model\labels.txt")

offset = 20
imgSize = 300

folder = "Data/A"
counter = 0

labels = ["A", "B", "C"]

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)

    if hands:
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255  # Create a white background

        if len(hands) == 2:  # If both hands are detected
            hand1 = hands[0]
            hand2 = hands[1]
            x1, y1, w1, h1 = hand1['bbox']
            x2, y2, w2, h2 = hand2['bbox']

            # Combine the bounding boxes of both hands
            x = min(x1, x2)
            y = min(y1, y2)
            w = max(x1 + w1, x2 + w2) - x
            h = max(y1 + h1, y2 + h2) - y

            # Crop the frame with both hands
            imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
            imgWhite = draw_hand(imgCrop, imgWhite, imgSize)
            cv2.imshow("ImageWhite", imgWhite)

        elif len(hands) == 1:  # If one hand is detected
            hand = hands[0]
            x, y, w, h = hand['bbox']
            imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
            imgWhite = draw_hand(imgCrop, imgWhite, imgSize)
            cv2.imshow("ImageWhite", imgWhite)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
