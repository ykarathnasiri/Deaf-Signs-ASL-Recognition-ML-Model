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
        prediction, index = classifier.getPrediction(imgWhite)
        print(prediction, index)

    else:
        k = imgSize / imgCropShape[1]
        hCal = math.ceil(k * imgCropShape[0])
        imgResize = cv2.resize(imgCrop, (imgSize, hCal))
        hGap = math.ceil((imgSize - hCal) / 2)
        imgWhite[hGap + offset[0]:hCal + hGap + offset[0], :] = imgResize
        prediction, index = classifier.getPrediction(imgWhite)
        print(prediction, index)

    return imgWhite


cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2)  # Detect up to 2 hands
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")

offset = 20
imgSize = 300

labels = ["A", "B", "C", "D", "E", "F", "G", "H", "Hello", "I Love You", "Okay", "Sorry", "Thanks",]

while True:
    success, img = cap.read()
    imgOutput = img.copy()
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
            imgWhite = draw_hand(imgCrop, imgWhite, imgSize, offset= (0, 0))

            # Show prediction and bounding box for both hands
            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            label_text = labels[index]
            (label_width, label_height), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_COMPLEX, 1.7, 2)

            label_width += 10
            label_height += 10

            cv2.rectangle(imgOutput, (x - offset, y - offset - label_height - 10),
                          (x - offset + label_width, y - offset), (255, 0, 0), cv2.FILLED)

            cv2.putText(imgOutput, label_text, (x - offset, y - offset - 5), cv2.FONT_HERSHEY_COMPLEX, 1.7,
                        (255, 0, 255), 2)

            cv2.rectangle(imgOutput, (x - offset, y - offset),
                          (x + w + offset, y + h + offset), (255, 0, 255), 2)
            cv2.imshow("ImageWhite", imgWhite)

        elif len(hands) == 1:  # If one hand is detected
            hand = hands[0]
            x, y, w, h = hand['bbox']
            imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
            imgWhite = draw_hand(imgCrop, imgWhite, imgSize)
            cv2.imshow("ImageWhite", imgWhite)

            #Draw bounding box and prediction lable on imgOutput for single hand
            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            cv2.rectangle(imgOutput, (x - offset, y - offset - 50),
                          (x - offset + 90, y - offset - 50 + 50), (255, 0, 0), cv2.FILLED)
            cv2.putText(imgOutput, labels[index], (x, y - 26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 0, 255), 2)
            cv2.rectangle(imgOutput, (x - offset, y - offset),
                          (x + w + offset, y + h + offset), (255, 0, 255), 2)

            cv2.imshow("ImageWhite", imgWhite)

    cv2.imshow("Image", imgOutput)
    cv2.waitKey(1)