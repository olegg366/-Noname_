import cv2
import numpy as np
import cvzone
from cvzone.PoseModule import PoseDetector
from tqdm import trange
import time
import matplotlib.pyplot as plt
import math
import os

def angle_legs(lmlist):
    if len(lmlist) != 0:
        point1, point2 = lmlist[31], lmlist[32]
        point3, point4 = lmlist[23], lmlist[24]

        x1, y1 = point1[1:-1]
        x2, y2 = point2[1:-1]
        x3, y3 = point3[1:-1]
        x4, y4 = point4[1:-1]

        x3 = (x3 + x4) / 2
        y3 = (y3 + y4) / 2

        a = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        b = math.sqrt((x2 - x3) ** 2 + (y2 - y3) ** 2)
        c = math.sqrt((x1 - x3) ** 2 + (y1 - y3) ** 2)

        angle = math.acos((c ** 2 + b ** 2 - a ** 2) / (2 * c * b))

        return int(angle > 0.68)

def classify(video):
    try:
        lenn = 0
        s = 0

        ret, img = video.read()
        if not ret: return
        pose.findPose(img, draw=1)
        lmlist, bbox = pose.findPosition(img, draw=0)

        xl = lmlist[15][2]
        xc = (lmlist[12][2] + lmlist[11][2]) // 2

        while xl * 1.1 >= xc:
            ret, img = video.read()
            if not ret: break
            pose.findPose(img, draw=1)
            lmlist, bbox = pose.findPosition(img, draw=0)

            img = cv2.resize(img, (850, 500))
            cv2.imshow('frame', img)
            cv2.waitKey(1)

            xl = lmlist[15][2]
            xc = (lmlist[12][2] + lmlist[11][2]) // 2
            val = angle_legs(lmlist)
            if val != None:
                lenn += 1
                s += val
        yl = lmlist[15][1]
        yc = (lmlist[12][1] + lmlist[11][1]) // 2
        while True:
            ret,  img = video.read()
            if not ret: break
            pose.findPose(img, draw=1)
            lmlist, bbox = pose.findPosition(img, draw=0)

            img = cv2.resize(img, (850, 500))
            cv2.imshow('frame', img)
            cv2.waitKey(1)

            val = angle_legs(lmlist)
            if val != None:
                lenn += 1
                s += val
        if val < 0.68:
            if yl < yc: return 'right'
            else: return 'left'
        else: return 'front'
    finally:
        video.release()
        cv2.destroyAllWindows()

pose = PoseDetector(trackCon=.7, detectionCon=.7)

path=r'train_dataset_Синтез\raw\Lefty\REC177522980354433669.mp4'

video = cv2.VideoCapture(path)

print('This swing is made from', classify(video))



# pose = PoseDetector(trackCon=.7, detectionCon=.7)
#
# l1 = r"train_dataset_Синтез\face_on_videos\\"
# l2 = r'train_dataset_Синтез\raw\Lefty\\'
# l3 = r'train_dataset_Синтез\raw\Righty\\'
#
# check = []
#
# for l in l2, l3:
#     ls = os.listdir(l)
#     for i in trange(10):
#         r = np.random.randint(0, len(ls))
#         path = l + ls[r]
#         video = cv2.VideoCapture(path)
#         res = classify(video)
#         if (res == 0 and l == l2) or (res == 1 and l == l3): check.append(1)
#         else: check.append(0)
# s = 0
# for elem in check: s += elem
# print(s / len(check) * 100)
