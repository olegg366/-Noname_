import cv2
import numpy as np
import cvzone
from cvzone.PoseModule import PoseDetector
from tqdm import trange
import time
import matplotlib.pyplot as plt
import math
import os

# 12, 24, 26 - early extression for right
# 11, 23, 25 - early extraction for left
def angles(lmlist, p1, p2, p3, p4, p5, p6, drawpoints):

    if len(lmlist) != 0:
        point1, point2 = lmlist[p1], lmlist[p2]
        point3, point4 = lmlist[p3], lmlist[p4]
        point5, point6 = lmlist[p5], lmlist[p6]

        x1, y1 = point1[1:-1]
        x2, y2 = point2[1:-1]
        x3, y3 = point3[1:-1]
        x4, y4 = point4[1:-1]
        x5, y5 = point5[1:-1]
        x6, y6 = point6[1:-1]

        if drawpoints == True:
                cv2.circle(img,(x1,y1),10,(255,0,255),3)
                cv2.circle(img, (x1, y1), 10, (0,255, 0),3)
                cv2.circle(img, (x2, y2), 10, (255, 0, 255), 3)
                cv2.circle(img, (x2, y2), 10, (0, 255, 0), 3)
                cv2.circle(img, (x3, y3), 10, (255, 0, 255), 3)
                cv2.circle(img, (x3, y3), 10, (0, 255, 0), 3)
                cv2.circle(img, (x4, y4), 10, (255, 0, 255), 3)
                cv2.circle(img, (x4, y4), 10, (0, 255, 0), 3)
                cv2.circle(img, (x5, y5), 10, (255, 0, 255), 3)
                cv2.circle(img, (x5, y5), 10, (0, 255, 0), 3)
                cv2.circle(img, (x6, y6), 10, (255, 0, 255), 3)
                cv2.circle(img, (x6, y6), 10, (0, 255, 0), 3)

                # dist between
                cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)
                cv2.line(img, (x2,y2), (x3, y3), (0, 0, 255), 2)
                cv2.line(img, (x4, y4), (x5, y5), (0, 0, 255), 2)
                cv2.line(img, (x5, y5), (x6, y6), (0, 0, 255), 2)
                cv2.line(img, (x1, y1), (x4, y4), (0, 0, 255), 2)


                lefthandangle = math.degrees(math.atan2(y3-y2, x3-x2)-
                                             math.atan2(y1-y2, x1-x2))
                
                rightdangle = math.degrees(math.atan2(y6-y5, x6-x5)-
                                             math.atan2(y4-y5, x4-x5))
        
                print(lefthandangle, rightdangle)

def distance(lmlist, p1, p2, p3, p4, p5, p6):
    if len(lmlist) != 0:
        point1, point2 = lmlist[p1], lmlist[p2]
        point3, point4 = lmlist[p3], lmlist[p4]
        point5, point6 = lmlist[p5], lmlist[p6]

        x1, y1 = point1[1:-1]
        x2, y2 = point2[1:-1]
        x3, y3 = point3[1:-1]
        x4, y4 = point4[1:-1]
        x5, y5 = point5[1:-1]
        x6, y6 = point6[1:-1]

        cx1 = (x1 + x2 + x3) / 3
        cy1 = (y1 + y2 + y3) / 3
        cx2 = (x4 + x5 + x6) / 3
        cy2 = (y4 + y5 + y6) / 3

        dist = math.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)

        return dist

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

        return angle

#path = r'train_dataset_Синтез\face_on_videos\image_picker429321974312927988.mp4'
#path = r'train_dataset_Синтез\raw\Lefty\REC177522980354433669.mp4'
#path = r'train_dataset_Синтез\raw\Righty\0c901a44-8916-45e0-860c-874774412596.mp4
pose = PoseDetector(trackCon=.7, detectionCon=.7)

l1 = r"train_dataset_Синтез\face_on_videos\\"
l2 = r'train_dataset_Синтез\raw\Lefty\\'
l3 = r'train_dataset_Синтез\raw\Righty\\'

plt.figure(figsize = (10, 10))

check = []

for l in l1, l2, l3:
    ls = os.listdir(l)

    ans = []

    for i in trange(5):
        lenn = 0
        s = 0

        r = np.random.randint(0, len(ls))

        path = l + ls[r]
        video = cv2.VideoCapture(path)

        while 1:
            ret, img = video.read()
            if not ret:
                video = cv2.VideoCapture(path)
                break
            img = cv2.resize(img, (510, 300))

            pose.findPose(img, draw=0)
            # all points

            lmlist, bbox = pose.findPosition(img, draw=0, bboxWithHands=0)

            # 20 is a right_index's point (arm)

            #angles(lmlist, 11, 13, 15, 12, 14, 16, drawpoints=1)

            val = angle_legs(lmlist)
            if val != None:
                lenn += 1
                s += val
        if (s / lenn > 0.68 and l != l1) or (s / lenn < 0.68 and l == l1): check.append(0)
        else: check.append(1)
s = 0
for elem in check: s += elem
print(s / len(check) * 100)



