from cmath import inf
from re import template
from PIL import Image, ImageFilter
import numpy as np
import shutil
import math
import cv2
import os


template = cv2.imread("template.png", 0)
train = cv2.imread("IC_1_thresh.png")
height, width, _ = train.shape
img_gray = cv2.cvtColor(train, cv2.COLOR_BGR2GRAY)
w, h = template.shape[::-1]
res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
threshold = 0.8
loc = np.where(res >= threshold)
rLocations = zip(*loc[::-1])
firstR = list(rLocations)[0]
firstRw = firstR[0]
firstRh = firstR[1]
cropHeight = firstRh
cropWidth = 0
# print(fw, fh)
brainHeight = inf
brainWidth = inf
for pt in zip(*loc[::-1]):
    # print(pt[0], pt[1])
    if firstRw == pt[0] and firstRh == pt[1]:
        continue
    elif firstRw == pt[0]:
        brainHeight = min(brainHeight, pt[1] - firstRh)
    elif firstRh == pt[1]:
        cropWidth = max(cropWidth, pt[0])
        brainWidth = min(brainWidth, pt[0] - firstRw)

print(brainWidth, brainHeight, cropWidth + brainWidth, cropHeight)
train = train[cropHeight:height, 0:cropWidth+brainWidth]
croppedHeight, croppedWidth, _ = train.shape
cv2.imshow("Image", train)
cv2.waitKey(0)
cv2.destroyAllWindows()

print(croppedHeight, croppedWidth, brainHeight, brainWidth)
imgNo = 0
ch = 0
while ch + brainHeight < croppedHeight:
    cw = 0
    while cw + brainWidth < croppedWidth:
        imgNo += 1
        print(ch + 10, ch + brainHeight, cw, cw + brainWidth)
        img = train[ch + 6:ch + brainHeight, cw:cw + brainWidth]
        cw += brainWidth
        cv2.imwrite(str(imgNo) + ".png", img)
    ch += brainHeight
        
# for h in range(0, height, brainHeight):
#     for w in range(0, width, brainWidth):
#         imgNo += 1
#         print(h, h+brainHeight, w, w+brainWidth)
#         img = train[h:h+brainHeight, w+4:w+brainWidth]
#         cv2.imshow("Image", img)
#         # cv2.imwrite(str(imgNo) + ".png", img)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()