import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

import warpImages

imagePath = './data/opencv-stitching/'
#imageNames = ['a1.png', 'a2.png', 'a3.png']
imageNames = ['boat1.jpg', 'boat2.jpg', 'boat3.jpg', 'boat4.jpg', 'boat5.jpg', 'boat6.jpg']
#imageNames = ['newspaper1.jpg', 'newspaper2.jpg', 'newspaper3.jpg', 'newspaper4.jpg']

images = []

#append images
for name in imageNames:
    images.append(cv.imread(imagePath + name))

# Initialize SIFT detector
sift = cv.SIFT_create()

keypoints_and_descriptors = []

for image in images:
    # Find the keypoints and descriptors with SIFT
    kp, des = sift.detectAndCompute(image, None)
    keypoints_and_descriptors.append({kp: kp, des: des})


keypoints_and_descriptors