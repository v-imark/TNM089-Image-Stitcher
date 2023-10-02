import cv2.detail
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

import warpImages

imagePath = './data/opencv-stitching/'
#imageNames = ['a1.png', 'a2.png', 'a3.png']
imageNames = ['boat1.jpg', 'boat2.jpg', 'boat3.jpg']
#  'boat3.jpg', 'boat4.jpg', 'boat5.jpg', 'boat6.jpg']
#imageNames = ['newspaper1.jpg', 'newspaper2.jpg', 'newspaper3.jpg', 'newspaper4.jpg']

images = []

# Append images
for name in imageNames:
    images.append(cv.imread(imagePath + name))

# Initialize SIFT detector
sift = cv.SIFT_create()

keypoints_and_descriptors = []

for image in images:
    # Find the keypoints and descriptors with SIFT
    kp, des = sift.detectAndCompute(image, None)
    keypoints_and_descriptors.append({"kp": kp, "des": des})

# Initialize flann-matcher
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)  # or pass empty dictionary
flann = cv.FlannBasedMatcher(index_params, search_params)

base = images[0]
base_kp = keypoints_and_descriptors[0].get("kp")
base_des = keypoints_and_descriptors[0].get("des")

print("Progress:", str(1) + " / " + str(len(images)))

MIN_MATCH_COUNT = 10

for i in range(1, len(images)):
    curr_kp = keypoints_and_descriptors[i].get("kp")
    curr_des = keypoints_and_descriptors[i].get("des")

    # Find matches
    matches = flann.knnMatch(base_des, curr_des, k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    # If enough matches are found, we extract the locations of matched keypoints in both the images.
    # They are passed to find the perspective transformation.
    # Once we get this 3x3 transformation matrix,
    # we use it to transform the corners of queryImage to corresponding points in trainImage. Then we draw it.
    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([base_kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([curr_kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
        base = warpImages.warp_images(images[i], base, M)

        print("Progress:", str(i + 1) + " / " + str(len(images)))

        if i < len(images) - 1:
            base_kp, base_des = sift.detectAndCompute(base, None)
    else:
        print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))


cv.imwrite('result.png', base)
