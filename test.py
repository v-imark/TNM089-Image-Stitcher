import cv2
import numpy as np

import warpImages

imagePath = './data/opencv-stitching/'
image1 = cv2.imread(imagePath + 'boat1.jpg')
image2 = cv2.imread(imagePath + 'boat2.jpg')

sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(image1, None)
kp2, des2 = sift.detectAndCompute(image2, None)

matcher = cv2.BFMatcher()
matches = matcher.knnMatch(des2, des1, k=2)


def blend_images(img1, img2):
    mask = np.where(img1 != 0, 1, 0).astype(np.float32)
    blended_img = img1 * mask + img2 * (1 - mask)
    return blended_img.astype(np.uint8)


good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

if len(good_matches) > 10:
    src_pts = np.float32([kp2[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp1[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    warped_image = warpImages.warp_images(image1, image2, H)
    #warped_image = blend_images(warped_image, image2)
    # result = cv2.addWeighted(image1, 0.5, warped_image, 0.5, 1)

cv2.imwrite('Warped.png', warped_image)

# cv2.imwrite('result.png', result)
