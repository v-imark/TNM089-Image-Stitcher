import cv2

from test2 import estimate_homography, warp_images, blend_images, detect_and_match_features

imagePath = './data/opencv-stitching/'
img1 = cv2.imread(imagePath + 'boat1.jpg')
img2 = cv2.imread(imagePath + 'boat3.jpg')


keypoints1, keypoints2, matches = detect_and_match_features(img1, img2)
H, mask = estimate_homography(keypoints1, keypoints2, matches)
warped_img = warp_images(img2, img1, H)
cv2.imwrite('output.png', warped_img)

img1 = cv2.imread('output.png')
img2 = cv2.imread(imagePath + 'boat2.jpg')

keypoints1, keypoints2, matches = detect_and_match_features(img1, img2)
H, mask = estimate_homography(keypoints1, keypoints2, matches)
warped_img = warp_images(img2, img1, H)
cv2.imwrite('output2.png', warped_img)

