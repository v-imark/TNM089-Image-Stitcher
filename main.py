import cv2

from stitcher_utils import estimate_homography, warp_images, blend_images, detect_and_match_features, draw_matches, draw_inliers, \
    stitch_two_images
import blendImages

imagePath = './data/opencv-stitching/'

# Boat
img1 = cv2.imread(imagePath + 'boat1.jpg')
img2 = cv2.imread(imagePath + 'boat2.jpg')

# Feature extraction and matching
keypoints1, keypoints2, matches, good, matchedMask = detect_and_match_features(img1, img2)

# Draw keypoints on first pic and draw matches
kp_img = cv2.drawKeypoints(img1, keypoints1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imwrite('sift_keypoints.png', kp_img)
draw_matches(img1, img2, keypoints1, keypoints2, matches, matchedMask, filename="flann")
#draw_matches(img1, img2, keypoints1, keypoints2, good, matchedMask, filename="flann_good")

# Estimate homography with the good matches
H, mask = estimate_homography(keypoints1, keypoints2, good)

#Draw inliers
draw_inliers(img1, img2, keypoints1, keypoints2, good, mask, filename="boat_inliers.png")

warped_img = warp_images(img2, img1, H)

cv2.imwrite('boat_two_images.png', warped_img)

# Boat 3 images
img3 = cv2.imread(imagePath + 'boat3.jpg')
temp_img = stitch_two_images(img1, img3)
cv2.imwrite('temp.png', temp_img)

temp_img = cv2.imread('temp.png')
warped_img = stitch_two_images(temp_img, img2)
cv2.imwrite('boat_three_images.png', warped_img)

# Church 3 images
img1 = cv2.imread(imagePath + 'a1.png')
img2 = cv2.imread(imagePath + 'a2.png')
img3 = cv2.imread(imagePath + 'a3.png')

temp_img = stitch_two_images(img1, img3)
cv2.imwrite('temp.png', temp_img)

temp_img = cv2.imread('temp.png')
warped_img = stitch_two_images(temp_img, img2)
cv2.imwrite('church_three_images.png', warped_img)

