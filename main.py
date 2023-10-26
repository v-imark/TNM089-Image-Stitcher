import cv2

from stitcher_utils import estimate_homography, warp_images, detect_and_match_features, draw_matches, draw_inliers, \
    stitch_two_images, stitcher
import blendImages

imagePath = './data/opencv-stitching/'

# Boat
img1 = cv2.imread(imagePath + 'boat1.jpg')
img2 = cv2.imread(imagePath + 'boat2.jpg')

# Feature extraction and matching
keypoints1, keypoints2, matches, good, matchedMask = detect_and_match_features(img1, img2)

# Draw keypoints on first pic and draw matches
kp_img = cv2.drawKeypoints(img1, keypoints1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imwrite('output/sift_keypoints.png', kp_img)
draw_matches(img1, img2, keypoints1, keypoints2, matches, matchedMask, filename="flann")
# draw_matches(img1, img2, keypoints1, keypoints2, good, matchedMask, filename="flann_good")

# Estimate homography with the good matches
H, mask = estimate_homography(keypoints1, keypoints2, good)

# Draw inliers
draw_inliers(img1, img2, keypoints1, keypoints2, good, mask, filename="output/boat_inliers.png")

warped_img = warp_images(img2, img1, H)

cv2.imwrite('output/boat_two_images_no_blend.png', warped_img[0])

# Boat
img1 = cv2.imread(imagePath + 'boat1.jpg')
img2 = cv2.imread(imagePath + 'boat2.jpg')
img3 = cv2.imread(imagePath + 'boat3.jpg')
img4 = cv2.imread(imagePath + 'boat4.jpg')
img5 = cv2.imread(imagePath + 'boat5.jpg')

imgs = [img1, img2, img3, img4]
for i, img in enumerate(imgs):
    imgs[i] = cv2.resize(img, (int(img.shape[1] / 4), int(img.shape[0] / 4)))

result = stitcher(imgs)
cv2.imwrite('stitcher_boat_4.png', result)

# Church 3 images
img1 = cv2.imread(imagePath + 'a1.png')
img2 = cv2.imread(imagePath + 'a2.png')
img3 = cv2.imread(imagePath + 'a3.png')

imgs = [img1, img2, img3]
result = stitcher(imgs)
cv2.imwrite('output/stitcher_church_result.png', result)
