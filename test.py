# Church 3 images
import cv2

from stitcher_utils import stitch_two_images

imagePath = './data/opencv-stitching/'
img1 = cv2.imread(imagePath + 'boat5.jpg')
img2 = cv2.imread(imagePath + 'boat6.jpg')


result = stitch_two_images(img1, img2)
cv2.imwrite('stitcher_boat_2_result.png', result)
