import cv2
import PIL
import imutils
from PIL import Image
from PIL import ImageOps

from stitcher_utils import stitcher

imagePath = './data/opencv-stitching/'

# Boat
img1 = cv2.imread(imagePath + 'boat1.jpg')
img2 = cv2.imread(imagePath + 'boat2.jpg')
img3 = cv2.imread(imagePath + 'boat3.jpg')
img4 = cv2.imread(imagePath + 'boat4.jpg')
img5 = cv2.imread(imagePath + 'boat5.jpg')

imgs = [img1, img2, img3, img4]
for i, img in enumerate(imgs):
    imgs[i] = cv2.resize(img, (int(img.shape[1] / 4), int(img.shape[0] / 4)))

#result = stitcher([imgs[0], imgs[1], imgs[2]])
#cv2.imwrite('stitcher_result_3.png', result)
#result = stitcher(imgs)
#cv2.imwrite('stitcher_result_4.png', result)

# Church 3 images
img1 = cv2.imread(imagePath + 'a1.png')
img2 = cv2.imread(imagePath + 'a2.png')
img3 = cv2.imread(imagePath + 'a3.png')

imgs = [img1, img2, img3]
result = stitcher(imgs)
cv2.imwrite('stitcher_church_result.png', result)
