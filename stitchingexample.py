import cv2

imagePath = './data/opencv-stitching/'
# imageNames = ['a1.png', 'a2.png', 'a3.png']
imageNames = ['boat1.jpg', 'boat2.jpg', 'boat3.jpg', 'boat4.jpg']
# imageNames = ['newspaper1.jpg', 'newspaper2.jpg', 'newspaper3.jpg', 'newspaper4.jpg']

images = []

# append images
for name in imageNames:
    images.append(cv2.imread(imagePath + name))
    images[-1] = cv2.resize(images[-1], (0, 0), fx=0.5, fy=0.5)  # optional resize step

### https://www.geeksforgeeks.org/opencv-panorama-stitching/ ###

stitchy = cv2.Stitcher.create()
(dummy, output) = stitchy.stitch(images)

if dummy != cv2.STITCHER_OK:
    # checking if the stitching procedure is successful
    # .stitch() function returns a true value if stitching is
    # done successfully
    print("stitching ain't successful")
else:
    print('Your Panorama is ready!!!')

# final output
cv2.imwrite("opencv_boat_4.png", output)

