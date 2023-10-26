import math

import cv2
import imutils
import numpy as np

import blendImages


def detect_and_match_features(img1, img2, fe_algo="sift", match_algo="flann"):
    if fe_algo == "orb":
        keypoints1, descriptors1, keypoints2, descriptors2 = orb_feature_extraction(img1, img2)
    else:
        keypoints1, descriptors1, keypoints2, descriptors2 = sift_feature_extraction(img1, img2)

    if match_algo == "bf":
        matches = brute_force_matcher(descriptors1, descriptors2)
    else:
        matches = flann_matcher(descriptors1, descriptors2)

    good, matchedMask = lowe_ratio_test(matches)

    return keypoints1, keypoints2, matches, good, matchedMask


def sift_feature_extraction(img1, img2):
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    return kp1, des1, kp2, des2


def orb_feature_extraction(img1, img2):
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    return kp1, des1, kp2, des2


def flann_matcher(des1, des2):
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)

    return matches


def brute_force_matcher(des1, des2):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    return matches


def lowe_ratio_test(matches):
    good = []
    matchesMask = [[0, 0] for i in range(len(matches))]

    for i, (m, n) in enumerate(matches):
        if m.distance < 0.7 * n.distance:
            good.append(m)
            matchesMask[i] = [1, 0]

    return good, matchesMask


def draw_matches(img1, img2, kp1, kp2, matches, matchesMask, filename="drawn"):
    draw_params = dict(matchColor=(0, 255, 0),
                       singlePointColor=(255, 0, 0),
                       matchesMask=matchesMask,
                       flags=cv2.DrawMatchesFlags_DEFAULT)

    img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, **draw_params)
    cv2.imwrite(filename + "_matches.png", img3)


def estimate_homography(keypoints1, keypoints2, matches, threshold=5):
    src_points = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_points = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC, threshold)
    return H, mask


def draw_inliers(img1, img2, kp1, kp2, matches, mask, filename="inliers.png"):
    matchesMask = mask.ravel().tolist()
    draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                       singlePointColor=None,
                       matchesMask=matchesMask,  # draw only inliers
                       flags=2)
    img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, **draw_params)
    cv2.imwrite(filename, img3)


def warp_images(img1, img2, H):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    corners1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
    corners2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
    warped_corners2 = cv2.perspectiveTransform(corners2, H)

    corners = np.concatenate((corners1, warped_corners2), axis=0)
    [xmin, ymin] = np.int32(corners.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(corners.max(axis=0).ravel() + 0.5)

    t = [-xmin, -ymin]
    Ht = np.array([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]])

    warped_img1 = cv2.warpPerspective(img2, Ht @ H, (xmax - xmin, ymax - ymin))
    warped_img2 = cv2.warpPerspective(img2, Ht @ H, (xmax - xmin, ymax - ymin))
    warped_img2[t[1]:h1 + t[1], t[0]:w1 + t[0]] = img1
    warped_img = warped_img2

    # create partitions for blending function
    partition_1 = cv2.warpPerspective(img2, Ht @ H, (xmax - xmin, ymax - ymin))
    partition_2 = np.zeros(partition_1.shape)
    partition_2 = partition_2.astype(partition_1.dtype)
    partition_2[t[1]:h1 + t[1], t[0]:w1 + t[0]] = img1

    return warped_img, partition_1, partition_2


def stitch_two_images(img1, img2):
    print("Stitching two images")
    keypoints1, keypoints2, matches, good, _ = detect_and_match_features(img1, img2)
    H, mask = estimate_homography(keypoints1, keypoints2, good)
    warped_img, partition_1, partition_2 = warp_images(img2, img1, H)
    print("Warping done")

    result, blending_function, gradient, overlap_mask = blendImages.blend_images(partition_1, partition_2)

    return result


def resize_and_pad(img, w, h):
    # middle = imutils.resize(img, width=int(img.shape[1] * 1.2))
    hor = int((w / 2 - img.shape[1] / 2))
    vert = int((h / 2 - img.shape[0] / 2))
    middle = cv2.copyMakeBorder(img, vert, vert, hor, hor, 0)
    return middle


def stitcher(imgs):
    temp_img = imgs[0]
    for i in range(1, len(imgs)):
        if i == len(imgs):
            break

        temp_img = stitch_two_images(temp_img, imgs[i])
        temp_img = cv2.normalize(temp_img, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')

    cv2.imwrite("output/img_before_rotate.png", temp_img)
    img1 = temp_img

    middle_index = math.floor(len(imgs) / 2)
    img2 = resize_and_pad(imgs[middle_index], img1.shape[1], img1.shape[0])

    keypoints1, keypoints2, matches, good, _ = detect_and_match_features(img1, img2)
    H, _ = estimate_homography(keypoints1, keypoints2, good)
    width, height = img2.shape[1], img2.shape[0]
    result = cv2.warpPerspective(img1, H, (width, height))

    return result
