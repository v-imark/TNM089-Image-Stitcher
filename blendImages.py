import cv2
import numpy as np
import imutils

debugCircles = False


def blend_images(warped_image_1, warped_image_2):
    print("Start blending")

    # create masks for images
    mask1 = cv2.cvtColor(warped_image_1, cv2.COLOR_RGB2GRAY)
    mask2 = cv2.cvtColor(warped_image_2, cv2.COLOR_RGB2GRAY)
    # threshold the masks
    threshhold1 = cv2.threshold(mask1, 0, 255, cv2.THRESH_BINARY)[1]
    threshhold2 = cv2.threshold(mask2, 0, 255, cv2.THRESH_BINARY)[1]

    # get overlap of masks
    overlap_mask = cv2.bitwise_and(threshhold1, threshhold2)

    # get average position of white pixels in threshholded images
    white_pixels1 = np.where(threshhold1 == 255)
    white_pixels2 = np.where(threshhold2 == 255)
    average_position_1 = np.flip(np.average(white_pixels1, axis=1))
    average_position_2 = np.flip(np.average(white_pixels2, axis=1))
    # average_position_1 = [average_position_1[0], average_position_1[1] + 1000]

    print("average_position_1: ", average_position_1)
    print("average_position_2: ", average_position_2)

    # make green circle at average position of white pixels
    if debugCircles:
        cv2.circle(warped_image_1, (int(average_position_1[0]), int(average_position_1[1])), 10, (0, 255, 0), -1)
        cv2.circle(warped_image_2, (int(average_position_2[0]), int(average_position_2[1])), 10, (0, 255, 0), -1)

    # get direction vector from average_position_1 to average_position_2
    gradient_direction = average_position_1 - average_position_2
    gradient_direction = gradient_direction / np.linalg.norm(gradient_direction)  # doesnt matter

    # generate 2d image with a smooth linear gradient in the direction of the "gradient_direction" vector
    gradient = np.zeros((warped_image_1.shape[0], warped_image_1.shape[1]), dtype=np.float32)
    for i in range(gradient.shape[0]):
        for j in range(gradient.shape[1]):
            gradient[i, j] = np.dot([j, i], gradient_direction)

    # remap values of blending function to range [0, 1]
    gradient = gradient - gradient.min()
    gradient = gradient / gradient.max()

    # multiply gradient with overlap
    blending_function = gradient * overlap_mask

    # normalize blending function
    blending_function = blending_function - np.min(blending_function[np.nonzero(blending_function)])
    blending_function = blending_function / blending_function.max()
    blending_function = blending_function

    # make blending_function mask an rgb image
    overlap_mask = np.stack((overlap_mask, overlap_mask, overlap_mask), axis=2)
    blending_function = np.stack((blending_function, blending_function, blending_function), axis=2)

    # compile the final blend

    # mask warped_image_1 with overlap_mask
    compiled_image = warped_image_1 * ((255 - overlap_mask) / 255)
    compiled_image += warped_image_2 * ((255 - overlap_mask) / 255)

    compiled_image += warped_image_1 * blending_function * (overlap_mask / 255)
    compiled_image += warped_image_2 * (1 - blending_function) * (overlap_mask / 255)

    print("Blending finished")

    result = compiled_image
    return result, 255 * blending_function, 255 * gradient, overlap_mask
