import cv2
import numpy as np

img1 = cv2.imread("frame1.png", 0)
img2 = cv2.imread("frame2.png", 0)

def get_mask(frame1, frame2, kernel = np.ones((3,3), np.uint8)):

    frame_diff = cv2.absdiff(img1, img2)

    frame_diff = cv2.medianBlur(frame_diff, 3)

    mask = cv2.adaptiveThreshold(frame_diff, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV, 11, 3)

    mask = cv2.medianBlur(mask, 3)

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations = 1)

    return mask

def get_contour(mask, thresh = 400):
    contours, ret = cv2.findContours(mask, 
                                   cv2.RETR_EXTERNAL, # cv2.RETR_TREE, 
                                   cv2.CHAIN_APPROX_TC89_L1)
    detection = []
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        area = w*h
        if area > thresh:
            detection.append([x,y,w+x, y+h, area])
    return np.array(detection)


mask = get_mask(img1, img2)

box = get_contour(mask)
for i in box:
    print(i)

grayscale_diff = cv2.absdiff(img1, img2)
cv2.imshow("1",grayscale_diff)
cv2.imshow("2", mask)
cv2.waitKey(0)

