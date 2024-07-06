import cv2

img1 = cv2.imread("frame1.png", 0)
img2 = cv2.imread("frame2.png", 0)

grayscale_diff = cv2.absdiff(img1, img2)

cv2.imshow("1",grayscale_diff)
cv2.imshow("2", img2)
cv2.waitKey(0)