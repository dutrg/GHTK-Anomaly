import cv2

vid = cv2.VideoCapture("demovideo1_test.mp4")

fps = vid.get(cv2.CAP_PROP_FPS)

print(fps)

vid.set(cv2.CAP_PROP_POS_FRAMES, 153)

ret, frame = vid.read()

cv2.imshow("frame", frame);
cv2.waitKey(0)
cv2.imwrite('frame2.png', frame)

