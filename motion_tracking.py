import cv2
import numpy as np


cap = cv2.VideoCapture("demovideo1_test.mp4")
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

ret, frame1 = cap.read()
ret, frame2 = cap.read()
print(frame1.shape)

area = np.zeros((frame_height, frame_width), np.float32)

while cap.isOpened():
    diff = cv2.absdiff(frame1, frame2)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=3)
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)

        if cv2.contourArea(contour) < 2000:
            continue
        cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # for y in range(y, min(y + h, frame_height)):
        #     for x in range(x,min(x + w, frame_width)):
        #         area[y, x] = 1           

    # cv2.rectangle(area, (50, 50), (250, 200), (0, 0, 255), 2)
    cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)
    image = cv2.resize(frame1, (1280, 720))
    cv2.imshow("feed", frame1)
    frame1 = frame2
    ret, frame2 = cap.read()

    if cv2.waitKey(25) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
cap.release()
