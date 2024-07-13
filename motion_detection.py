import cv2
import numpy as np


def get_mask(frame1, frame2, kernel=np.ones((3, 3), np.uint8)):

    frame_diff = cv2.absdiff(frame1, frame2)

    frame_diff = cv2.cvtColor(frame_diff, cv2.COLOR_BGR2GRAY)

    frame_diff = cv2.medianBlur(frame_diff, 3)

    mask = cv2.adaptiveThreshold(
        frame_diff, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 3
    )

    mask = cv2.medianBlur(mask, 3)

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    return mask


def get_contour(mask, thresh=50):
    contours, ret = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1  # cv2.RETR_TREE,
    )
    detection = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if area > thresh:
            detection.append([x, y, w + x, y + h, area])
    return np.array(detection)


def remove_overlap(boxes):
    check = np.array([True, True, False, False, False])
    keep = list(range(0, len(boxes)))
    for i in keep:
        for j in keep:
            if i != j and np.all((boxes[j, :4] >= boxes[i, :4]) == check[:4]):
                try:
                    keep.remove(j)
                except ValueError:
                    print("Error")
                    continue
    return keep


# def non_max_suppression(boxes, scores, threshold = 1e-1):
#     boxes = boxes[np.argsort(scores)[::-1]]

#     order = remove_overlap(boxes)

#     keep = []
#     while order:
#         i = order.pop(0)
#         keep.append(i)
#         for j in order:

#             intersection = max(0, min(boxes[i][2], boxes[j][2]) - max(boxes[i][0], boxes[j][0])) * \
#                            max(0, min(boxes[i][3], boxes[j][3]) - max(boxes[i][1], boxes[j][1]))
#             union = (boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1]) + \
#                     (boxes[j][2] - boxes[j][0]) * (boxes[j][3] - boxes[j][1]) - intersection
#             iou = intersection / union

#             if iou > threshold:
#                 order.remove(j)

#     return boxes[keep]


# img1 = cv2.imread("frame1.png", 0)
# img2 = cv2.imread("frame2.png", 0)
vid = cv2.VideoCapture("demovideo1_test.mp4")

ret, frame1 = vid.read()
ret, frame2 = vid.read()

while vid.isOpened():
    mask = get_mask(frame1, frame2)
    boxes = get_contour(mask, 50)
    # print(boxes)
    box_check = remove_overlap(boxes)
    for i, box in enumerate(boxes):
        if i in box_check:
            frame1 = cv2.rectangle(
                frame1, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 1
            )

    cv2.imshow("2", frame1)
    frame1 = frame2
    ret, frame2 = vid.read()

    if cv2.waitKey(25) & 0xFF == ord("q"):
        break

vid.release()
cv2.destroyAllWindows()
