import cv2
import pandas as pd

cap = cv2.VideoCapture("demovideo1_test.mp4")
box_file = open("video1_15fps_t.txt")
box_df = pd.read_csv(box_file, header = None)

while(True):
    ret, frame = cap.read()

    pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
    frame_data = box_df[box_df[0] == pos_frame]
    print(pos_frame)
    for i, row in frame_data.iterrows():
        min_cor = (round(row[2]), round(row[3]))
        max_cor = (round(row[4] + row[2]), round(row[5] + row[3]))

        frame = cv2.rectangle(frame, min_cor, max_cor, (255,0,0), 1)

    if (ret == True):
        cv2.imshow("vid", frame)

    if cv2.waitKey(25) & 0xFF == ord('q'):
            break  

    if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
        # If the number of captured frames is equal to the total number of frames,
        # we stop
        break

cap.release()
cv2.destroyAllWindows()
