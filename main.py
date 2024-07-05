import cv2

cap = cv2.VideoCapture("demovideo1_test.mp4")
box_file = open("video1_15fps_gt.txt")

while(True):
    ret, frame = cap.read()

    pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)

    box = box_file.readline()
    box_arr = box.split(',')
    box_arr = [float(i) for i in box_arr]
    min_cor = (round(box_arr[2]), round(box_arr[3]))
    max_cor = (round(box_arr[4] + box_arr[2]), round(box_arr[5] + box_arr[3]))

    if(pos_frame ==  box_arr[0]):
        frame = cv2.rectangle(frame, min_cor, max_cor, (255,0,0), 1)

    print(pos_frame, box_arr[0])  
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
