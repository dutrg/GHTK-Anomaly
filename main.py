import cv2
import pandas as pd
import numpy as np

cap = cv2.VideoCapture("demovideo1_test.mp4")
box_file = open("video1_15fps_gt.txt")
box_df = pd.read_csv(box_file, header = None)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Initialize an empty accumulator for the heatmap
heatmap = np.zeros((frame_height, frame_width), dtype=np.float32)

kernel_size = 150
kernel = cv2.getGaussianKernel(kernel_size, 0)

while(True):
    ret, frame = cap.read()

    pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
    frame_data = box_df[box_df[0] == pos_frame]
    print(pos_frame)

    for i, row in frame_data.iterrows():
        min_cor = (round(row[2]), round(row[3]))
        max_cor = (round(row[4] + row[2]), round(row[5] + row[3]))

        frame = cv2.rectangle(frame, min_cor, max_cor, (255,0,0), 1)

        center_x = round(row[2] + row[4] / 2)
        center_y = round(row[3] + row[5] / 2)
        
        for y in range(max(0, center_y - kernel_size // 2), min(center_y + kernel_size // 2, heatmap.shape[0])):
            for x in range(max(0, center_x - kernel_size // 2), min(center_x + kernel_size // 2, heatmap.shape[1])):
                heatmap[y, x] += kernel[y - center_y + kernel_size // 2, 0] * kernel[x - center_x + kernel_size // 2, 0]


    heatmap_normalized = cv2.normalize(heatmap * 5, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    heatmap_frame = cv2.applyColorMap(heatmap_normalized, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(frame, 0.3, heatmap_frame, 0.7, 0)

    if (ret == True):
        cv2.imshow("vid", overlay)

    if cv2.waitKey(25) & 0xFF == ord('q'):
            break  

    if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
        break

cap.release()
cv2.destroyAllWindows()
