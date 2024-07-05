import pandas as pd

box_file = open("video1_15fps_gt.txt")
for i in range(1, 50, 1):

    box = box_file.readlin
    box_arr = box.split(',')

    min_cor = (box_arr[2], box_arr[3])
    max_cor = (box_arr[4], box_arr[5])
    # print(min_cor)

box_df = pd.read_csv(box_file, header = None)

print(box_df.head())

box_file.close()

