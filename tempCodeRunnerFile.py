for i in range(1, 50, 1):

    box = box_file.readline()
    box_arr = box.split(',')

    min_cor = (box_arr[2], box_arr[3])
    max_cor = (box_arr[4], box_arr[5])
    # print(min_cor)