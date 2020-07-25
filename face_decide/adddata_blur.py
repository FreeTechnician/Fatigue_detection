import cv2
import os
import numpy as np
data_path = r"D:\data\wflw_data"
save_path = r"D:\data\wflw_data_3"

positive_file_name = os.path.join(data_path,"positive")
part_file_name = os.path.join(data_path,"part")
negative_file_name = os.path.join(data_path,"negative")

save_positive_file_name = os.path.join(save_path,"positive")
save_part_file_name = os.path.join(save_path,"part")
save_negative_file_name = os.path.join(save_path,"negative")

files_name = [positive_file_name,part_file_name,negative_file_name]
save_files_path = [save_positive_file_name,save_part_file_name,save_negative_file_name]

for a in save_files_path:
    if not os.path.exists(a):
        os.makedirs(a)



for i,file_name in enumerate(files_name):
    imgs_name = os.listdir(file_name)
    save_file_path = save_files_path[i]
    for img_name in imgs_name:
        save_img_path = os.path.join(save_file_path,img_name)
        img_path = os.path.join(file_name,img_name)
        img = cv2.imread(img_path)
        dst = cv2.blur(img, (np.random.randint(5,9), np.random.randint(5,9)))
        cv2.imwrite(save_img_path, dst)
        # cv2.imshow("1",dst)
        # cv2.imshow("2",img)
        # cv2.waitKey(0)
        # break