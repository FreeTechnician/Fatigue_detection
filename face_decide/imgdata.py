import os
import cv2

target_doc = os.path.join("D:\data\WFLW_annotations\list_98pt_rect_attr_train_test","list_98pt_rect_attr_train.txt")
img_file = r"D:\data\WFLW_images"
for i, line in enumerate(open(target_doc)):
    strs = line.split()
    # print(strs)
    img_path = os.path.join(img_file,strs[-1])
    img = cv2.imread(img_path)
    # cv2.imshow("0",img)
    # cv2.waitKey(0)

    face_point = strs[0:-1]
    i = len(face_point)
    print(i)
    points = []
    j = 0
    while(i>0):

        a = int(float(face_point[j]))
        j+=1
        b = int(float(face_point[j]))
        points.append([a,b])
        j+=1
        i-=2
    # print(points)
    for point in points:
        draw_0 = cv2.rectangle(img, (point[0], point[1]), (point[0]+1, point[1]+1), (0, 0, 255), 2)
    cv2.imshow("1",img)
    cv2.waitKey(0)


