import os
from PIL import Image
import numpy as np
from face_decide import tool

anno_src = r"D:\data\WFLW_annotations\list_98pt_rect_attr_train_test\list_98pt_rect_attr_train.txt"
img_dir = r"D:\data\WFLW_images"
save_path = r"D:\data\wflw_data"

float_num = [0.2, 0.05, 0.05, 0.05, 0.05] #大量正样本
# float_num = [0.2, 0.95,0.95,0.95,0.9,0.5,0.5] #大量负样本

def gen_sample(face_size):
    positive_image_dir = os.path.join(save_path, "positive")
    negative_image_dir = os.path.join(save_path, "negative")
    part_image_dir = os.path.join(save_path, "part")

    # 造出三种路径下的9个文件夹，// 12,24,48
    for dir_path in [positive_image_dir, negative_image_dir, part_image_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    # 样本标签存储路径
    positive_anno_filename = os.path.join(save_path, "positive.txt")
    negative_anno_filename = os.path.join(save_path, "negative.txt")
    part_anno_filename = os.path.join(save_path, "part.txt")

    positive_anno_file = open(positive_anno_filename, "w")
    negative_anno_file = open(negative_anno_filename, "w")
    part_anno_file = open(part_anno_filename, "w")
    positive_count = 0
    negative_count = 0
    part_count = 0

    for i, line in enumerate(open(anno_src)):
        strs = line.split()
        image_filename = strs[-1].strip()
        image_file = os.path.join(img_dir, image_filename)

        with Image.open(image_file) as img:
            img_w, img_h = img.size
            x1 = float(strs[196].strip())
            y1 = float(strs[197].strip())
            x2 = float(strs[198].strip())
            y2 = float(strs[199].strip())
            w = x2 - x1
            h = y2 - y1

            points_x = strs[0:196:2]
            points_y = strs[1:196:2]
            points_x = [float(point_x) for point_x in points_x]
            points_y = [float(point_y) for point_y in points_y]
            boxes = [[x1, y1, x2, y2]]

            cx = x1 + w / 2
            cy = y1 + h / 2
            side_len = max(w, h)
            seed = float_num[np.random.randint(0, len(float_num))]
            for _ in range(5):
                _side_len = side_len + np.random.randint(int(-side_len * seed),
                                                         int(side_len * seed))  # ，偏移边长，最大的边长再加上或减去一个随机系数
                _cx = cx + np.random.randint(int(-cx * seed), int(cx * seed))  # 偏移中心点X
                _cy = cy + np.random.randint(int(-cy * seed), int(cy * seed))  # 偏移中心点Y

                _x1 = _cx - _side_len / 2  # 偏移后的中心点换算回偏移后起始点X,Y
                _y1 = _cy - _side_len / 2
                _x2 = _x1 + _side_len  # 获得偏移后的X2,Y2
                _y2 = _y1 + _side_len
                # 偏移后的的坐标点对应的是正方形

                if _x1 < 0 or _y1 < 0 or _x2 > img_w or _y2 > img_h:  # 判断偏移超出整张图片的就跳过，不截图
                    continue

                offset_x1 = (x1 - _x1) / _side_len  # 获得换算后的偏移率
                offset_y1 = (y1 - _y1) / _side_len
                offset_x2 = (x2 - _x2) / _side_len
                offset_y2 = (y2 - _y2) / _side_len

                np_points_x = np.array(points_x)
                np_points_y = np.array(points_y)
                offset_points_x = (np_points_x - _x1) / _side_len
                offset_points_y = (np_points_y - _y1) / _side_len

                crop_box = [_x1, _y1, _x2, _y2]  # 获得需要截取图片样本的坐标
                face_crop = img.crop(crop_box)
                face_resize = face_crop.resize((face_size, face_size))
                box = [_x1, _y1, _x2, _y2]

                iou = tool.iou(box, np.array(boxes))[0]
                all_points = [offset_x1, offset_y1, offset_x2, offset_y2]
                for i in range(len(points_x)):
                    all_points.append(offset_points_x[i])
                    all_points.append(offset_points_y[i])
                # print(len(all_points))

                if iou > 0.65:  # 正样本// >0.65
                    positive_anno_file.write(
                        "positive/{0}.jpg {1} {2}\n".format(
                            positive_count, 1, all_points))
                    positive_anno_file.flush()
                    face_resize.save(os.path.join(positive_image_dir, "{0}.jpg".format(positive_count)))
                    positive_count += 1
                elif 0.6 > iou > 0.4:  # 部分样本// >0.4
                    part_anno_file.write(
                        "part/{0}.jpg {1} {2}\n".format(
                            part_count, 2, all_points))
                    part_anno_file.flush()
                    face_resize.save(os.path.join(part_image_dir, "{0}.jpg".format(part_count)))
                    part_count += 1
                elif iou < 0.2:  # 负样本// <0.3
                # if iou < 0.2:  # 负样本// <0.3
                    negative_anno_file.write(
                        "negative/{0}.jpg {1} {2}\n".format(negative_count, 0, list(np.zeros(200))))
                    negative_anno_file.flush()
                    face_resize.save(os.path.join(negative_image_dir, "{0}.jpg".format(negative_count)))
                    negative_count += 1
                count = positive_count + part_count + negative_count
                if count % 1000 == 0:
                    print(count)


gen_sample(128)
