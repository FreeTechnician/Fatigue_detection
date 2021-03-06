from face_decide.net import *
import torch
from torchvision import transforms
import time
from PIL import Image
from face_decide.tool import *
import cv2

class Detecter():
    def __init__(self):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.p_net_param = r"param\p_net.pth"
        self.p_net = P_net().to(self.device)
        self.p_net.load_state_dict(torch.load(self.p_net_param))
        self.p_net.eval()

        self.net_param = r"param\net.pth"
        self.net = MainNet().to(self.device)
        self.net.load_state_dict(torch.load(self.net_param))
        self.net.eval()

        self.r_net_param = r"param\r_net.pth"
        self.r_net = R_net().to(self.device)
        self.r_net.load_state_dict(torch.load(self.r_net_param))
        self.r_net.eval()

        self.img_transform = transforms.Compose([transforms.ToTensor()])

    def detecter(self,image):
        '''将图片送入P网络中，并计算所用时间'''
        start_time = time.time()
        pnet_boxes = self.p_net_detect(image)
        # print(pnet_boxes.shape)

        if pnet_boxes.shape[0] == 0:
            return np.array([])

        end_time = time.time()
        t_pnet = end_time - start_time
        # print("结束")
        start_time = time.time()
        rnet_boxes = self.__rnet_detect(image, pnet_boxes)

        if rnet_boxes.shape[0] == 0:
            return np.array([])

        end_time = time.time()
        t_rnet = end_time - start_time
        '''再将P网络的输出和原图送入R网络中，并计算所用时间'''
        start_time = time.time()
        net_boxes = self.net_detect(image, rnet_boxes)
        cls = net_boxes[:,4]
        off = net_boxes[:,0:4]
        points = net_boxes[:,5:]

        if net_boxes.shape[0] == 0:
            return np.array([])

        end_time = time.time()
        t_net = end_time - start_time
        print("识别结束，用时：{}".format(t_pnet+t_rnet+t_net))

        return cls,off,points

    def p_net_detect(self, image):
        boxes = []
        img = image
        w, h = img.size
        max_side = max(w, h)

        scale = 1
        while max_side > 12:

            img_data = self.img_transform(img)
            img_data = img_data.to(self.device)
            img_data.unsqueeze_(0)
            # print(img_data.shape)
            p_cls, p_offset = self.p_net(img_data)
            p_cls = p_cls[0][0].cpu().data
            p_offset = p_offset[0].cpu().data

            index = torch.nonzero(torch.gt(p_cls, 0.6))
            for ind in index:
                boxes.append(self.find_box(ind, p_cls[ind[0], ind[1]], p_offset, scale))
            # print(len(boxes))

            scale *= 0.7
            _w = int(w * scale)
            _h = int(h * scale)
            # print(_w, _h)

            img = img.resize((_w,_h))
            max_side = min(_w,_h)
        return nms(np.array(boxes), 0.3)
        # return np.array(boxes)


    def find_box(self, index, cls, offset, scale, stride=2, side_len=12):
        _x1 = int(index[1] * stride / scale)
        _y1 = int(index[0] * stride / scale)
        _x2 = int((index[1] * stride + side_len) / scale)
        _y2 = int((index[0] * stride + side_len) / scale)
        _w = _x2 - _x1
        _h = _y2 - _y1
        _offset = offset[:, index[0], index[1]]

        x1 = _x1 + _w * _offset[0]
        y1 = _y1 + _w * _offset[1]
        x2 = _x2 + _h * _offset[2]
        y2 = _y2 + _h * _offset[3]

        return [x1, y1, x2, y2, cls]
    def __rnet_detect(self, image, pnet_boxes):
        _img_dataset = []
        '''将输入的框进行正方形化'''
        _pnet_boxes = convert_to_square(pnet_boxes)
        p_offset_p = _pnet_boxes[:, 5:]

        for _box in _pnet_boxes:
            _x1 = int(_box[0])
            _y1 = int(_box[1])
            _x2 = int(_box[2])
            _y2 = int(_box[3])
            '''将输入的框进行截图'''
            img = image.crop((_x1, _y1, _x2, _y2))
            img = img.resize((24, 24))
            img_data = self.img_transform(img)
            _img_dataset.append(img_data)

        img_dataset = torch.stack(_img_dataset)

        img_dataset = img_dataset.to(self.device)

        _cls, _offset = self.r_net(img_dataset)
        _cls = _cls.cpu().data.numpy()
        offset = _offset.cpu().data.numpy()

        # r_offset_p = offset[:, 4:]
        # offest_p = (r_offset_p  + p_offset_p ) / 2
        # offset = np.hstack((offset[:, 0:5], offest_p))

        boxes = []

        idxs, _ = np.where(_cls > 0.5)
        for idx in idxs:
            _box = _pnet_boxes[idx]
            _x1 = int(_box[0])
            _y1 = int(_box[1])
            _x2 = int(_box[2])
            _y2 = int(_box[3])

            ow = _x2 - _x1
            oh = _y2 - _y1

            x1 = _x1 + ow * offset[idx][0]
            y1 = _y1 + oh * offset[idx][1]
            x2 = _x2 + ow * offset[idx][2]
            y2 = _y2 + oh * offset[idx][3]
            cls = _cls[idx][0]

            # offset_px1 = _x1 + ow * offset[idx][4]
            # offset_py1 = _y1 + oh * offset[idx][5]
            # offset_px2 = _x1 + ow * offset[idx][6]
            # offset_py2 = _y1 + oh * offset[idx][7]
            # offset_px3 = _x1 + ow * offset[idx][8]
            # offset_py3 = _y1 + oh * offset[idx][9]
            # offset_px4 = _x1 + ow * offset[idx][10]
            # offset_py4 = _y1 + oh * offset[idx][11]
            # offset_px5 = _x1 + ow * offset[idx][12]
            # offset_py5 = _y1 + oh * offset[idx][13]

            boxes.append([x1, y1, x2, y2, cls])

        return nms(np.array(boxes), 0.3)

    def net_detect(self, image, pnet_boxes):
        _img_dataset = []
        '''将输入的框进行正方形化'''
        _pnet_boxes = convert_to_square(pnet_boxes)

        for _box in _pnet_boxes:
            _x1 = int(_box[0])
            _y1 = int(_box[1])
            _x2 = int(_box[2])
            _y2 = int(_box[3])
            '''将输入的框进行截图'''
            img = image.crop((_x1, _y1, _x2, _y2))
            img = img.resize((128, 128))
            img_data = self.img_transform(img)
            _img_dataset.append(img_data)

        img_dataset = torch.stack(_img_dataset)


        img_dataset = img_dataset.to(self.device)

        _cls, _offset = self.net(img_dataset)
        _cls = _cls.cpu().data.numpy()
        offset = _offset.cpu().data.numpy()
        # print(offset.shape)

        boxes = []
        cls_max = max(_cls)
        idxs, _ = np.where(_cls == cls_max)
        for idx in idxs:
            _box = _pnet_boxes[idx]
            _x1 = int(_box[0])
            _y1 = int(_box[1])
            _x2 = int(_box[2])
            _y2 = int(_box[3])

            ow = _x2 - _x1
            oh = _y2 - _y1

            x1 = _x1 + ow * offset[idx][0]
            y1 = _y1 + oh * offset[idx][1]
            x2 = _x2 + ow * offset[idx][2]
            y2 = _y2 + oh * offset[idx][3]
            cls = _cls[idx][0]
            offset_px = _x1 + ow * offset[idx][4::2]
            offset_py = _y1 + oh * offset[idx][5::2]
            box_n = [x1,y1,x2,y2,cls]
            for i,a in enumerate(offset_px):
                # print(i,a)
                box_n.append(a)
                box_n.append(offset_py[i])

            boxes.append(box_n)


        return nms(np.array(boxes), 0.3, isMin=True)
        # return np.array(boxes)



if __name__ == '__main__':
    x = time.time()
    '''不计算梯度'''
    with torch.no_grad() as grad:

        img_path = r"3.jpg"
        img = Image.open(img_path)
        dec = Detecter()
        cls, off, points = dec.detecter(img)
        print(cls)
        img = cv2.imread(img_path)
        for i in range(len(cls)):
            # if i == 0:
            #     continue

            x1 = int(off[i][0])
            y1 = int(off[i][1])
            x2 = int(off[i][2])
            y2 = int(off[i][3])

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            n = 196
            j = 0
            points_box = []
            while(n > 0):
                a = int(points[i][j])
                # print(a)
                j+=1
                b = int(points[i][j])
                j+=1
                # print(b)
                points_box.append([a,b])
                n-=2
                idn = 0
            for point in points_box:
                # if idn<80:
                #     idn+=1
                #     continue
                draw_0 = cv2.rectangle(img, (point[0], point[1]), (point[0] + 1, point[1] + 1), (0, 0, 255), 2)

        cv2.imshow("1", img)
        cv2.waitKey(0)
        # print(off[i].shape,points[i].shape)
