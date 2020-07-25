import numpy as np


def iou(box, boxes, isMin=False):
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    # print(box_area,boxes_area)
    xx1 = np.maximum(box[0], boxes[:, 0])
    yy1 = np.maximum(box[1], boxes[:, 1])
    xx2 = np.minimum(box[2], boxes[:, 2])
    yy2 = np.minimum(box[3], boxes[:, 3])
    h = np.maximum(0, yy2 - yy1)
    w = np.maximum(0, xx2 - xx1)
    inter = h * w

    if isMin:
        box_iou = np.true_divide(inter, np.minimum(box_area, boxes_area))

    else:
        # box_iou = inter/ (box_area + boxes_area - inter)
        box_iou = np.true_divide(inter, (box_area + boxes_area - inter))
    return box_iou


def nms(boxes, thresh, isMin=False):
    if boxes.shape[0] == 0:
        return np.array([])
    _boxes = boxes[(-boxes[:, 0]).argsort()]
    r_boxer = []
    while _boxes.shape[0] > 0:
        a = _boxes[0]
        b_boxes = _boxes[1:]
        r_boxer.append(a)
        # index = np.where(iou(a[:-1], b_boxes[:, :-1], isMin) < thresh)
        iou_data = iou(a, b_boxes, isMin)
        # print(iou_data)
        index = np.where(iou_data <= thresh)
        # print(index)
        _boxes = b_boxes[index]
    if _boxes.shape[0] > 0:
        r_boxer.append(_boxes)
    return np.stack(r_boxer)

def convert_to_square(bbox):
    square_box = bbox.copy()

    '''计算原数据框的宽w和高h，以及最大变成max_side'''
    w = bbox[:,2] - bbox[:,0]
    h = bbox[:,3] - bbox[:,1]
    max_side = np.maximum(w,h)

    '''计算将原数据正方形化后的坐标值'''
    square_box[:,0] = bbox[:,0] + w * 0.5 - max_side * 0.5
    square_box[:,1] = bbox[:,1] + h * 0.5 - max_side * 0.5
    square_box[:,2] = square_box[:,0] + max_side
    square_box[:,3] = square_box[:,1] + max_side

    return square_box