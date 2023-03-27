import os
import json
import cv2
import numpy as np

def xywh2xyxy(points):
    x,y,w,h = points
    # print(x,y,w,h)
    x1 = x - w/2
    x2 = x + w/2
    y1 = y - h/2
    y2 = y + h/2
    # print([x1,y1,x2,y2])
    return [x1,y1,x2,y2]

def xyxy2xywh(points):
    x1,y1,x2,y2 = points
    # print(x1,y1,x2,y2)
    w = abs(x2-x1)
    h = abs(y2-y1)
    x = (x1+x2)/2
    y = (y1+y2)/2
    # print([x,y,w,h])
    return [x,y,w,h]

def xyxy2xywh_norm(points, width, height):
    x1,y1,x2,y2 = points
    # print(x1,y1,x2,y2)
    w = abs(x2-x1) / width
    h = abs(y2-y1) / height
    x = (x1+x2)/(2*width)
    y = (y1+y2)/(2*height)
    # print([x,y,w,h])
    return [x,y,w,h]



def four_point_transform(image, pts):
    (tl, tr, br, bl) = pts
    
    width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    max_width = max(int(width_a), int(width_b))

    height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    max_height = max(int(height_a), int(height_b))

    dst = np.array([[0, 0], [max_width - 1 + 0, 0], [max_width - 1 + 0, max_height - 1 + 0], [0, max_height - 1 + 0]], dtype="float32")
    M = cv2.getPerspectiveTransform(np.float32(pts), dst)
    warped = cv2.warpPerspective(image, M, (max_width, max_height))
    return warped


def read_labelme_1object(label_filepath):
    data = json.load(open(label_filepath, 'r'))
    list_object = data['shapes']
    bbox = []
    for oj in list_object:
        if oj['shape_type'] == 'polygon':
            bbox = oj['points']
        elif oj['shape_type'] == 'rectangle':
            tl = oj['points'][0]
            br = oj['points'][1]
            bbox = [tl, [br[0], tl[1]], br, [tl[0], br[1]]]
        return bbox
    return bbox

def read_labelme(label_filepath):
    data = json.load(open(label_filepath, 'r'))
    
    bboxes = []
    labels = []
    list_object = data['shapes']
    for oj in list_object:
        if oj['shape_type'] == 'polygon':
            bboxes.append(oj['points'][:4])
            labels.append(oj['label'])
        elif oj['shape_type'] == 'rectangle':
            tl = oj['points'][0]
            br = oj['points'][1]
            bboxes.append([tl, [br[0], tl[1]], br, [tl[0], br[1]]])
            labels.append(oj['label'])
    return bboxes, labels

def read_labelme_value(label_filepath):
    data = json.load(open(label_filepath, 'r'))
    
    bboxes = []
    labels = []
    list_object = data['shapes']
    for oj in list_object:
        if oj['shape_type'] == 'polygon':
            label_text, name, is_value = oj['label'].split('_')
            if is_value == '1':
                bboxes.append(oj['points'][:4])
                labels.append(name)
        elif oj['shape_type'] == 'rectangle':
            tl = oj['points'][0]
            br = oj['points'][1]
            label_text, name, is_value = oj['label'].split('_')
            if is_value == '1':
                bboxes.append([tl, [br[0], tl[1]], br, [tl[0], br[1]]])
                labels.append(name)
    return bboxes, labels