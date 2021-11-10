import sys
import os
import time
import math
import numpy as np

import itertools
import struct  # get_image_size
import imghdr  # get_image_size


def sigmoid(x):
    return 1.0 / (np.exp(-x) + 1.)


def softmax(x):
    x = np.exp(x - np.expand_dims(np.max(x, axis=1), axis=1))
    x = x / np.expand_dims(x.sum(axis=1), axis=1)
    return x


def bbox_iou(box1, box2, x1y1x2y2=True):

    if x1y1x2y2:
        mx = min(box1[0], box2[0])
        Mx = max(box1[2], box2[2])
        my = min(box1[1], box2[1])
        My = max(box1[3], box2[3])
        w1 = box1[2] - box1[0]
        h1 = box1[3] - box1[1]
        w2 = box2[2] - box2[0]
        h2 = box2[3] - box2[1]
    else:
        w1 = box1[2]
        h1 = box1[3]
        w2 = box2[2]
        h2 = box2[3]

        mx = min(box1[0], box2[0])
        Mx = max(box1[0] + w1, box2[0] + w2)
        my = min(box1[1], box2[1])
        My = max(box1[1] + h1, box2[1] + h2)
    uw = Mx - mx
    uh = My - my
    cw = w1 + w2 - uw
    ch = h1 + h2 - uh
    carea = 0
    if cw <= 0 or ch <= 0:
        return 0.0

    area1 = w1 * h1
    area2 = w2 * h2
    carea = cw * ch
    uarea = area1 + area2 - carea
    return carea / uarea


def nms_cpu(boxes, confs, nms_thresh=0.5, min_mode=False):
    # print(boxes.shape)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 0] + boxes[:, 2]
    y2 = boxes[:, 1] + boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = confs.argsort()[::-1]

    keep = []
    while order.size > 0:
        idx_self = order[0]
        idx_other = order[1:]

        keep.append(idx_self)

        xx1 = np.maximum(x1[idx_self], x1[idx_other])
        yy1 = np.maximum(y1[idx_self], y1[idx_other])
        xx2 = np.minimum(x2[idx_self], x2[idx_other])
        yy2 = np.minimum(y2[idx_self], y2[idx_other])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h

        if min_mode:
            over = inter / np.minimum(areas[order[0]], areas[order[1:]])
        else:
            over = inter / (areas[order[0]] + areas[order[1:]] - inter)

        inds = np.where(over <= nms_thresh)[0]
        order = order[inds + 1]
    
    return np.array(keep)



def plot_boxes_cv2(img, boxes, fps, class_names=None, color=None):
    import cv2
    img = np.copy(img)
    colors = np.array([[1, 0, 1], [0, 0, 1], [0, 1, 1], [0, 1, 0], [1, 1, 0], [1, 0, 0]], dtype=np.float32)

    def get_color(c, x, max_val):
        ratio = float(x) / max_val * 5
        i = int(math.floor(ratio))
        j = int(math.ceil(ratio))
        ratio = ratio - i
        r = (1 - ratio) * colors[i][c] + ratio * colors[j][c]
        return int(r * 255)
  
    width = int(img.shape[1])
    height = int(img.shape[0])
    FPS_str = ""
    for i in range(len(boxes)):
        box = boxes[i]
        for j in range(len(box)):
            x1 = int((box[j][0] - box[j][2] / 2.0) * width)
            y1 = int((box[j][1] - box[j][3] / 2.0) * height)
            x2 = int((box[j][0] + box[j][2] / 2.0) * width)
            y2 = int((box[j][1] + box[j][3] / 2.0) * height)
 
            if color:
                rgb = color
            else:
                rgb = (255, 0, 0)
            if len(box[j]) >= 7 and class_names:
                cls_conf = round(box[j][5], 2)
                cls_id = box[j][6]
                #print('%s: %f' % (class_names[cls_id], cls_conf))
                classes = len(class_names)
                offset = cls_id * 123457 % classes
                red = get_color(2, offset, classes)
                green = get_color(1, offset, classes)
                blue = get_color(0, offset, classes)
                FPS_str = "FPS: {:.1f}".format(fps)
                if color is None:
                    rgb = (red, green, blue)
                img = cv2.putText(img, class_names[cls_id], (x1, y1-3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, rgb, 1)
                img = cv2.putText(img, str(cls_conf), (x1, y2+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, rgb, 1)
                
            img = cv2.rectangle(img, (x1, y1), (x2, y2), rgb, 1)
    img = cv2.putText(img, FPS_str, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)   

    return img

def trt_plot_boxes_cv2(img, boxes, fps, class_names=None, color=None):
    import cv2
    img = np.copy(img)
    colors = np.array([[1, 0, 1], [0, 0, 1], [0, 1, 1], [0, 1, 0], [1, 1, 0], [1, 0, 0]], dtype=np.float32)

    def get_color(c, x, max_val):
        ratio = float(x) / max_val * 5
        i = int(math.floor(ratio))
        j = int(math.ceil(ratio))
        ratio = ratio - i
        r = (1 - ratio) * colors[i][c] + ratio * colors[j][c]
        return int(r * 255)

    width = img.shape[1]
    height = img.shape[0]
    FPS_str = ""
    for i in range(len(boxes)):
        box = boxes[i]
        x1 = int((box[0] - box[2] / 2.0) * width)
        y1 = int((box[1] - box[3] / 2.0) * height)
        x2 = int((box[0] + box[2] / 2.0) * width)
        y2 = int((box[1] + box[3] / 2.0) * height)

        if color:
            rgb = color
        else:
            rgb = (255, 0, 0)
        if len(box) >= 7 and class_names:
            cls_conf = box[5]
            cls_id = box[6]
            #print('%s: %f' % (class_names[cls_id], cls_conf))
            classes = len(class_names)
            offset = cls_id * 123457 % classes
            red = get_color(2, offset, classes)
            green = get_color(1, offset, classes)
            blue = get_color(0, offset, classes)
            FPS_str = "FPS: {:.1f}".format(fps)
            if color is None:
                rgb = (red, green, blue)
            img = cv2.putText(img, class_names[cls_id], (x1, y1-3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, rgb, 1)
            
        img = cv2.rectangle(img, (x1, y1), (x2, y2), rgb, 1)
    img = cv2.putText(img, FPS_str, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)    

    return img

def read_truths(lab_path):
    if not os.path.exists(lab_path):
        return np.array([])
    if os.path.getsize(lab_path):
        truths = np.loadtxt(lab_path)
        truths = truths.reshape(truths.size / 5, 5)  # to avoid single truth problem
        return truths
    else:
        return np.array([])


def load_class_names(namesfile):
    class_names = []
    with open(namesfile, 'r') as fp:
        lines = fp.readlines()
    for line in lines:
        line = line.rstrip()
        class_names.append(line)
    return class_names

def post_processing(img, conf_thresh, nms_thresh, output):

    # anchors = [12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401]
    # num_anchors = 9
    # anchor_masks = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    # strides = [8, 16, 32]
    # anchor_step = len(anchors) // num_anchors
    
    if type(output).__name__ != 'ndarray':
        output = output.cpu().detach().numpy()

    # [batch, num, 4]
    box_array = output[:, :, :4]

    # [batch, num, num_classes]
    confs = output[:, :, 4:]

    # [batch, num, num_classes] --> [batch, num]
    max_conf = np.max(confs, axis=2)
    max_id = np.argmax(confs, axis=2)

    bboxes_batch = []
    for i in range(box_array.shape[0]):
        argwhere = max_conf[i] > conf_thresh
        l_box_array = box_array[i, argwhere, :]
        l_max_conf = max_conf[i, argwhere]
        l_max_id = max_id[i, argwhere]
        keep = nms_cpu(l_box_array, l_max_conf, nms_thresh)
        bboxes = []
    
        if (keep.size > 0):
            l_box_array = l_box_array[keep, :]
            l_max_conf = l_max_conf[keep]
            l_max_id = l_max_id[keep]
      
            for j in range(l_box_array.shape[0]):
                # ############ car bonnet iou
                # bonnet = np.array([img.shape[2]*0.07, img.shape[3]*0.8, img.shape[2]*0.93, img.shape[3]*0.98])
                
                # bx1 = (l_box_array[j,0] - l_box_array[j,2] / 2.0) * img.shape[2]
                # by1 = (l_box_array[j,1] - l_box_array[j,3] / 2.0) * img.shape[3]
                # bx2 = (l_box_array[j,0] + l_box_array[j,2] / 2.0) * img.shape[2]
                # by2 = (l_box_array[j,1] + l_box_array[j,3] / 2.0) * img.shape[3]
                # com_box = np.array([bx1, by1, bx2, by2])

                # x1 = np.maximum(bonnet[0], com_box[0])
                # y1 = np.maximum(bonnet[1], com_box[1])
                # x2 = np.minimum(bonnet[2], com_box[2])
                # y2 = np.minimum(bonnet[3], com_box[3])

                # bw = np.maximum(x2 - x1 + 1., 0.)
                # bh = np.maximum(y2 - y1 + 1., 0.)
                # inters = bw * bh

                # union = ((com_box[2] - com_box[0] + 1.) * (com_box[3] - com_box[1] + 1.) +
                #    (bonnet[2] - bonnet[0] + 1.) *
                #    (bonnet[3] - bonnet[1] + 1.) - inters)
         
                # bonnet_iou = inters / union
                # ############################
                # if bonnet_iou<0.5:  
                if l_max_id[j] == 2 or l_max_id[j] == 0:
                    bboxes.append([l_box_array[j, 0], l_box_array[j, 1], l_box_array[j, 2], l_box_array[j, 3], l_max_conf[j], l_max_conf[j], l_max_id[j]])
                    
        bboxes_batch.append(bboxes)
    return bboxes_batch


def eval_post_processing(img, conf_thresh, nms_thresh, output, img_name):

    # [batch, num, 1, 4]
    box_array = output[0]
    # [batch, num, num_classes]
    confs = output[1]

    if type(output).__name__ != 'ndarray':
        output = output.cpu().detach().numpy()

    # [batch, num, 4]
    box_array = output[:, :, :4]
    # [batch, num, num_classes]
    confs = output[:, :, 4:]

    # [batch, num, num_classes] --> [batch, num]
    max_conf = np.max(confs, axis=2)
    max_id = np.argmax(confs, axis=2)
 
    bboxes_batch = []
    for i in range(box_array.shape[0]):
       
        argwhere = max_conf[i] > conf_thresh
        l_box_array = box_array[i, argwhere, :]
        l_max_conf = max_conf[i, argwhere]
        l_max_id = max_id[i, argwhere]

        keep = nms_cpu(l_box_array, l_max_conf, nms_thresh)
        bboxes = []
        
        if (keep.size > 0):
            l_box_array = l_box_array[keep, :]
            l_max_conf = l_max_conf[keep]
            l_max_id = l_max_id[keep]
            for j in range(l_box_array.shape[0]):  
                if l_max_id[j]==0 or l_max_id[j]==1 or l_max_id[j]==2 or l_max_id[j]==3 or l_max_id[j]==4 or l_max_id[j]==5:
                    x1 = int((l_box_array[j, 0] - l_box_array[j, 2] /2.0 ) * 640)
                    y1 = int((l_box_array[j, 1] - l_box_array[j, 3] /2.0 ) * 403)
                    x2 = int((l_box_array[j, 0] + l_box_array[j, 2] /2.0 ) * 640)
                    y2 = int((l_box_array[j, 1] + l_box_array[j, 3] /2.0 ) * 403)
                    x1, y1, x2, y2 = remove_bonnet(img, [x1, y1, x2, y2])
                    bboxes.append([x1, y1, x2, y2, l_max_conf[j], l_max_conf[j], l_max_id[j], img_name[i]])
      
            bboxes_batch.append(bboxes)

    return bboxes_batch


