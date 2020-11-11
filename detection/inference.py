try:
    from mmdet.apis import inference_detector
except:
    from detection.detector_test import inference_detector
    print("Error occurred when importing mmdetection.\nUsing unit test file.")
import cv2
import os 
# import gdal
import numpy as np
from tqdm import trange
import six
import gdal
import argparse
import random
import math

COLORS = {'Blue': (0, 130, 200), 'Red': (230, 25, 75), 
          'Yellow': (255, 225, 25), 'Green': (60, 180, 75), 
          'Orange': (245, 130, 48), 'Purple': (145, 30, 180), 
          'Cyan': (70, 240, 240), 'Magenta': (240, 50, 230), 
          'Lavender': (230, 190, 255), 'Lime': (210, 245, 60), 
          'Teal': (0, 128, 128), 'Pink': (250, 190, 190), 
          'Brown': (170, 110, 40), 'Beige': (255, 250, 200), 
          'Maroon': (128, 0, 0), 'Mint': (170, 255, 195), 
          'Olive': (128, 128, 0), 'Apricot': (255, 215, 180), 
          'Navy': (0, 0, 128), 'Grey': (128, 128, 128), 
          'White': (255, 255, 255), 'Black': (0, 0, 0)}

def mkdir_or_exist(dir_name, mode=0o777):
    if dir_name == '':
        return
    dir_name = os.path.expanduser(dir_name)
    if six.PY3:
        os.makedirs(dir_name, mode=mode, exist_ok=True)
    else:
        if not os.path.isdir(dir_name):
            os.makedirs(dir_name, mode=mode)

def is_str(x):
    """Whether the input is an string instance."""
    return isinstance(x, six.string_types)

def color_val(color = None):
    if is_str(color):
        color = color[0].upper() + color[1:].lower()
        return list(COLORS[color])[::-1]
    elif color == None:
        color_name = random.choice(list(COLORS.keys()))
        return list(COLORS[color_name])[::-1]
    elif type(color) == int:
        return list(COLORS[list(COLORS.keys())[color]])[::-1]
    else:
        return list(COLORS['Red'])[::-1]

def thetaobb2pointobb(thetaobb):
    """
    docstring here
        :param self: 
        :param thetaobb: list, [x, y, w, h, theta]
    """
    box = cv2.boxPoints(((thetaobb[0], thetaobb[1]), (thetaobb[2], thetaobb[3]), thetaobb[4]*180.0/np.pi))
    box = np.reshape(box, [-1, ]).tolist()
    pointobb = [box[0], box[1], box[2], box[3], box[4], box[5], box[6], box[7]]

    return pointobb

def pointobb2bbox(pointobb):
    """
    docstring here
        :param self: 
        :param pointobb: list, [x1, y1, x2, y2, x3, y3, x4, y4]
        return [xmin, ymin, xmax, ymax]
    """
    xmin = min(pointobb[0::2])
    ymin = min(pointobb[1::2])
    xmax = max(pointobb[0::2])
    ymax = max(pointobb[1::2])
    bbox = [xmin, ymin, xmax, ymax]
    
    return bbox

def mask2rbbox(mask, dilate=False):
    # mask = (mask > 0.5).astype(np.uint8)
    mask = mask.astype(np.uint8)
    gray = np.array(mask*255, dtype=np.uint8)

    if dilate:
        h, w = gray.shape[0], gray.shape[1]
        size = np.sum(gray / 255)
        niter = int(math.sqrt(size * min(w, h) / (w * h)) * 2)
        print(niter)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1 + niter, 1 + niter))
        gray = cv2.dilate(gray, kernel)
    contours, hierarchy = cv2.findContours(gray.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours != []:
        imax_cnt_area = -1
        imax = -1
        cnt = max(contours, key = cv2.contourArea)
        rect = cv2.minAreaRect(cnt)
        x, y, w, h, theta = rect[0][0], rect[0][1], rect[1][0], rect[1][1], rect[2]
        theta = theta * np.pi / 180.0
        thetaobb = [x, y, w, h, theta]
        pointobb = thetaobb2pointobb([x, y, w, h, theta])
    else:
        thetaobb = [0, 0, 0, 0, 0]
        pointobb = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    return thetaobb, pointobb


def read_gaofen(self, img_file):
    def rescale(band, minval=None, maxval=None):
        minval = minval if minval is not None else 0
        maxval = maxval if maxval is not None else 1
        maxval = maxval if maxval != minval else maxval + 1

        band = 255 * (band.astype(np.float32) - minval) / (maxval - minval)
        band = np.clip(band, 0, 255).astype(np.uint8)
        return band

    percents = [2, 98]
    src_image = gdal.Open(img_file).ReadAsArray()
    # 8 bit image: no need to rescale.
    if src_image.dtype == "uint8":
        if len(src_image.shape) == 3:
            return src_image.transpose([1, 2, 0])
        else:
            return np.stack([src_image, src_image, src_image], axis=-1).transpose([1, 2, 0])
    # 16 bit image: rescale to [0,255]
    elif len(src_image.shape) == 3:  # Shape is [ch, h, w]
        img_shape = list(src_image.shape)
        img_shape[0] = 3
        dst_image = np.zeros(img_shape, np.uint8)
        for i in range(3):
            minval = np.percentile(src_image[i], percents[0])
            maxval = np.percentile(src_image[i], percents[1])
            dst_image[i] = rescale(src_image[i], minval, maxval)
        dst_image = dst_image.transpose([1, 2, 0])
        print(dst_image.shape)

    else:  # Shape is [h, w]
        minval = np.percentile(src_image[src_image > 0], percents[0])
        maxval = np.percentile(src_image[src_image > 0], percents[1])
        dst_image = rescale(src_image, minval, maxval)
        dst_image = np.stack([dst_image, dst_image, dst_image], axis=-1)
    return dst_image

def get_key(dict, value):
    return [k for k,v in dict.items() if v==value][0]

def all_NMS(all_objects):
    iou_th = 0.4
    tmp_objects = []
    re_idx=[]
    num = len(all_objects)
    for idx, obj in enumerate(all_objects):
        if idx in re_idx:
            continue
        bbox1 = obj['bbox']
        score1 = obj['score']

        for idx_c in range(idx+1, num):
            if idx_c in re_idx:
                continue
            obj_c = all_objects[idx_c]
            bbox2 = obj_c['bbox']
            score2 = obj_c['score']
            iou, inter, area1, area2= box_iou(bbox1, bbox2)
            if iou >iou_th:
                id_m = idx if score1<score2 else idx_c
                re_idx.append(id_m)
            elif inter==area1 or inter==area2:
                id_m = idx if score1<score2 else idx_c
                re_idx.append(id_m)
        if idx not in re_idx:
            tmp_objects.append(obj)
    
    return tmp_objects

def box_iou(bbox1, bbox2):

    area1 = box_area(bbox1)
    area2 = box_area(bbox2)

    lt_x, lt_y = max(bbox1[0], bbox2[0]), max(bbox1[1], bbox2[1])
    rb_x, rb_y = min(bbox1[2], bbox2[2]), min(bbox1[3], bbox2[3])
    if rb_x-lt_x>0 and rb_y-lt_y>0:
        inter = (lt_x-rb_x)*(lt_y-rb_y)
    else:
        inter = 0
    iou = inter / (area1 + area2 - inter)
    return iou, inter, area1, area2

def box_area(bbox):
    return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

def split_image(img, subsize=1024, gap=200, mode='keep_all'):
    img_height, img_width = img.shape[0], img.shape[1]

    start_xs = np.arange(0, img_width, subsize - gap)
    if mode == 'keep_all':
        start_xs[-1] = img_width - subsize if img_width - start_xs[-1] <= subsize else start_xs[-1]
    elif mode == 'drop_boundary':
        if img_width - start_xs[-1] < subsize - gap:
            start_xs = np.delete(start_xs, -1)
    start_xs[-1] = np.maximum(start_xs[-1], 0)

    start_ys = np.arange(0, img_height, subsize - gap)
    if mode == 'keep_all':
        start_ys[-1] = img_height - subsize if img_height - start_ys[-1] <= subsize else start_ys[-1]
    elif mode == 'drop_boundary':
        if img_height - start_ys[-1] < subsize - gap:
            start_ys = np.delete(start_ys, -1)
    start_ys[-1] = np.maximum(start_ys[-1], 0)

    subimages = dict()
    for start_x in start_xs:
        for start_y in start_ys:
            end_x = np.minimum(start_x + subsize, img_width)
            end_y = np.minimum(start_y + subsize, img_height)
            subimage = img[start_y:end_y, start_x:end_x, ...]
            coordinate = (start_x, start_y)
            subimages[coordinate] = subimage
    return subimages

def simple_obb_txt_dump(objects, img_name, save_dir):
    bboxes, pointobbs, labels, scores, num = [], [], [], [], 0
    for obj in objects:
        bboxes.append(obj['bbox'])
        pointobbs.append(obj['pointobbs'])
        labels.append(obj['label'])
        scores.append(obj['score'])
        num += 1
    
    txt_file = open((save_dir + '/' + img_name + '.txt'), 'w', encoding='utf-8')

    for idx in range(num):
        name = labels[idx]
        sco  = str(scores[idx])
        pointres = str(pointobbs[idx][0]) + ' ' + str(pointobbs[idx][1]) + ' ' + str(pointobbs[idx][2]) + ' ' + str(pointobbs[idx][3]) + ' ' + str(pointobbs[idx][4]) + ' ' + str(pointobbs[idx][5]) + ' ' + str(pointobbs[idx][6]) + ' ' + str(pointobbs[idx][7])
        txt_file.write(name + ' ' + sco + ' ' + pointres + '\n')
    
def process_mask(det_result,num_class,sorces_th,cls_map,subimage_coordinate,all_objects):
    for idx_cls in range(num_class):
        mask_cls = det_result[1][idx_cls]
        obbox_cls = det_result[0][idx_cls]
        for ids, each_mask in enumerate(mask_cls):
            object_struct = {}
            thetaobb, pointobb = mask2rbbox(each_mask)
            score = obbox_cls[ids][4]
            bbox = pointobb2bbox(pointobb)
            bbox = [bbox[0] + subimage_coordinate[0], bbox[1] + subimage_coordinate[1],
                    bbox[2] + subimage_coordinate[0], bbox[3] + subimage_coordinate[1]]
            thetaobb[0] = thetaobb[0] + subimage_coordinate[0]
            thetaobb[1] = thetaobb[1] + subimage_coordinate[1]
            pointobb = [pointobb[0] + subimage_coordinate[0],
                        pointobb[1] + subimage_coordinate[1],
                        pointobb[2] + subimage_coordinate[0],
                        pointobb[3] + subimage_coordinate[1],
                        pointobb[4] + subimage_coordinate[0],
                        pointobb[5] + subimage_coordinate[1],
                        pointobb[6] + subimage_coordinate[0],
                        pointobb[7] + subimage_coordinate[1]]
            if score > sorces_th:
                object_struct['bbox'] = bbox
                object_struct['rbbox'] = thetaobb
                object_struct['pointobbs'] = pointobb
                object_struct['label'] = get_key(cls_map, idx_cls + 1)
                object_struct['score'] = score
                all_objects.append(object_struct)

def process_bbox(det_result,num_class,sorces_th,cls_map,subimage_coordinate,all_objects):
    for idx_cls in range(num_class):
        bbox_cls = det_result[idx_cls]
        for each_bbox in bbox_cls:
            object_struct = {}
            x1,y1,x2,y2,score = each_bbox
            bbox = [x1 + subimage_coordinate[0], y1 + subimage_coordinate[1],
                    x2 + subimage_coordinate[0], y2 + subimage_coordinate[1]]
            thetaobb = [
                (bbox[0] + bbox[2]) / 2,  #cx
                (bbox[1] + bbox[3]) / 2,  #cy
                abs(bbox[2] - bbox[0]),  #w
                abs(bbox[3] - bbox[1]),  #h
                0,  #theta
            ]
            pointobb = thetaobb2pointobb(thetaobb)
            if score > sorces_th:
                object_struct['bbox'] = bbox
                object_struct['rbbox'] = thetaobb
                object_struct['pointobbs'] = pointobb
                object_struct['label'] = get_key(cls_map, idx_cls + 1)
                object_struct['score'] = score
                all_objects.append(object_struct)

def inference(detector=None,img=None,is_obb=False,crop_size=800,crop_overlap=300,sorces_th=0.3):
    cls_list = detector.CLASSES
    num_class = len(cls_list)
    cls_map = {}
    ###generate cls_map = {"cls":int}
    for i in range(1,len(cls_list)+1):
        cls_map[cls_list[i-1]] = i
    if max(img.shape) <= crop_size:
        subimages = {(0,0):img}
        subimage_coordinates = [(0,0)]
    else:
        subimages = split_image(img, subsize=crop_size, gap=crop_overlap)
        subimage_coordinates = list(subimages.keys())

    all_objects = []

    for subimage_coordinate in subimage_coordinates:
        im = subimages[subimage_coordinate]
        det_result = inference_detector(detector, im)
        if is_obb:
            process_mask(det_result,num_class,sorces_th,cls_map,subimage_coordinate,all_objects)
        else:
            process_bbox(det_result,num_class,sorces_th,cls_map,subimage_coordinate,all_objects)

    final_objects = all_NMS(all_objects)
    return final_objects
