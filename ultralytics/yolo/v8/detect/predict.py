# Ultralytics YOLO 🚀, GPL-3.0 license

import torch
import cv2
import torch
from numpy import random
from ultralytics.yolo.engine.predictor import BasePredictor
from ultralytics.yolo.engine.results import Results
from ultralytics.yolo.utils import DEFAULT_CFG, ROOT, ops
from ultralytics.yolo.utils.plotting import Annotator
import config
import cv2
from ultralytics.yolo.v8.detect.deep_sort_pytorch.utils.parser import get_config
from ultralytics.yolo.v8.detect.deep_sort_pytorch.deep_sort import DeepSort
from collections import deque
import numpy as np
from os import getcwd

palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
data_deque = {}

deepsort = None

object_counter = {}

object_counter1 = {}

# Đường kẻ để đếm đối tượng di chuyển qua lại
line = [(800, 100), (800, 600)]  


# Cấu hình cho DeepSort
cwd = getcwd()
cfg_deep = get_config()
cfg_deep.merge_from_file(cwd+"/ultralytics/yolo/v8/detect/deep_sort_pytorch/configs/deep_sort.yaml")

deepsort= DeepSort(cfg_deep.DEEPSORT.REID_CKPT,
                        max_dist=cfg_deep.DEEPSORT.MAX_DIST, min_confidence=cfg_deep.DEEPSORT.MIN_CONFIDENCE,
                        nms_max_overlap=cfg_deep.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg_deep.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg_deep.DEEPSORT.MAX_AGE, n_init=cfg_deep.DEEPSORT.N_INIT, nn_budget=cfg_deep.DEEPSORT.NN_BUDGET,
                        use_cuda=True)
##########################################################################################
def xyxy_to_xywh(xyxy):
    """ Chuyển đổi bounding box từ (x1, y1, x2, y2) sang (x_center, y_center, width, height). """
    x_c = (xyxy[..., 0] + xyxy[..., 2]) / 2  # x center
    y_c = (xyxy[..., 1] + xyxy[..., 3]) / 2  # y center
    w = xyxy[..., 2] - xyxy[..., 0]  # width
    h = xyxy[..., 3] - xyxy[..., 1]  # height
    return x_c, y_c, w, h

def xyxy_to_tlwh(bbox_xyxy):
    tlwh_bboxs = []
    for i, box in enumerate(bbox_xyxy):
        x1, y1, x2, y2 = [int(i) for i in box]
        top = x1
        left = y1
        w = int(x2 - x1)
        h = int(y2 - y1)
        tlwh_obj = [top, left, w, h]
        tlwh_bboxs.append(tlwh_obj)
    return tlwh_bboxs

def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    if label == 0: #person
        color = (85,45,255)
    elif label == 2: # Car
        color = (222,82,175)
    elif label == 3:  # Motobike
        color = (0, 204, 255)
    elif label == 5:  # Bus
        color = (0, 149, 255)
    else:
        color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)

def draw_border(img, pt1, pt2, color, thickness, r, d):
    #Hàm để vẽ bounding box lên ảnh, hiển thị đối tượng và theo dõi ID.
    x1,y1 = pt1
    x2,y2 = pt2
    # Top left
    cv2.line(img, (x1 + r, y1), (x1 + r + d, y1), color, thickness)
    cv2.line(img, (x1, y1 + r), (x1, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)
    # Top right
    cv2.line(img, (x2 - r, y1), (x2 - r - d, y1), color, thickness)
    cv2.line(img, (x2, y1 + r), (x2, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
    # Bottom left
    cv2.line(img, (x1 + r, y2), (x1 + r + d, y2), color, thickness)
    cv2.line(img, (x1, y2 - r), (x1, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)
    # Bottom right
    cv2.line(img, (x2 - r, y2), (x2 - r - d, y2), color, thickness)
    cv2.line(img, (x2, y2 - r), (x2, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)

    cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1, cv2.LINE_AA)
    cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r - d), color, -1, cv2.LINE_AA)
    
    cv2.circle(img, (x1 +r, y1+r), 2, color, 12)
    cv2.circle(img, (x2 -r, y1+r), 2, color, 12)
    cv2.circle(img, (x1 +r, y2-r), 2, color, 12)
    cv2.circle(img, (x2 -r, y2-r), 2, color, 12)
    
    return img

def UI_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]

        img = draw_border(img, (c1[0], c1[1] - t_size[1] -3), (c1[0] + t_size[0], c1[1]+3), color, 1, 8, 2)

        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

#Hàm intersect để kiểm tra xem đối tượng có đi qua vạch hay không
def intersect(A,B,C,D):
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

def ccw(A,B,C):
    return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])


def get_direction(point1, point2):
    direction_str = ""

    # calculate y axis direction
    if point1[1] > point2[1]:
        direction_str += "South"
    elif point1[1] < point2[1]:
        direction_str += "North"

    # calculate x axis direction
    if point1[0] > point2[0]:
        direction_str += "East"
    elif point1[0] < point2[0]:
        direction_str += "West"

    return direction_str


def draw_boxes(img, bbox, names, object_id, identities=None, offset=(0, 0)):
    # Vẽ vạch kẻ dọc
    cv2.line(img, line[0], line[1], (46,162,112), 3)
    height, width, _ = img.shape

    # Xử lý buffer của đối tượng bị mất dấu
    for key in list(data_deque):
        if key not in identities:
            data_deque.pop(key)

    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]

        # Tính tọa độ tâm của đối tượng
        center = (int((x2 + x1) / 2), int((y2 + y1) / 2))

        # Lấy ID của đối tượng
        id = int(identities[i]) if identities is not None else 0

        # Tạo buffer mới cho đối tượng nếu chưa có
        if id not in data_deque:
            data_deque[id] = deque(maxlen=64)
        color = compute_color_for_labels(object_id[i])
        obj_name = names[object_id[i]]
        label = '{}{:d}'.format("", id) + ":" + '%s' % (obj_name)

        # Thêm tâm vào buffer
        data_deque[id].appendleft(center)

        # Kiểm tra hướng di chuyển và đếm số người
        if len(data_deque[id]) >= 2:
            direction = get_direction(data_deque[id][0], data_deque[id][1])

            # Kiểm tra khi đối tượng vượt qua vạch theo hướng West (xuống xe)
            if intersect(data_deque[id][0], data_deque[id][1], line[0], line[1]):
                cv2.line(img, line[0], line[1], (255, 255, 255), 3)
                if "West" in direction:  # Đếm người đi ra khỏi xe buýt
                    if obj_name not in object_counter:
                        object_counter[obj_name] = 1
                    else:
                        object_counter[obj_name] += 1
                # Kiểm tra khi đối tượng vượt qua vạch theo hướng East (lên xe)
                if "East" in direction:  # Đếm người đi lên xe buýt
                    if obj_name not in object_counter1:
                        object_counter1[obj_name] = 1
                    else:
                        object_counter1[obj_name] += 1

        # Hiển thị bounding box và ID của đối tượng
        UI_box(box, img, label=label, color=color, line_thickness=2)

        # Vẽ đường di chuyển của đối tượng
        for j in range(1, len(data_deque[id])):
            if data_deque[id][j - 1] is None or data_deque[id][j] is None:
                continue
            thickness = int(np.sqrt(64 / float(j + j)) * 1.5)
            cv2.line(img, data_deque[id][j - 1], data_deque[id][j], color, thickness)

    # Hiển thị số lượng người đã đếm ở góc màn hình
    # Số người đi lên xe buýt
    for idx, (key, value) in enumerate(object_counter1.items()):
        cnt_str = str(key) + ":" + str(value)
        cv2.line(img, (width - 500, 25), (width, 25), [85, 45, 255], 40)
        cv2.putText(img, f'Number of People Entering', (width - 500, 35), 0, 1, [225, 255, 255], thickness=2, lineType=cv2.LINE_AA)
        cv2.line(img, (width - 150, 65 + (idx * 40)), (width, 65 + (idx * 40)), [85, 45, 255], 30)
        cv2.putText(img, cnt_str, (width - 150, 75 + (idx * 40)), 0, 1, [255, 255, 255], thickness=2, lineType=cv2.LINE_AA)

    # Số người đi xuống xe buýt
    for idx, (key, value) in enumerate(object_counter.items()):
        cnt_str1 = str(key) + ":" + str(value)
        cv2.line(img, (20, 25), (500, 25), [85, 45, 255], 40)
        cv2.putText(img, f'Number of People Exiting', (11, 35), 0, 1, [225, 255, 255], thickness=2, lineType=cv2.LINE_AA)
        cv2.line(img, (20, 65 + (idx * 40)), (127, 65 + (idx * 40)), [85, 45, 255], 30)
        cv2.putText(img, cnt_str1, (11, 75 + (idx * 40)), 0, 1, [225, 255, 255], thickness=2, lineType=cv2.LINE_AA)

    # Lưu thông tin đếm người vào config để có thể sử dụng ở nơi khác
    config.OBJECT_COUNTER = object_counter
    config.OBJECT_COUNTER1 = object_counter1

    return img


class DetectionPredictor(BasePredictor):

    def get_annotator(self, img):
        return Annotator(img, line_width=self.args.line_width, example=str(self.model.names))

    def postprocess(self, preds, img, orig_img):
        preds = ops.non_max_suppression(preds,
                                        self.args.conf,
                                        self.args.iou,
                                        agnostic=self.args.agnostic_nms,
                                        max_det=self.args.max_det)

        results = []
        for i, pred in enumerate(preds):
            orig_img = orig_img[i] if isinstance(orig_img, list) else orig_img
            if not isinstance(orig_img, torch.Tensor):
                pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            path = self.batch[0]
            img_path = path[i] if isinstance(path, list) else path
            results.append(Results(orig_img=orig_img, path=img_path, names=self.model.names, boxes=pred))
        return results
    
    def write_results(self, idx, results, batch):
        """Write inference results to a file or directory."""
        p, im, im0 = batch
        log_string = ''
        all_outputs = []
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        self.seen += 1
        im0 = im0.copy()
        if self.source_type.webcam or self.source_type.from_img:  # batch_size >= 1
            log_string += f'{idx}: '
            frame = self.dataset.count
        else:
            frame = getattr(self.dataset, 'frame', 0)
        self.data_path = p
        self.txt_path = str(self.save_dir / 'labels' / p.stem) + ('' if self.dataset.mode == 'image' else f'_{frame}')
        log_string += '%gx%g ' % im.shape[2:]  # print string
        result = results[idx]
        log_string += result.verbose()

        if self.args.save or self.args.show:  # Add bbox to image
            plot_args = dict(line_width=self.args.line_width,
                             boxes=self.args.boxes,
                             conf=self.args.show_conf,
                             labels=self.args.show_labels)
            if not self.args.retina_masks:
                plot_args['im_gpu'] = im[idx]
            self.plotted_img = result.plot(**plot_args)
        self.annotator = self.get_annotator(im0)

        all_outputs.append(result)

        if len(result) == 0:
            return log_string
        for c in result.boxes.cls.unique():
            n = (result.boxes.cls == c).sum()  # detections per class
            log_string += f"{n} {self.model.names[int(c)]}{'s' * (n > 1)}, "
                
        # Write
        if self.args.save_txt:
            result.save_txt(f'{self.txt_path}.txt', save_conf=self.args.save_conf)
        if self.args.save_crop:
            result.save_crop(save_dir=self.save_dir / 'crops', file_name=self.data_path.stem)

        xywh_bboxs = []
        confs = []
        oids = []
        outputs = []
        for r in result:
            x_c, y_c, bbox_w, bbox_h = xyxy_to_xywh(r.boxes.xyxy)
            xywh_obj = [x_c, y_c, bbox_w, bbox_h]
            xywh_bboxs.append(xywh_obj)
            confs.append([r.boxes.conf.item()])
            oids.append(int(r.boxes.cls))
        xywhs = torch.Tensor(xywh_bboxs)
        confss = torch.Tensor(confs)
          
        outputs = deepsort.update(xywhs, confss, oids, im0)
        if len(outputs) > 0:
            bbox_xyxy = outputs[:, :4]
            identities = outputs[:, -2]
            object_id = outputs[:, -1]
            
            draw_boxes(im0, bbox_xyxy, self.model.names, object_id,identities)

        return log_string


def predict(cfg=DEFAULT_CFG, use_python=False):
    """Runs YOLO model inference on input image(s)."""
    model = cfg.model or 'yolov8n.pt'
    source = cfg.source if cfg.source is not None else ROOT / 'assets' if (ROOT / 'assets').exists() \
        else 'https://ultralytics.com/images/bus.jpg'

    args = dict(model=model, source=source)
    if use_python:
        from ultralytics import YOLO
        YOLO(model)(**args)
    else:
        predictor = DetectionPredictor(overrides=args)
        predictor.predict_cli()


if __name__ == "__main__":
    predict()
