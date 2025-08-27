# import time
# import tkinter as tk
# from tkinter import Label
# import cv2
# from PIL import Image, ImageTk
# import threading
# import multiprocessing
# import re
# import onnxruntime as ort
# import numpy as np
# import math
# import torch
# import torchvision
#
# class YOLOv8:
#     def __init__(self, onnx_model, input_image, confidence_thres=0.5, iou_thres=0.1):
#         self.onnx_model = onnx_model
#         self.input_image = input_image
#         self.confidence_thres = confidence_thres
#         self.iou_thres = iou_thres
#
#         self.classes = ["yuan"]
#         self.color_palette = np.random.uniform(0, 255, size=(len(self.classes), 3))
#
#         # 创建 ONNX 运行环境
#         self.session = ort.InferenceSession(
#             onnx_model,
#             providers=["CPUExecutionProvider", "CUDAExecutionProvider"]
#             if ort.get_device() == "GPU"
#             else ["CPUExecutionProvider"],
#         )
#
#         # 获取输入张量的形状，并解析 input_width 和 input_height
#         model_inputs = self.session.get_inputs()
#         input_shape = model_inputs[0].shape  # 一般形如 [1, 3, H, W] 或 [1, 3, None, None]
#         self.input_height = input_shape[2] if input_shape[2] else 640  # 设默认值
#         self.input_width = input_shape[3] if input_shape[3] else 640
#
#     def preprocess(self):
#         # self.img = cv2.imread(self.input_image)
#         self.img = self.input_image
#         self.img_height, self.img_width = self.img.shape[:2]
#         img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
#         img = cv2.resize(img, (self.input_width, self.input_height))
#         image_data = np.array(img) / 255.0
#         image_data = np.transpose(image_data, (2, 0, 1))
#         image_data = np.expand_dims(image_data, axis=0).astype(np.float32)
#         return image_data
#
#     # def draw_detections(self, img, box, score, class_id):
#     #     x1, y1, w, h = box
#     #     color = self.color_palette[class_id]
#     #     cv2.rectangle(img, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, 2)
#     #     label = f"{self.classes[class_id]}: {score:.2f}"
#     #     (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
#     #     label_x = x1
#     #     label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10
#     #     cv2.rectangle(img, (label_x, label_y - label_height), (label_x + label_width, label_y + label_height), color,
#     #                   cv2.FILLED)
#     #     cv2.putText(img, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
#
#     def draw_detections(self, img, box, score, class_id):
#         x1, y1, x2, y2 = box
#         color = self.color_palette[class_id]
#         cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
#         label = f"{self.classes[class_id]}: {score:.2f}"
#         (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
#         label_x = x1
#         label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10
#         cv2.rectangle(img, (label_x, label_y - label_height), (label_x + label_width, label_y + label_height), color,
#                       cv2.FILLED)
#         cv2.putText(img, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
#
#     def handle_preds(preds, device, conf_thresh=0.25, nms_thresh=0.45):
#         total_bboxes, output_bboxes = [], []
#         # 将特征图转换为检测框的坐标
#         N, C, H, W = preds.shape
#         bboxes = torch.zeros((N, H, W, 6))
#         pred = preds.permute(0, 2, 3, 1)
#         # 前背景分类分支
#         pobj = pred[:, :, :, 0].unsqueeze(dim=-1)
#         # 检测框回归分支
#         preg = pred[:, :, :, 1:5]
#         # 目标类别分类分支
#         pcls = pred[:, :, :, 5:]
#
#         # 检测框置信度
#         bboxes[..., 4] = (pobj.squeeze(-1) ** 0.6) * (pcls.max(dim=-1)[0] ** 0.4)
#         bboxes[..., 5] = pcls.argmax(dim=-1)
#
#         # 检测框的坐标
#         gy, gx = torch.meshgrid([torch.arange(H), torch.arange(W)])
#         bw, bh = preg[..., 2].sigmoid(), preg[..., 3].sigmoid()
#         bcx = (preg[..., 0].tanh() + gx.to(device)) / W
#         bcy = (preg[..., 1].tanh() + gy.to(device)) / H
#
#         # cx,cy,w,h = > x1,y1,x2,y1
#         x1, y1 = bcx - 0.5 * bw, bcy - 0.5 * bh
#         x2, y2 = bcx + 0.5 * bw, bcy + 0.5 * bh
#
#         bboxes[..., 0], bboxes[..., 1] = x1, y1
#         bboxes[..., 2], bboxes[..., 3] = x2, y2
#         bboxes = bboxes.reshape(N, H * W, 6)
#         total_bboxes.append(bboxes)
#
#         batch_bboxes = torch.cat(total_bboxes, 1)
#
#         # 对检测框进行NMS处理
#         for p in batch_bboxes:
#             output, temp = [], []
#             b, s, c = [], [], []
#             # 阈值筛选
#             t = p[:, 4] > conf_thresh
#             pb = p[t]
#             for bbox in pb:
#                 obj_score = bbox[4]
#                 category = bbox[5]
#                 x1, y1 = bbox[0], bbox[1]
#                 x2, y2 = bbox[2], bbox[3]
#                 s.append([obj_score])
#                 c.append([category])
#                 b.append([x1, y1, x2, y2])
#                 temp.append([x1, y1, x2, y2, obj_score, category])
#             # Torchvision NMS
#             if len(b) > 0:
#                 b = torch.Tensor(b).to(device)
#                 c = torch.Tensor(c).squeeze(1).to(device)
#                 s = torch.Tensor(s).squeeze(1).to(device)
#                 keep = torchvision.ops.batched_nms(b, s, c, nms_thresh)
#                 for i in keep:
#                     output.append(temp[i])
#             output_bboxes.append(torch.Tensor(output))
#         return output_bboxes
#
#     def postprocess(self, output):
#         outputs = np.transpose(np.squeeze(output[0]))
#         print(outputs.shape)
#         rows = outputs.shape[0]
#         boxes, scores, class_ids, centers = [], [], [], []
#         x_factor = self.img_width / self.input_width
#         y_factor = self.img_height / self.input_height
#         print(outputs)
#         for i in range(rows):
#             classes_scores = outputs[i][4:]
#             max_score = np.amax(classes_scores)
#             if max_score >= self.confidence_thres:
#                 class_id = np.argmax(classes_scores)
#                 x, y, w, h = outputs[i][:4]
#
#                 obj_score = outputs[i][4]
#                 print(outputs[i][5])
#                 category = int(outputs[i][5])
#
#                 x1, y1 = int(outputs[i][0] * x_factor), int(outputs[i][1] * y_factor)
#                 x2, y2 = int(outputs[i][2] * x_factor), int(outputs[i][3] * y_factor)
#
#                 left = int((x - w / 2) * x_factor)
#                 top = int((y - h / 2) * y_factor)
#                 width = int(w * x_factor)
#                 height = int(h * y_factor)
#
#                 boxes.append([x1,y1, x2, y2])
#                 scores.append(obj_score)
#                 class_ids.append(category)
#                 centers.append([int((x1 + x2) / 2), int((y1 + y2) / 2)])
#         indices = cv2.dnn.NMSBoxes(boxes, scores, self.confidence_thres, self.iou_thres)
#         # 将 indices 转换为 NumPy 数组并展平
#         indices = np.array(indices).flatten()
#         # 初始化空列表，用于存储 NMS 后的结果
#         nms_class_ids = []
#         nms_scores = []
#         nms_centers = []
#         nms_widths = []
#         if boxes is not None:
#             # 根据 NMS 的索引筛选结果
#             for i in indices.flatten():
#                 nms_class_ids.append(class_ids[i])
#                 nms_scores.append(scores[i])
#                 nms_centers.append(centers[i])
#                 nms_widths.append(boxes[i][2])
#                 self.draw_detections(self.img, boxes[i], scores[i], class_ids[i])
#
#         # 返回 NMS 后的结果
#         return nms_class_ids, nms_scores, nms_centers, self.img, nms_widths
#
#     def main(self):
#         img_data = self.preprocess()
#         outputs = self.session.run(None, {self.session.get_inputs()[0].name: img_data})
#         #return self.postprocess(outputs)
#
#         return self.handle_preds(outputs,0.6)
#
#
# model_path = 'my.onnx'
# frame = cv2.imread('imga107.jpg')
# conf_threshold = 0.5
#
# start_time = time.time()
# count = 0
# while count<5:
#     yolo = YOLOv8(model_path, frame, conf_threshold, 0.1)
#     cls_, confs, centers, image, width = yolo.main()  # 调用 main 方法获取结果
#     count+=1
# print((time.time()-start_time))
# cv2.imwrite('imga1.jpg', image)
# cv2.imshow('YOLOv8', image)
# cv2.waitKey(1)


import time
from json import detect_encoding
import os

import cv2
import numpy as np


def nms(dets, nmsThreshold):
    """
    非极大值抑制
    dets: [[x1, y1, x2, y2, score], [x1, y1, x2, y2, score], ...]
    """
    # dets:N*M,N是bbox的个数，M的前4位是对应的（x1,y1,x2,y2），第5位是对应的分数
    # #thresh:0.3,0.5....
    if dets.shape[0] == 0:
        return []
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)  # 求每个bbox的面积
    order = scores.argsort()[::-1]  # 对分数进行倒排序
    keep = []  # 用来保存最后留下来的bboxx下标
    while order.size > 0:
        i = order[0]  # 无条件保留每次迭代中置信度最高的bbox
        keep.append(i)
        # 计算置信度最高的bbox和其他剩下bbox之间的交叉区域
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        # 计算置信度高的bbox和其他剩下bbox之间交叉区域的面积
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        # 求交叉区域的面积占两者（置信度高的bbox和其他bbox）面积和的比例
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        # 保留ovr小于thresh的bbox，进入下一次迭代。
        inds = np.where(ovr <= nmsThreshold)[0]
        # 因为ovr中的索引不包括order[0]所以要向后移动一位
        order = order[inds + 1]
    output = []
    for i in keep:
        output.append(dets[i].tolist())
    return output


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def tanh(x):
    return 2.0 / (1 + np.exp(-2 * x)) - 1


def draw_pred(frame, class_name, conf, left, top, right, bottom):
    """
    绘制预测结果
    """
    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), thickness=2)
    label = f"{class_name}: {conf:.2f}"
    labelSize, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    top = max(top - 10, labelSize[1])
    left = max(left, 0)
    cv2.putText(
        frame,
        label,
        (left, top),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        thickness=2,
    )


class QRScanner_Yolo(object):
    def __init__(self, confThreshold=0.6, nmsThreshold=0.5, drawOutput=False):
        """
        YoloV3 二维码识别
        confThreshold: 置信度阈值
        nmsThreshold: 非极大值抑制阈值
        drawOutput: 是否在图像上画出识别结果
        """
        self.confThreshold = confThreshold
        self.nmsThreshold = nmsThreshold
        self.inpWidth = 416
        self.inpHeight = 416
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
        cfg_path = os.path.join(path, "qrcode-yolov3-tiny.cfg")
        weights_path = os.path.join(path, "qrcode-yolov3-tiny.weights")
        self.net = cv2.dnn.readNet(cfg_path, weights_path)
        self.drawOutput = drawOutput

    def post_process(self, frame, outs):
        """
        后处理, 对输出进行筛选
        """
        frameHeight = frame.shape[0]
        frameWidth = frame.shape[1]
        confidences = []
        boxes = []
        centers = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                classId = np.argmax(scores)
                confidence = scores[classId]
                if confidence > self.confThreshold:
                    center_x = int(detection[0] * frameWidth)
                    center_y = int(detection[1] * frameHeight)
                    width = int(detection[2] * frameWidth)
                    height = int(detection[3] * frameHeight)
                    left = int(center_x - width / 2)
                    top = int(center_y - height / 2)
                    boxes.append([left, top, width, height])
                    confidences.append(float(confidence))
                    centers.append((center_x, center_y))
        indices = cv2.dnn.NMSBoxes(
            boxes, confidences, self.confThreshold, self.nmsThreshold
        )
        indices = np.array(indices).flatten().tolist()
        ret = [(centers[i], confidences[i]) for i in indices]
        if self.drawOutput:
            for i in indices:
                box = boxes[i]
                left = box[0]
                top = box[1]
                width = box[2]
                height = box[3]
                draw_pred(
                    frame,
                    "QRcode",
                    confidences[i],
                    left,
                    top,
                    left + width,
                    top + height,
                )
        return ret

    def detect(self, frame):
        """
        执行识别
        return: 识别结果列表: (中点坐标, 置信度)
        """
        blob = cv2.dnn.blobFromImage(
            frame,
            1 / 255.0,
            (self.inpWidth, self.inpHeight),
            [0, 0, 0],
            swapRB=True,
            crop=False,
        )
        # 加载网络
        self.net.setInput(blob)
        # 前向传播
        outs = self.net.forward(self.net.getUnconnectedOutLayersNames())
        return self.__post_process(frame, outs)


class FastestDet:
    def __init__(self, confThreshold=0.5, nmsThreshold=0.4, drawOutput=False):
        """
        FastestDet 目标检测网络
        confThreshold: 置信度阈值
        nmsThreshold: 非极大值抑制阈值
        """
        #path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
        path_names = os.path.join( "my.names")  # 识别类别
        path_onnx = os.path.join("my.onnx")
        self.classes = list(map(lambda x: x.strip(), open(path_names, "r").readlines()))
        self.inpWidth = 352
        self.inpHeight = 352
        self.net = cv2.dnn.readNet(path_onnx)
        self.confThreshold = confThreshold
        self.nmsThreshold = nmsThreshold
        self.drawOutput = drawOutput

    def post_process(self, frame, outs):
        """
        后处理, 对输出进行筛选
        """
        outs = outs.transpose(1, 2, 0)  # 将维度调整为 (H, W, C)
        frameHeight = frame.shape[0]
        frameWidth = frame.shape[1]
        feature_height = outs.shape[0]
        feature_width = outs.shape[1]
        preds = []
        confidences = []
        boxes = []
        ret = []
        for h in range(feature_height):
            for w in range(feature_width):
                data = outs[h][w]
                obj_score, cls_score = data[0], data[5:].max()
                score = (obj_score ** 0.6) * (cls_score ** 0.4)
                if score > self.confThreshold:
                    classId = np.argmax(data[5:])
                    # 检测框中心点偏移
                    x_offset, y_offset = tanh(data[1]), tanh(data[2])
                    # 检测框归一化后的宽高
                    box_width, box_height = sigmoid(data[3]), sigmoid(data[4])
                    # 检测框归一化后中心点
                    box_cx = (w + x_offset) / feature_width
                    box_cy = (h + y_offset) / feature_height
                    x1, y1 = box_cx - 0.5 * box_width, box_cy - 0.5 * box_height
                    x2, y2 = box_cx + 0.5 * box_width, box_cy + 0.5 * box_height
                    x1, y1, x2, y2 = (
                        int(x1 * frameWidth),
                        int(y1 * frameHeight),
                        int(x2 * frameWidth),
                        int(y2 * frameHeight),
                    )
                    preds.append([x1, y1, x2, y2, score, classId])
                    boxes.append([x1, y1, x2 - x1, y2 - y1])
                    confidences.append(score)
        indices = cv2.dnn.NMSBoxes(
            boxes, confidences, self.confThreshold, self.nmsThreshold
        )
        indices = np.array(indices).flatten().tolist()
        for i in indices:
            pred = preds[i]
            score, classId = pred[4], int(pred[5])
            x1, y1, x2, y2 = pred[0], pred[1], pred[2], pred[3]
            center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
            ret.append(((center_x, center_y), self.classes[classId], score))
            if self.drawOutput:
                draw_pred(frame, self.classes[classId], score, x1, y1, x2, y2)
        return ret

    def detect(self, frame):
        """
        执行识别
        return: 识别结果列表: (中点坐标, 类型名称, 置信度)
        """
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (self.inpWidth, self.inpHeight))
        self.net.setInput(blob)
        pred = self.net.forward(self.net.getUnconnectedOutLayersNames())[0][0]
        return self.post_process(frame, pred)

class FastestDetOnnx(FastestDet):
    """
    使用 onnxruntime 运行 FastestDet 目标检测网络
    """

    def __init__(self, confThreshold=0.6, nmsThreshold=0.2, drawOutput=False):
        """
        FastestDet 目标检测网络
        confThreshold: 置信度阈值
        nmsThreshold: 非极大值抑制阈值
        """
        import onnxruntime

        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
        path_names = os.path.join( "my.names")  # 识别类别
        path_onnx = os.path.join("my.onnx")
        self.classes = list(map(lambda x: x.strip(), open(path_names, "r").readlines()))
        self.inpWidth = 640
        self.inpHeight = 640
        self.session = onnxruntime.InferenceSession(path_onnx)
        self.confThreshold = confThreshold
        self.nmsThreshold = nmsThreshold
        self.drawOutput = drawOutput

    def detect(self, frame):
        """
        执行识别
        return: 识别结果列表: (中点坐标, 类型名称, 置信度)
        """
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (self.inpWidth, self.inpHeight))
        input_name = self.session.get_inputs()[0].name
        feature_map = self.session.run([], {input_name: blob})[0][0]
        return self.post_process(frame, feature_map)


class HAWP(object):
    """
    使用 onnxruntime 运行 HAWP 线框检测
    """

    def __init__(self, confThreshold=0.95, drawOutput=False):
        """
        HAWP 线框检测网络
        confThreshold: 置信度阈值
        """
        import onnxruntime

        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
        path_onnx = os.path.join(path, "HAWP.onnx")

        self.onnx_session = onnxruntime.InferenceSession(path_onnx)

        self.input_name = self.onnx_session.get_inputs()[0].name
        self.output_name = self.onnx_session.get_outputs()[0].name

        self.input_shape = self.onnx_session.get_inputs()[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

        self.confThreshold = confThreshold
        self.drawOutput = drawOutput

    def pre_process(self, frame):
        """
        图像预处理
        """
        frame = cv2.resize(frame, dsize=(self.input_width, self.input_height))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = (frame.astype(np.float32) / 255.0 - self.mean) / self.std
        frame = frame.transpose(2, 0, 1)
        frame = np.expand_dims(frame, axis=0)
        return frame

    def post_process(self, frame, feature_map):
        """
        数据后处理
        """
        lines, scores = feature_map[0], feature_map[1]
        image_width, image_height = frame.shape[1], frame.shape[0]
        for line in lines:
            line[0] = int(line[0] / 128 * image_width)
            line[1] = int(line[1] / 128 * image_height)
            line[2] = int(line[2] / 128 * image_width)
            line[3] = int(line[3] / 128 * image_height)
        output = []
        for n in range(len(lines)):
            if scores[n] > self.confThreshold:
                output.append((lines[n], scores[n]))
        if self.drawOutput:
            for line, score in output:
                x1, y1 = int(line[0]), int(line[1])
                x2, y2 = int(line[2]), int(line[3])
                cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        return output

    def detect(self, frame):
        """
        执行识别
        return: 识别结果列表: ((x1,y1,x2,y2),score)
        """
        blob = self.pre_process(frame)
        feature_map = self.onnx_sessio.run(None, {self.input_name: blob})
        return self.post_process(frame, feature_map)

# if __name__ == "__main__":
    # det = FastestDet(0.5, 0.4, True)
    # img = cv2.imread("imga24.jpg")
    # start_time =time.time()
    # res = det.detect(img)
    # print("time:",time.time()-start_time)
    # for item in res:
    #     print(item)
    # cv2.imshow("img",img)
    # cv2.waitKey(0)

    det = FastestDetOnnx(0.5, 0.4, True)
    # img = cv2.imread("imga24.jpg")
    # start_time = time.time()
    # res = det.detect(img)
    # print("time:", time.time() - start_time)
    # for item in res:
    #     print(item)
    # cv2.imshow("img", img)
    # cv2.waitKey(0)


def process_images(input_folder, output_folder, det):
    os.makedirs(output_folder, exist_ok=True)
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

    for image_name in image_files:
        img_path = os.path.join(input_folder, image_name)
        img = cv2.imread(img_path)
        if img is None:
            print(f"跳过无法读取的图像: {img_path}")
            continue

        # start_time = time.time()
        res = det.detect(img)
        # print(f"{image_name} - time: {time.time() - start_time:.3f}s")
        #
        # # 在图像上绘制检测结果
        # for item in res:
        #     class_id, conf, x1, y1, x2, y2 = item  # 假设 item 格式为 [class_id, conf, x1, y1, x2, y2]
        #     cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        #     label = f"{class_id}:{conf:.2f}"
        #     cv2.putText(img, label, (int(x1), int(y1)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # 保存图像
        out_path = os.path.join(output_folder, image_name)
        cv2.imwrite(out_path, img)
        print(f"已保存结果图: {out_path}")

det = FastestDetOnnx(0.6, 0.1, True)
# 设置路径
input_dir = r"E:\25ticup\train_num_7_30\val"
output_dir = r"E:\25ticup\train_num_7_30\val_out"

# 运行处理
# process_images(input_dir, output_dir, det)
def undistort(frame):

    k=np.array([[1.96295802e+03, 0.00000000e+00, 9.04350359e+02],
                [0.00000000e+00, 1.95866974e+03, 5.68555114e+02],
                [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])


    d=np.array([-0.36102308, -0.19379845, -0.00559319,  0.00637392,  1.47648705])
    h,w=frame.shape[:2]
    mapx,mapy=cv2.initUndistortRectifyMap(k,d,None,k,(w,h),5)
    return cv2.remap(frame,mapx,mapy,cv2.INTER_LINEAR)


cap = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
cap.set(cv2.CAP_PROP_FOURCC, fourcc)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
print(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
print(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# img = cv2.imread("img0.jpg")
# print(img.shape)
# print(det.detect(img))
while True:

    start_time = time.time()
    ret, frame = cap.read()
    if ret:
        frame = undistort(frame)
        frame = cv2.rotate(frame, cv2.ROTATE_180)
        frame = frame[400:1080, 620:1240]
        res = det.detect(frame)
        print(res)
        cv2.imshow("frame", frame)
        cv2.waitKey(1)
        # print(res)
        # # 在图像上绘制检测结果
        # for item in res:
        #     class_id, conf, x1, y1, x2, y2 = item  # 假设 item 格式为 [class_id, conf, x1, y1, x2, y2]
        #     cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        #     label = f"{class_id}:{conf:.2f}"
        #     cv2.putText(frame, label, (int(x1), int(y1)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)