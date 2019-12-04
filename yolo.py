# -*- coding: utf-8 -*-
"""
Class definition of YOLO_v3 style detection model on image and video
"""
import time
import colorsys
import os
from timeit import default_timer as timer

import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
from PIL import Image, ImageFont, ImageDraw

from yolo3.model import yolo_eval, yolo_body, tiny_yolo_body
from yolo3.utils import letterbox_image
import os
from keras.utils import multi_gpu_model

class YOLO(object):
    _defaults = {
        "model_path": './model_data/yolo_save.h5',
        "anchors_path": './model_data/my_anchors.txt',
        "classes_path": './model_data/my_classes.txt',
        "score" : 0.1,
        "iou" : 0.5,
        "model_image_size" : (416, 416),
        "gpu_num" : 1,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults) # set up default values
        self.__dict__.update(kwargs) # and update with user overrides
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        self.boxes, self.scores, self.classes = self.generate()

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path, encoding='utf-8') as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors==6 # default setting
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model = tiny_yolo_body(Input(shape=(None,None,3)), num_anchors//2, num_classes) \
                if is_tiny_version else yolo_body(Input(shape=(None,None,3)), num_anchors//3, num_classes)
            self.yolo_model.load_weights(self.model_path) # make sure model, anchors and classes match
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                num_anchors/len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        print('{} model, anchors, and classes loaded.'.format(model_path))

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2, ))
        if self.gpu_num>=2:
            self.yolo_model = multi_gpu_model(self.yolo_model, gpus=self.gpu_num)   # 多GPU模式
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                                           len(self.class_names), self.input_image_shape,
                                           score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    def compute_iou(self, rec1, rec2):
        """
        computing IoU
        :param rec1: (y0, x0, y1, x1), which reflects
                (top, left, bottom, right)
        :param rec2: (y0, x0, y1, x1)
        :return: scala value of IoU
        """
        # computing area of each rectangles
        S_rec1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
        S_rec2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])

        # computing the sum_area
        sum_area = S_rec1 + S_rec2

        # find the each edge of intersect rectangle
        left_line = max(rec1[1], rec2[1])
        right_line = min(rec1[3], rec2[3])
        top_line = max(rec1[0], rec2[0])
        bottom_line = min(rec1[2], rec2[2])

        # judge if there is an intersect
        if left_line >= right_line or top_line >= bottom_line:
            return [0,0]
        else:
            intersect = (right_line - left_line) * (bottom_line - top_line)   # iou面积
            return [(intersect / (sum_area - intersect)) * 1.0 ,intersect]   # 返回iou和面积


    def detect_image(self, image):
        # print(type(image))
        # start = timer()

        if self.model_image_size != (None, None):
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        # print(image_data.shape)
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        # tic = time.time()
        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict=
            {
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })
        # toc = time.time()
        # print('time is:', toc-tic)

        # my code
        num = len(out_scores)
        iou_thresh = 0.5
        currRatio_thresh = 0.7
        index = set()  # 注意创建空集合用set(),而不是{},后者是创建空字典
        if(num>1):
            for i in range(num-1):
                areaI = (out_boxes[i][2] - out_boxes[i][0]) * (out_boxes[i][3] - out_boxes[i][1])   # i的面积
                for j in range(i+1,num):
                    iou_area = self.compute_iou(out_boxes[i], out_boxes[j])
                    if(iou_area[0]>iou_thresh):  # 有重叠
                        if(out_scores[i]>out_scores[j]):
                            index.add(j)
                        else:
                            index.add(i)
                    else:
                        #  or currRatioI>currRatio_thresh or currRatioJ>currRatio_thresh
                        areaJ = (out_boxes[j][2] - out_boxes[j][0]) * (out_boxes[j][3] - out_boxes[j][1])  # j的面积
                        currRatioI = iou_area[1] / areaI
                        currRatioJ = iou_area[1] / areaJ
                        if(out_scores[i]>out_scores[j]):  # j较小且大部分在i中
                            if(currRatioJ>currRatio_thresh or currRatioI>currRatio_thresh):
                                index.add(j)
                        else:
                            if (currRatioJ > currRatio_thresh or currRatioI > currRatio_thresh):
                                index.add(i)

            index = list(index)
            index.sort(reverse=True)  # 降序排列
            out_boxes = list(out_boxes)
            out_scores = list(out_scores)
            out_classes = list(out_classes)
            for i in range(len(index)):
                del out_boxes[index[i]]
                del out_scores[index[i]]
                del out_classes[index[i]]
        # my code

        # area_thresh = 250
        # scare_thresh = 0.4
        flag = 0
        if(len(out_scores)):
            font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                                      size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
            thickness = (image.size[0] + image.size[1]) // 300
            for i, c in reversed(list(enumerate(out_classes))):
                # area = (out_boxes[i][2] - out_boxes[i][0]) * (out_boxes[i][3] - out_boxes[i][1])  # j的面积
                # if (c == 3 or c == 6 or (area < area_thresh and out_scores[i] < scare_thresh)):  # 这是两类正常的 or (area < area_thresh and out_scores[i] < scare_thresh)
                #     continue
                # else:
                #     if(c==4 or c==5):
                #         c = 0  # 两类特殊的RDL类型转移
                flag = 1
                print('Found {} boxes for {}'.format(len(out_boxes), 'img'))
                predicted_class = self.class_names[c]
                box = out_boxes[i]
                score = out_scores[i]

                label = '{} {:.2f}'.format(predicted_class, score)
                draw = ImageDraw.Draw(image)
                label_size = draw.textsize(label, font)

                top, left, bottom, right = box
                top = max(0, np.floor(top + 0.5).astype('int32'))
                left = max(0, np.floor(left + 0.5).astype('int32'))
                bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
                right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
                # print(label, (left, top), (right, bottom))

                if top - label_size[1] >= 0:
                    text_origin = np.array([left, top - label_size[1]])
                else:
                    text_origin = np.array([left, top + 1])

                # My kingdom for a good redistributable image drawing library.
                for i in range(thickness):
                    # print(self.colors[c])
                    draw.rectangle([left + i, top + i, right - i, bottom - i],outline=None)
                draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)],fill=None)
                draw.text(text_origin, label, fill=None, font=font)
                del draw
        # end = timer()
        # print(end - start)
        # return image
        # return image,out_boxes,out_scores,out_classes
        return image, flag

    def close_session(self):
        self.sess.close()

def detect_video(yolo, video_path, output_path=""):
    import cv2
    vid = cv2.VideoCapture(video_path)
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")
    video_FourCC    = int(vid.get(cv2.CAP_PROP_FOURCC))
    video_fps       = vid.get(cv2.CAP_PROP_FPS)
    video_size      = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    isOutput = True if output_path != "" else False
    if isOutput:
        print("!!! TYPE:", type(output_path), type(video_FourCC), type(video_fps), type(video_size))
        out = cv2.VideoWriter(output_path, video_FourCC, video_fps, video_size)
    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    prev_time = timer()
    while True:
        return_value, frame = vid.read()
        image = Image.fromarray(frame)
        image = yolo.detect_image(image)
        result = np.asarray(image)
        curr_time = timer()
        exec_time = curr_time - prev_time
        prev_time = curr_time
        accum_time = accum_time + exec_time
        curr_fps = curr_fps + 1
        if accum_time > 1:
            accum_time = accum_time - 1
            fps = "FPS: " + str(curr_fps)
            curr_fps = 0
        cv2.putText(result, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.50, color=(255, 0, 0), thickness=2)
        cv2.namedWindow("result", cv2.WINDOW_NORMAL)
        cv2.imshow("result", result)
        if isOutput:
            out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    yolo.close_session()

