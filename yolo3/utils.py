"""Miscellaneous utility functions."""

from functools import reduce
import cv2
import random
import numpy as np
from PIL import Image, ImageFilter
from data_aug.data_aug import *
from data_aug.bbox_util import *
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb

def compose(*funcs):
    """Compose arbitrarily many functions, evaluated left to right.

    Reference: https://mathieularose.com/function-composition-in-python/
    """
    # return lambda x: reduce(lambda v, f: f(v), funcs, x)
    if funcs:
        return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)
    else:
        raise ValueError('Composition of empty sequence not supported.')


def letterbox_image(image, size):
    '''resize image with unchanged aspect ratio using padding'''
    iw, ih = image.size
    w, h = size
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)

    image = image.resize((nw,nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128,128,128))
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    return new_image


def rand(a=0, b=1):   # 随机产生（0，1）之间的数
    return np.random.rand()*(b-a) + a


# 输入一张图像，图像中所有的bb(array格式)，输出增强后的图像和对应的bb
def aug_image_sequence(annotation_line, input_shape, num):
    line = annotation_line.split()
    box = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]])  # 框

    image = cv2.imread(line[0])  # 输入图像
    ih, iw = image.shape[:2]   # 图像实际尺寸
    h, w = input_shape         # 图像目标尺寸
    if (iw != w):   # 对图像要进行缩放
        image = cv2.resize(image, (w, h), interpolation=cv2.INTER_CUBIC)

        # correct boxes
        scaleW = w / iw
        scaleH = h / ih
        box[:, [0, 2]] = box[:, [0, 2]] * scaleW
        box[:, [1, 3]] = box[:, [1, 3]] * scaleH

    if num == 0:  # 取原图进行处理
        image_data = np.array(image) / 255.
        box_data = box
        return image_data, box_data

    elif num == 1:  # 水平镜像                            ################### 镜像
        image, box = HorizontalFlip()(image, box)

    elif num == 2:  # 垂直镜像
        image, box = VertialFlip()(image, box)

    elif num == 3:  # 水平镜像 + 垂直镜像
        image, box = HorizontalFlip()(image, box)
        image, box = VertialFlip()(image, box)

    elif num == 4:  # change hsv                         #################### 镜像 + HSV
        image, box = RandomHSV(15, 15, 15)(image, box)

    elif num == 5:  # 水平镜像 + HSV
        image, box = HorizontalFlip()(image, box)
        image, box = RandomHSV(20, 20, 20)(image, box)

    elif num == 6:  # 垂直镜像 + HSV
        image, box = VertialFlip()(image, box)
        image, box = RandomHSV(25, 25, 25)(image, box)

    elif num == 7:  # 水平镜像 + 垂直镜像 + HSV
        image, box = HorizontalFlip()(image, box)
        image, box = VertialFlip()(image, box)
        image, box = RandomHSV(10, 10, 10)(image, box)

    elif num == 8:  # blur and add noise                ######################## 镜像 + blur and add noise
        image, box = RandomBlur(1)(image, box)

    elif num == 9:  # 水平镜像 + blur and add noise
        image, box = HorizontalFlip()(image, box)
        image, box = RandomBlur(1)(image, box)

    elif num == 10:  # 垂直镜像 + blur and add noise
        image, box = VertialFlip()(image, box)
        image, box = RandomBlur(1)(image, box)

    elif num == 11:  # 水平镜像 + 垂直镜像 + blur and add noise
        image, box = HorizontalFlip()(image, box)
        image, box = VertialFlip()(image, box)
        image, box = RandomBlur(1)(image, box)

    elif num == 12:  # rotate                            ######################## 镜像 + rotate
        image, box = RandomRotate(5)(image, box)

    elif num == 13:  # 水平镜像 + rotate
        image, box = HorizontalFlip()(image, box)
        image, box = RandomRotate(6)(image, box)

    elif num == 14:  # 垂直镜像 + rotate
        image, box = VertialFlip()(image, box)
        image, box = RandomRotate(7)(image, box)

    elif num == 15:  # 水平镜像 + 垂直镜像 + rotate
        image, box = HorizontalFlip()(image, box)
        image, box = VertialFlip()(image, box)
        image, box = RandomRotate(8)(image, box)

    elif num == 16:  # shearing                         ######################### 镜像 + shearing
        image, box = RandomShear(0.15)(image, box)

    elif num == 17:  # 水平镜像 + shearing
        image, box = HorizontalFlip()(image, box)
        image, box = RandomShear(0.15)(image, box)

    elif num == 18:  # 垂直镜像 + shearing
        image, box = VertialFlip()(image, box)
        image, box = RandomShear(0.15)(image, box)

    elif num == 19:  # 水平镜像 + 垂直镜像 + shearing
        image, box = HorizontalFlip()(image, box)
        image, box = VertialFlip()(image, box)
        image, box = RandomShear(0.15)(image, box)

    elif num == 20:  # scale
        image, box = RandomScale(0.15, diff=True)(image, box)

    elif num == 21:  # resize
        image, box = RandomTranslate(0.15, diff=True)(image, box)

    # max_boxes = len(box)
    # box_data1 = np.zeros((max_boxes, 5))
    # for i in range(max_boxes):
    #     box_data1[i,:] = list(box[i])

    image_data = np.array(image) / 255.
    box_data = box

    return image_data, box_data


def get_random_data(annotation_line, input_shape, random=True, max_boxes=10, jitter=.3, hue=.1, sat=1.5, val=1.5, proc_img=True):
    '''random preprocessing for real-time data augmentation'''
    # annotation_line = 'D:/PYCHARM_SOLUTION/data6/VOCdevkit/VOC2007/JPEGImages/_0_326.jpg 367,263,397,295,0 406,305,429,332,0'
    line = annotation_line.split()
    image = Image.open(line[0])  # 输入图像
    iw, ih = image.size
    h, w = input_shape
    box = np.array([np.array(list(map(int,box.split(',')))) for box in line[1:]])   # 框

    if not random:
        # resize image
        scale = min(w/iw, h/ih)    # scale = w/iw 得到浮点小数
        nw = int(iw*scale)         # nw = 416
        nh = int(ih*scale)         # nh = 249
        dx = (w-nw)//2       # dx = 0
        dy = (h-nh)//2       # dy = 83
        image_data=0
        if proc_img:
            image = image.resize((nw,nh), Image.BICUBIC)
            new_image = Image.new('RGB', (w,h), (128,128,128))
            new_image.paste(image, (dx, dy))           # 有效图像部分移到中间，其余为128填充
            image_data = np.array(new_image)/255.

        # correct boxes
        box_data = np.zeros((max_boxes,5))
        if len(box)>0:
            np.random.shuffle(box)
            if len(box)>max_boxes: box = box[:max_boxes]
            box[:, [0,2]] = box[:, [0,2]]*scale + dx
            box[:, [1,3]] = box[:, [1,3]]*scale + dy
            box_data[:len(box)] = box

        return image_data, box_data

    # resize image
    new_ar = w/h * rand(1-jitter,1+jitter)/rand(1-jitter,1+jitter)    # 利用抖动产生更多数据
    scale = rand(.5, 2)   # 尺寸缩放
    if new_ar < 1:
        nh = int(scale*h)     # 首先对高度进行缩放，然后宽度在高度的基础上进行缩放
        nw = int(nh*new_ar)   # nh > nw
    else:
        nw = int(scale*w)
        nh = int(nw/new_ar)   # nh > nw
    image = image.resize((nw,nh), Image.BICUBIC)

    # place image
    dx = int(rand(0, w-nw))   # 随机产生x,x方向偏移
    dy = int(rand(0, h-nh))
    new_image = Image.new('RGB', (w,h), (128,128,128))
    new_image.paste(image, (dx, dy))
    image = new_image         # 有效图像部分进行拼接与偏移

    # flip image or not
    flipH = rand()<.5
    if flipH: image = image.transpose(Image.FLIP_LEFT_RIGHT)   # 水平镜像
    flipV = rand() < .5
    if flipV: image = image.transpose(Image.FLIP_TOP_BOTTOM)   # 竖直镜像

    # blur and add noise
    blur_flag = rand()<.5
    if blur_flag:
        image = image.filter(ImageFilter.GaussianBlur(radius=3))  # radius指定平滑半径，也就是模糊的程度。

    # distort image
    hue = rand(-hue, hue)    # 通过调整色调来产生更多数据
    sat = rand(1, sat) if rand()<.5 else 1/rand(1, sat)     # 通过调整饱和度来产生更多数据
    val = rand(1, val) if rand()<.5 else 1/rand(1, val)     # 通过调整亮度来产生更多数据
    x = rgb_to_hsv(np.array(image)/255.)
    x[..., 0] += hue
    x[..., 0][x[..., 0]>1] -= 1
    x[..., 0][x[..., 0]<0] += 1
    x[..., 1] *= sat
    x[..., 2] *= val
    x[x>1] = 1
    x[x<0] = 0
    image_data = hsv_to_rgb(x)    # numpy array, 0 to 1

    # correct boxes
    box_data = np.zeros((max_boxes,5))
    if len(box)>0:
        np.random.shuffle(box)
        box[:, [0,2]] = box[:, [0,2]]*nw/iw + dx
        box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy
        if flipH: box[:, [0,2]] = w - box[:, [2,0]]
        if flipV: box[:, [1,3]] = h - box[:, [3,1]]
        box[:, 0:2][box[:, 0:2]<0] = 0
        box[:, 2][box[:, 2]>w] = w
        box[:, 3][box[:, 3]>h] = h
        box_w = box[:, 2] - box[:, 0]    # 框的宽度
        box_h = box[:, 3] - box[:, 1]    # 框的高度
        box = box[np.logical_and(box_w>1, box_h>1)]      # discard invalid box
        if len(box)>max_boxes: box = box[:max_boxes]
        box_data[:len(box)] = box

    return image_data, box_data
