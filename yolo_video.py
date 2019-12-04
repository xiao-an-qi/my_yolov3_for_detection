import sys
from keras.preprocessing.image import array_to_img
import argparse
from yolo import YOLO, detect_video
from PIL import Image, ImageFont, ImageDraw
from PIL import ImageFilter
from PIL import ImageEnhance

import numpy as np
import os
import cv2


# 这个代码可以进行单张图像的显示
# def detect_img(yolo):
#     while True:
#         img = input('Input image filename:')
#         try:
#             image = Image.open(img)
#         except:
#             print('Open Error! Try again!')
#             continue
#         else:
#             r_image = yolo.detect_image(image)  # return image,out_boxes,out_scores,out_classes
#             r_image.show()
#     yolo.close_session()


# 测试图像批量运行与保存
# def detect_img(yolo):
#     class_list = ['RDL','partical','PI','solder','crack','pie']   # 预定义类别
#     save_address = 'C:/Users/肖安七/Desktop/test1/'
#     f = open('2007_val.txt','r')
#     datas = f.readlines()
#     f.close()
#     for line in datas:
#         Class = class_list[int(line.strip().split(',')[-1])]  # 正确类别
#         address = line.split()[0]
#         image = Image.open(address)
#         detect_image = yolo.detect_image(image)
#
#         draw = ImageDraw.Draw(detect_image)
#         font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
#                                   size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
#         text = 'ground truth:'+ Class
#         xy = np.array([30, 30])
#         draw.text(xy, text, fill=None, font=font)
#         name = address.split('/')[-1]
#         detect_image.save(save_address + name)
#     yolo.close_session()

# 图像文件夹测试
def detect_img(yolo):
    path = 'E:/deep_learning_datas/华天数据/Defect library/RDL_Defect/'
    save_path_ng = 'E:/deep_learning_datas/华天数据/Defect library/result_ng/'
    save_path_ok = 'E:/deep_learning_datas/华天数据/Defect library/result_ok/'
    filelist = os.listdir(path)
    for item in filelist:
        address = path + item
        image = Image.open(address)
        detect_image = yolo.detect_image(image)
        if(detect_image[1]):
            detect_image[0].save(save_path_ng + item)
        else:
            detect_image[0].save(save_path_ok + item)
    yolo.close_session()

FLAGS = None

if __name__ == '__main__':
    # class YOLO defines the default value, so suppress any default here
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    '''
    Command line options
    '''
    parser.add_argument(
        '--model', type=str,
        help='path to model weight file, default ' + YOLO.get_defaults("model_path")
    )

    parser.add_argument(
        '--anchors', type=str,
        help='path to anchor definitions, default ' + YOLO.get_defaults("anchors_path")
    )

    parser.add_argument(
        '--classes', type=str,
        help='path to class definitions, default ' + YOLO.get_defaults("classes_path")
    )

    parser.add_argument(
        '--gpu_num', type=int,
        help='Number of GPU to use, default ' + str(YOLO.get_defaults("gpu_num"))
    )

    parser.add_argument(
        '--image', default=True, action="store_true",
        help='Image detection mode, will ignore all positional arguments'
    )
    '''
    Command line positional arguments -- for video detection mode
    '''
    parser.add_argument(
        "--input", nargs='?', type=str,required=False,default='1.mp4',
        help = "Video input path"
    )

    parser.add_argument(
        "--output", nargs='?', type=str, required=False,default="11.mp4",
        help = "[Optional] Video output path"
    )

    FLAGS = parser.parse_args()

    if FLAGS.image:
        """
        Image detection mode, disregard any remaining command line arguments
        """
        print("Image detection mode")
        if "input" in FLAGS:
            print(" Ignoring remaining command line arguments: " + FLAGS.input + "," + FLAGS.output)
        detect_img(YOLO(**vars(FLAGS)))
    elif "input" in FLAGS:
        detect_video(YOLO(**vars(FLAGS)), FLAGS.input, FLAGS.output)
    else:
        print("Must specify at least video_input_path.  See usage with --help.")
