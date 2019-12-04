import sys
import argparse
from yolo import YOLO, detect_video
from PIL import Image

import glob
import os
from shutil import copyfile

src_dir="F:/比赛事宜/裂纹识别/复赛数据/challengedataset-semifinal/test/test"
dst_dir="F:/比赛事宜/裂纹识别/复赛数据/challengedataset-semifinal/test/seg"
k=0
file_1=open('result_11_24.txt')
file_2=open('result_11_22_98.44.txt')
result_1=[]
result_2=[]
for line in file_1.readlines():
    curLine=line.strip().split(" ")
    result_1.append(curLine[1])

for line in file_2.readlines():
    curLine = line.strip().split(" ")
    result_2.append(curLine[1])

with open("image_name.txt","w") as w:
    with open("vote.txt", "w") as f:
        for i in range(len(result_1)):
            if (result_1[i] == "1") and (result_2[i] == "1"):
                f.write("{}.jpg {}\n".format(i + 1, 1))
                w.write("{}.jpg {}\n".format(i + 1, 1))
                k = k + 1
                copyfile(src_dir+"/"+str(i+1)+".jpg", dst_dir+"/"+str(i+1)+".jpg")
            else:
                f.write("{}.jpg {}\n".format(i + 1, 0))

print(k)


