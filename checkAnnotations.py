# 获取文件夹中的文件路径
import os
def getFilePathList(dirPath, partOfFileName=''):
    allFileName_list = list(os.walk(dirPath))[0][2]
    fileName_list = [k for k in allFileName_list if partOfFileName in k]
    filePath_list = [os.path.join(dirPath, k) for k in fileName_list]
    return filePath_list

# 此段代码检查标记好的文件夹是否有图片漏标，就是检查图片与xml文件是否对应
def check_1(dirPath):
    jpgFilePath_list = getFilePathList(dirPath, '.jpg')
    allFileMarked = True
    xmlFilePath = './VOCdevkit/VOC2007/Annotations/'
    for jpgFilePath in jpgFilePath_list:
        xml_name = jpgFilePath.split('/')[-1].split('.')[0]
        temp_xmlFilePath = xmlFilePath + xml_name + '.xml'
        if not os.path.exists(temp_xmlFilePath):
            print('%s this picture is not marked.' %jpgFilePath)
            allFileMarked = False
    if allFileMarked:
        print('congratulation! it is been verified that all jpg file are marked.')

# 此段代码检查标记的xml文件中是否有物体标记类别拼写错误        
import xml.etree.ElementTree as ET
def check_2(dirPath, className_list):
    className_set = set(className_list)
    xmlFilePath_list = getFilePathList(dirPath, '.xml')
    allFileCorrect = True
    for xmlFilePath in xmlFilePath_list:
        with open(xmlFilePath) as file:
            fileContent = file.read()
        root = ET.XML(fileContent)
        object_list = root.findall('object')
        for object_item in object_list:
            name = object_item.find('name')
            className = name.text
            if className not in className_set:
                print('%s this xml file has wrong class name "%s" ' %(xmlFilePath, className))
                allFileCorrect = False
    if allFileCorrect:
        print('congratulation! it is been verified that all xml file have right class name.')

# 此段代码检测标记的box是否超过图片的边界
# 如果有此类型的box，则直接删除与box相关的xml文件和图片文件
from PIL import Image
def check_3(dirPath, suffix):
    xmlFilePath_list = getFilePathList(dirPath, '.xml')
    allFileCorrect = True
    for xmlFilePath in xmlFilePath_list:
        imageFilePath = xmlFilePath[:-4] + '.' + suffix.strip('.')
        image = Image.open(imageFilePath)
        width, height = image.size
        with open(xmlFilePath) as file:
            fileContent = file.read()
        root = ET.XML(fileContent)
        object_list = root.findall('object')
        for object_item in object_list:
            bndbox = object_item.find('bndbox')
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)
            if xmin>=1 and ymin>=1 and xmax<=width and ymax<=height:
                continue
            else:
                os.remove(xmlFilePath)
                os.remove(imageFilePath)
                allFileCorrect = False
                break
    if allFileCorrect:
        print('congratulation! it is been verified that all xml file have right box.')

# 主函数    
if __name__ == '__main__':
    dirPath = './VOCdevkit/VOC2007/JPEGImages/'      # 待检查地址
    check_1(dirPath)

    className_list = ['RDL', 'partical', 'PI', 'solder', 'crack', 'pie', 'cloud']
    check_2(dirPath, className_list)

    suffix = 'jpg'
    check_3(dirPath, suffix)
    