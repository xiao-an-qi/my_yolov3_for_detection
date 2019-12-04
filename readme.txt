1、图像打标：把.jpg文件放到JPEGImage中，把相应的.xml文件放到Annotations文件夹中；
2、运行make_data.py把labelImg生成的.jpg文件和.xml文件转为txt文件；
3、运行voc_annotation.py将第二步生成的txt文件转为coco数据格式，注意修改类别行为自己的类别，
   并根据自己的路径修改存放训练数据集。运行后会生成三个txt文件，分别对应train.txt、test.txt、val.txt；
4、运行kmeans.py，生成yolo_anchors.txt，存放到model_data文件夹内；
5、修改yolov3.cfg文件中的以下位置：
 最开始几行，注释掉Testing的设置，打开Training的设置，并根据自己显卡的情况合理设置batch大小；
 修改classes类别数，总共有三处地方，改为自己对应的类别数，修改[yolo]上边对应的filters值，也是有三处地方要修改，修改为3*（类别数+5），
   原来是80类，对应的值是255，假如自己有2类，则应改为21。
6、在终端运行python convert.py yolov3.cfg yolov3.weights model_data/yolo.h5
   将yolo3网络结构和参数转化为keras接受的格式，此即为预训练参数；
7、模型训练，训练完成  后在logs/000/文件夹下，生成trained_weights_final.h5;
8、拷贝上述h5文件到model_data文件夹下，重命名为yolo.h5，然后运行yolo_viedo.py文件进行测试。
