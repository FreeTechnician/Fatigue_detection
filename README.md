# 注意力检测模型

## 当前进度：

2020.7.25 提高关键点识别精度

## 基本原理：

首先通过人脸关键点侦测，对关键点进行分析判断目光方向，头的朝向，是否犯困以及是否说话

![21](https://github.com/FreeTechnician/Fatigue_detection/blob/master/img/21.jpg)

关键点检测是采用MTCNN的思想，先用小网络检测人脸，再用大网络回归关键点坐标

## 环境配置：

windows 10 64位

Python 3.7

Pytorch 1.4.0

PIL 7.0.0

OpenCV 4.2.0

Numpy 1.18.1

cuda10.1、cudnn7.6.5

## 数据集：

数据集采用WFLW

下载地址：https://wywu.github.io/projects/LAB/WFLW.html

## 代码文件：

### imgdata.py

该文件用于查看数据集

| 数据集原图：                                                 | 数据集与标签：                                               |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| <img src="https://github.com/FreeTechnician/Fatigue_detection/blob/master/img/377.jpg" alt="51_Dresses_wearingdress_51_377" style="zoom:50%;" /> | <img src="https://github.com/FreeTechnician/Fatigue_detection/blob/master/img/378.jpg" alt="`75TUXZ4[3ZKG_Z%O]JS6{M" style="zoom:50%;" /> |

### dataset.py

对数据集进行简单的处理（根据IOU随机截取正样本，部分样本和负样本）

### adddata_blur.py、adddata_brightness.py、adddata_shift.py

这三个文件分别通过均值滤波、改变光照对比度、双边滤波进行数据集扩充

| 原图：                                                       | 均值滤波：                                                   | 改变光照对比度：                                             | 双边滤波：                                                   |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| ![](https://github.com/FreeTechnician/Fatigue_detection/blob/master/img/0.jpg) | ![](https://github.com/FreeTechnician/Fatigue_detection/blob/master/img/30.jpg) | ![2](https://github.com/FreeTechnician/Fatigue_detection/blob/master/img/10.jpg) | ![0](https://github.com/FreeTechnician/Fatigue_detection/blob/master/img/20.jpg) |

对WFLW数据集进行简单的处理，生成训练数据集

### adddata_blur.py、adddata_brightness.py、adddata_shift.py

这三个文件都是对样本进行扩充，分别

### simpling.py

制作DataLoader

### net.py

网络结构，包含P_net 与MainNet

P_net采用MTCNN中的P网络，对图片进行简单的人脸识别，寻找人脸

MainNet为MTCNN中的O网络的升级版，扩大了输入大小，增加了网络深度

网络都采用kaiming_normal_进行初始化

### tool.py

该文件为常用算法包，包括IOU与NMS，以及图像正方形化

在P_Net中IOU = 交集/并集

在MainNet中IOU = 最小集/并集

### trainer.py、net_trainer.py

这两个文件为MainNet训练代码

trainer.py同样适用于P_Net，但本次P网络是直接采用MTCNN中的P网络（懒得重新训练了……），所以没有直接训练代码

### detect.py

该文件为侦测代码，用于侦测图片

效果图：

![19](https://github.com/FreeTechnician/Fatigue_detection/blob/master/img/19.jpg)

### video_detect.py

该文件用于视频侦测

