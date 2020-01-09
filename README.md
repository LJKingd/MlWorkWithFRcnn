# MlWorkWithFRcnn

## Introduction

* 本大作业的项目使用了 faster Rcnn模型，使用多种方法实现解决样本不平衡。
* 本次实验采取的过采样方法，通过对core类图像操作生成新的图片，使2个类别的数量大致平衡。
* 生成新样本的方法及命名方式如下：
镜像（左右）:core_battery0001xxxx
旋转45度 :core_battery0002xxxx
旋转90度 :core_battery0003xxxx
平移:core_battery0004xxxx
缩放:core_battery0005xxxx
镜像（上下）:core_battery0006xxxx
高斯扰动:core_battery0007xxxx
颜色扰动:core_battery0008xxxx
锐化:core_battery0009xxxx
其中xxxx为所变换的图片的名称中的值。剔除不合格的样本，最终得到了约4200个新样本，此时带芯与不带芯充电宝数量大概为1：1。

## 操作方法
* 下载本项目代码和最终训练的模型，将最终训练的模型放到 models\vgg16\pascal_voc 下
* 使用pip命令安装所有的python依赖包：
 ```
 pip install -r requirements.txt
```
*Tips：要注意如果出现can’t import ‘imread’，请检查scipy的版本是否超过了1.0，因为1.0以上的版本里，scipy不再使用imread。并且推荐更改pillow版本为5.2.0。而且scipy和pillow要在同一路径下
 ```
 pip uninstall scipy
pip install scipy==1.0

pip uninstall pillow
pip install pillow==5.2.0 
```
* 测试：将测试的图片放到任意位置，输入下列命令测试
 ```
 python test_net.py --img_dir [图片绝对路径] --label_dir [标签绝对路径] --cuda
```

## 实验结果
![结果](https://github.com/LJKingd/MlWorkWithFRcnn/blob/master/re.png)

## 团队分工 
* 武易：组长，负责团队协调工作，寻找合适的模型，解决不平衡问题
* 耿琛明：负责生成平移数据，生成过采样数据
* 梁金：修改模型输入输出接口，调试模型运行环境，编写生成voc数据程序，
* 何馨蕾：负责生成过采样数据，对原数据进行高斯噪声和颜色扰动，
* 邓玉玲：研究不平衡问题，对core样本采用裁剪生成新的样本，编写实验报告总结
