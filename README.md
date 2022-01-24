# 百度网盘AI大赛——图像处理挑战赛：文档摩尔纹消除第7名方案

比赛链接：  
https://aistudio.baidu.com/aistudio/competition/detail/128/0/introduction

基于Image Demoireing with Learnable Bandpass Filters, CVPR2020（MBCNN）实现去除摩尔纹，我们的方案在aistudio平台上的链接：  
https://aistudio.baidu.com/aistudio/projectdetail/3438269?_=1643006001575&contributionType=1

项目中使用的数据集链接：  
https://aistudio.baidu.com/aistudio/datasetdetail/126450  
https://aistudio.baidu.com/aistudio/datasetdetail/120844  
预训练参数链接：  
https://aistudio.baidu.com/aistudio/datasetdetail/126516

使用项目代码时，需注意对应好各个数据集的路径，训练时使用两个数据路径，分别做traing和test；预测时使用一个数据集路径。

# **一、 赛题介绍**

1. **比赛任务**： 主要任务是建立模型，对比赛给定的带有摩尔纹的图片进行处理，消除屏摄产生的摩尔纹噪声，还原图片原本的样子，并提交模型输出的结果图片。

2. **数据集介绍：** 本次比赛最新发布的数据集共包含训练集、A榜测试集、B榜测试集三个部分，其中训练集共1000个样本，A榜测试集共200个样本，B榜测试集共200个样本； images 为带摩尔纹的源图像数据，gts 为无摩尔纹的真值数据（仅有训练集数据提供gts ，A榜测试集、B榜测试集数据均不提供gts）； images 与 gts 中的图片根据图片名称一一对应。

3. **评价指标：** 本次比赛的评价指标为：

    PSNR （Peak Signal-to-Noise Ratio）

    MSSSIM（Multi-Scale Structural Similarity Index）


### 读取数据集，data目录下有两个文件夹，data120844是官方的数据集，data126450是我们进行数据增强后的训练集
%cd /home/aistudio/data/
### 解压数据集1
!unzip data120844/moire_train_dataset.zip
!unzip data120844/moire_testA_dataset.zip
### 解压数据集2
!unzip data126450/moire_train_dataset_1.zip


### 安装依赖包
!pip install x2paddle  
!pip install scikit-image  
!pip install colour


### work目录下保存模型代码与模型权重参数
%cd /home/aistudio/work/
!unzip paddle_mbcnn.zip


# 二、模型代码

work目录下解压出paddle_mbcnn文件，该文件中保存模型代码；

dataset文件中保存数据处理脚本；

Net文件中保存模型结构；

Util文件中保存模型的算子；

test.py 和 train.py 分别为模型的测试和训练文件；

train_main.py 与 test_main.py设置训练与测试时的参数。






# 训练及评估
### 运行前，安装x2paddle, colour, scikit-image
### 训练中使用了MoireAttack生成的数据，因此training set中的psnr明显低于Test set中的psnr
%cd /home/aistudio/work/
!python paddle_mbcnn/train_main.py





出现结果：

![](https://ai-studio-static-online.cdn.bcebos.com/d358b5da9d6740eda10a9ff967a3bdb0aa726a28f9994e908cc7fc7063d52368)


表示正在开始训练，训练结果保存在 save_path = 的路径文件中




# 三、模型结果与参数调优




**1. 模型结构改进**

官方提供的baseline为：

![](https://ai-studio-static-online.cdn.bcebos.com/cd3cf74d472e4b169f7084754aed1314dfe31eaa6f8d42c6a77ef78a119a13d7)


整体的模型在三个scales上工作，并具有三种不同类型的blocks，分别是波纹纹理去除块（MTRB），全局色调映射块（GTMB）和局部色调映射块（LTMB）。

我们在原有的基础上，增加了模型的尺度，：

![](https://ai-studio-static-online.cdn.bcebos.com/1e70315091df42a89c7b33131584463f50be6e70b3dc4efcae54a771f60ba1dd)


该改进将psnr提升了3个点。






**2. 数据预处理**

由于将原数据作为输入时，处理后的图片效果不佳，因此提出一种数据集的改进方法。为了更好地保留图片细节，针对原有数据进行分块处理，将一张图像切割为多个部分作为输入，步骤如下：

* 首先使用固定尺寸对图像进行切割，我们分别尝试了128* 128，256* 256，512* 512三种尺度大小；（测试时使用的是同一个训练模型，训练1000轮至稳定）；

* 随机改变采样长宽比（其中size的范围为（0.25~2）* 尺度）

效果如下：

![](https://ai-studio-static-online.cdn.bcebos.com/8d9b4665f4544bb4be427691f2b0dba852e03fe91ef04df49fbf06e37bfe3744)





**3. 数据增强**

数据增强（Data Augmentation）是一种通过让有限的数据产生更多的等价数据来人工扩展训练数据集的技术。我们采取了以下方法进行数据增强：

* 对采样的图像进行-20度至20度的旋转处理

* 在clean图片上加入摩尔纹信息，项目地址为：https://github.com/Dantong88/Moire_Attack.

* 利用mosaic拼接方法对原始数据集进行拼接处理

效果如下所示：
![](https://ai-studio-static-online.cdn.bcebos.com/520ee84fc5ab4eb0bfbb298ea5ec6300f61dc8d1afc04d87be4743e13cb930fa)


由于mosaic数据增强使得效果变差，最后没有采用mosaic数据增强。






**4. 数据后处理**

由于数据预处理后的图片是分块的，因而经MBCNN处理后的图片也是分块的，需进行一个拼接处理。但不同分块经MBCNN处理后的色调有不同偏差，直接拼接会产生明显接缝，导致生成图片质量较差，因而需进行平滑过渡处理来消除拼接缝。消除拼接缝的方法有多种，主要的两种有：

  * 中值滤波法消除拼接缝，

* 利用加权平均融合消除拼接缝。

对于拼接缝的消除有两点要求：

* 拼接区域过渡平滑，

* 拼接区域亮度跳跃变化不大。

![](https://ai-studio-static-online.cdn.bcebos.com/d526f2721dae45bf9f574c7ab6ff80ed5c500b2f53644c9da27d24e5a8e1f7cc)


此处采用加权平均融合的方法来处理图片，分别尝试了线性与s型曲线所计算的权重。结果如下：

  
![](https://ai-studio-static-online.cdn.bcebos.com/47aac85207ce43c68f02597af62bdd7550b8fd16a0d94431833d3cf79af3e654)


![](https://ai-studio-static-online.cdn.bcebos.com/5c78a8694cd54f529141ad29acc4add789a51e53b1054f75a73109e0d1c3340a)



**5. 学习率调优**

调整为4e-4，固定值，以加快学习速率，提高训练稳定性（随迭代次数而缩小会更快达到稳定状态，但实际并未收敛）







**6. 训练与测试**

训练代码：

```
cd work/paddle_mbcnn
python train_main.py
```

测试代码:
```
cd work/paddle_mbcnn
python test_main.py
```


精度为：

![](https://ai-studio-static-online.cdn.bcebos.com/4af2b1170c1249e79a98a5a2397ec2d707b80febfbdf406bbcd547f6a084752e)





# 四、 说明

本次代码采用了官方的base 和 moire attack 项目的代码，
项目地址为：

https://github.com/zhenngbolun/Learnbale_Bandpass_Filter

https://github.com/Dantong88/Moire_Attack.
