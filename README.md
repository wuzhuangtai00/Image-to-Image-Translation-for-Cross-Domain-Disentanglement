# Image-to-image translation for cross-domain disentanglement

项目名称：Image-to-image translation for cross-domain disentanglement复现

参考论文：Image-to-image translation for cross-domain disentanglement, NeurIPS 2018

项目成员：吴瑾昭, 任轩笛, 顾宇晨, 邱元辰

### 要求

- Python 3.7

- Tensorflow 1.13.1

- Tensorlayer 1.11.1


### 说明

本项目用tensorlayer实现了论文中的encoder、decoder、exclusive_decoder和discriminator模块。

模型参考自[论文作者的开源项目](https://github.com/agonzgarc/cross-domain-disen)的实现。

训练部分沿用了[pix2pix](https://github.com/affinelayer/pix2pix-tensorflow/)，详见[这里](https://affinelayer.com/pix2pix/)。

### 训练结果

我们使用MNIST-CDCB训练集进行训练,部分训练结果如下:

![image-20200214202624761](./readme_image/image-20200214202624761.png)

![image-20200214202911406](./readme_image/image-20200214202911406.png)

剩余的训练结果因为太大不便展示,我们挑选了其中前50张作为样例放在result_sample/文件夹下