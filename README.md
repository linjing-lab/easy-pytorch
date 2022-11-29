# easy-pytorch🔖

<p align='center'>
    <a href="https://github.com/pytorch"> 
        <img src="https://www.vectorlogo.zone/logos/pytorch/pytorch-icon.svg"> 
    </a>
</p>

这是一个快速入门 PyTorch 框架的学习仓库，包括 PyTorch 的基本用法与常用模型的搭建等。

## 目录

<div align="center">

|名称（name）|数据集（datasets）|准确率（accuracy）|解释（explain）|
|--|--|--|--|
|[PyTorch基础](./introduce/pytorch_basics.ipynb)|/|/|PyTorch语法与概念|
|[线性回归](./introduce/linear_regression.ipynb)|/|/||
|[逻辑回归](./introduce/logistic_regression.ipynb)|[MNIST](./data/MNIST/)|92.17%||
|[前馈神经网络](./introduce/feedforward_neural_network.ipynb)|[MNIST](./data/MNIST/)|97.16%|前向传播|
|[简单卷积神经网络](./introduce/convolutional_neural_network.ipynb)|[MNIST](./data/MNIST/)|98.8%||
|[LeNet-5](./networks/lenet-5.ipynb)|[MNIST](./data/MNIST/)|99.04%|用于银行柜机手写数字识别的CNN模型|
|[循环神经网络](./networks/recurrent_neural_network.ipynb)|[MNIST](./data/MNIST/)|97.01%|用于处理和预测时序数据|
|[AlexNet](./networks/alexnet.ipynb)|[CIFAR10](./data/CIFAR10/)|86.1%|ImageNet比赛提出一个5层CNN模型|
|[VGGNet](./networks/vggnet.ipynb)|[CIFAR10](./data/CIFAR10/)|VGG-16: 92.23%<br />VGG-19: 91.99%|加深版的AlexNet|
|[GoogLeNet](./networks/googlenet.ipynb)|[CIFAR10](./data/CIFAR10/)|aux_logits=True: 86.99%<br />aux_logits=False: 85.88%|首次引入Inception结构|
|[ResNet](./networks/resnet.ipynb)|[CIFAR10](./data/CIFAR10/)|89.89%|引入残差块|
|[Networks Comparison with TensorBoard](./networks/comparison.ipynb)|[CIFAR10](./data/CIFAR10/)|/|多个CNN模型的对比|
|[Object Detection](./video_detection.ipynb)|[VIDEOS](./data/VIDEOS/input/)|/|基于YOLOv5s模型的目标检测|

</div>

## LICENSE
[MIT LICENSE](./LICENSE)