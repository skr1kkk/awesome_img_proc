import os

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from model.UNET import UNet
from model.real_UNET import REAL_UNET
"""
导入所需的Python库：
torch 是PyTorch的主库，用于构建和训练神经网络
torch.nn 是一个子模块，提供构建神经网络所需的层和函数
torch.optim 是另一个子模块，包含优化算法，如SGD、Adam等。
Dataset 和 DataLoader 是辅助加载和迭代数据集的类。
transforms 用于数据预处理和增强
Image 是PIL库中的一个类，用于处理图像
os 是Python的标准库，用于处理文件和目录
"""


# 1. 数据集类
class ImageDataset(Dataset):
    # 为了方便地从指定目录加载成对的图像
    def __init__(self, a_dir, b_dir, transform=None):
        self.a_dir = a_dir  # 图像a的文件夹路径
        self.b_dir = b_dir  # 图像b的文件夹路径
        self.transform = transform  # 要应用于这些图像的预处理转换
        # 将传入的参数赋值给类的实例变量，在类的其他方法中可以使用这些变量
        self.images = os.listdir(a_dir)

    #  使用 os.listdir 函数列出 a_dir 文件夹中的所有文件，这些文件的名称被存储在 self.images 实例变量中

    # ImageDataset类继承自Dataset
    # 用于定义如何加载和处理数据集
    # 构造函数__init__初始化类实例，并设置图像目录和转换

    def __len__(self):
        return len(self.images)

    # 返回self.images列表的长度，也就是在a_dir目录下找到的图像文件数量
    # 意味着len(dataset)（其中 dataset是 ImageDataset类的一个实例）将返回该数据集中的图像总数
    # PyTorch数据加载框架的一个重要部分：允许DataLoader知道数据集的大小，可以在训练神经网络时进行有效的批处理和迭代
    # 在使用DataLoader加载数据集时，它会调用这个__len__方法来确定需要迭代多少次才能覆盖整个数据集

    def __getitem__(self, idx):
        # # __getitem__方法定义了如何获取单个数据项，读取对应的图像对（a和b），应用转换（如果有的话），并返回它们
        # Dataset 对象的 __getitem__ 方法用于获取数据集的单个项
        # 在ImageDataset类中，这个方法被用来获取成对的图像——一张来自目录a_dir，另一张来自目录b_dir。
        a_img = Image.open(os.path.join(self.a_dir, self.images[idx]))
        b_img = Image.open(os.path.join(self.b_dir, self.images[idx]))
        # self.images[idx] 表示 a_dir 目录中第 idx 个图像的文件名：a_dir 和 b_dir 中的图像是成对的，文件名相同
        if self.transform:
            a_img = self.transform(a_img)
            b_img = self.transform(b_img)
        return a_img, b_img
        # 使得 ImageDataset 类的每个实例都可以像列表一样使用索引来访问单个图像对


# 2. 构建模型
class SimpleCNN(nn.Module):  # 后面找 U-NET， FasterCNN, RCNN... 不同算法模型，有不同的这个model实现
    # SimpleCNN 类定义了神经网络的结构，继承自 nn.Module
    # 在 __init__ 方法中，定义了网络的各层
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 16, 3, padding=1)
        self.conv4 = nn.Conv2d(16, 3, 3, padding=1)
        # 2D卷积层，用于在图像上执行卷积操作
        # 参数分别表示：输入通道数（3通道RGB）、输出通道数、卷积核的大小和填充大小
        self.relu = nn.ReLU()
        # ReLU用于引入非线性
        self.pool = nn.MaxPool2d(2, 2)
        # 最大池化层用于下采样图像

    # 首先调用基类的初始化方法，然后定义该网络的各个层
    # 定义了4个卷积层（nn.Conv2d）、1个ReLU激活函数（nn.ReLU）和1个最大池化层（nn.MaxPool2d）

    # forward 方法定义了数据通过网络的前向传播过程
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        # 通过第1个卷积层 self.conv1 对输入 x 进行处理。然后，应用ReLU激活函数 (self.relu) 添加非线性，并通过最大池化层 (self.pool) 进行下采样
        x = self.pool(self.relu(self.conv2(x)))
        # 通过第2个卷积层 self.conv2 ，对输出执行相同的操作序列：卷积，ReLU激活，然后最大池化
        x = self.relu(self.conv3(x))
        # 通过第3个卷积层 self.conv3 后应用ReLU激活函数  注意，这一步没有最大池化
        x = self.conv4(x)
        # 通过最后一个卷积层 self.conv4
        return x
    # 数据通过网络时的前向传播逻辑
    # forward方法描述了数据从输入到输出经过模型时的整个处理流程
    # 在这个特定的SimpleCNN实现中，使用了多个卷积层、ReLU激活函数和池化层————构建卷积神经网络的典型方法

# 3. 数据加载
transform = transforms.Compose([transforms.Resize((400, 1200)), transforms.ToTensor()])
# 定义了用于预处理图像的转换：使用了 transforms.Compose 来组合多个图像转换操作，在这个例子中只使用了一个转换：transforms.ToTensor()
# 会将PIL图像或NumPy数组转换为PyTorch张量，并将像素值从0-255缩放到0-1之间，实现归一化

train_dataset = ImageDataset(a_dir='a_pic', b_dir='b_pic', transform=transform)
# 创建了一个 ImageDataset 实例
# a_dir 和 b_dir 是两个目录的路径，分别包含成对的图像a和图像b：这些图像应该是一一对应的，即 a_dir 中的每个图像都有一个对应的图像在 b_dir 中，文件名相同
# transform 参数应用了之前定义的预处理转换
# a_dir 训练集 目录，b_dir 验证集 目录：两组成对图像的目录，而不是传统意义上的训练集和验证集
# 保证文件名一样（比如训练集 xxa.png, 验证集 xxa.png)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
# 这行代码创建了一个 DataLoader 实例，用于从 train_dataset 加载数据
# batch_size=2 指定了每个批量包含2个图像对
# shuffle=True 表明在每个epoch开始时，数据将被打乱：有助于模型训练时的泛化能力
# 这两行创建了一个 ImageDataset 实例来加载数据，并使用 DataLoader 来批量获取数据和洗牌

# 4. 初始化模型、损失函数（均方误差损失），并设置了优化器（Adam）
# 配置神经网络模型进行训练的标准步骤
# 在后续的训练循环中，将使用这个模型(model)，应用损失函数(criterion)来计算损失，然后使用优化器(optimizer)来更新模型的权重，以最小化损失

# model = SimpleCNN()
# model = UNet()
model = REAL_UNET(in_channels=3, num_classes=3)

# U-Net是一种流行的卷积神经网络，通常用于图像分割任务
# 这里假设已经定义了一个名为 UNet 的类，该类实现了U-Net网络的结构

criterion = nn.MSELoss()
# 定义了损失函数，这里使用的均方误差损失（Mean Squared Error, MSE）
# 在PyTorch中，nn.MSELoss() 创建了一个损失函数的实例，用于计算模型输出和目标值之间的均方误差
# 这种损失函数通常用于回归任务
optimizer = optim.Adam(model.parameters(), lr=0.001)
# 创建了一个优化器实例，用于优化模型的参数
# 使用的是Adam优化器，是一种基于梯度下降的优化算法，广泛用于训练深度学习模型
# model.parameters() 获取模型中所有可训练的参数，lr=0.001 设置了学习率为0.001
# 学习率是一个重要的超参数，控制了模型在每次迭代中参数更新的幅度



# 5. 训练模型
# 训练循环
num_epochs = 5
# num_epochs 定义了训练的轮数（epoch）
# 1个epoch意味着整个训练数据集将被遍历1次
for epoch in range(num_epochs):
    # 这个for循环将对整个训练数据集进行迭代，每次迭代称为一个epoch
    # 迭代次数，通常叫做epoch，每个epoch会把所有训练数据集的图片逐张输入模型训练
    for i, data in enumerate(train_loader, 0):
        # 内部循环遍历train_loader提供的训练数据
        # train_loader ————加载整个训练集图片
        # 一个 DataLoader 实例，负责批量加载训练数据，并且可以进行数据洗牌和多线程处理
        inputs, targets = data
        # 从 data 中分离出输入图像（inputs）和目标图像（targets）
        # 对于成对图像的任务，inputs 是带有干扰的图像，targets 是期望模型输出的清晰图像
        optimizer.zero_grad()  # 优化器（不用关注）
        # 在每次的参数更新之前，需要将梯度清零
        # 默认情况下，梯度是累加的，为防止上一次的梯度影响这一次的计算，需要手动清零

        outputs = model(inputs)
        # 模型入口，input是一个图片（对应带干扰图），output是model想要生成的图片
        # 通过模型进行前向传播，得到预测结果 outputs
        loss = criterion(outputs, targets)
        # 损失函数：targets告诉model，要它对比想生成的图片，让它自己学习减少误差
        # 计算模型的输出和真实目标之间的损失
        # 损失函数（如均方误差）量化了模型预测与实际目标之间的差异

        loss.backward()  # 回归（不用关注）
        # 执行反向传播，计算关于损失函数的所有模型参数的梯度
        optimizer.step()  # （不用关注）
        # 根据计算出的梯度更新模型的参数。这是优化器的工作，例如Adam或SGD。
# 整个训练过程包括数据的前向传播、损失计算、梯度的反向传播以及参数的更新。这些步骤共同实现了神经网络的训练过程。

# 6. 保存模型
# 保存了训练好的模型
torch.save(model.state_dict(), 'model.pth')

# 7.加载模型，推理
# model.load_state_dict('model.pth')
# output_img = model(new_input_img) #检查 output_img 效果
