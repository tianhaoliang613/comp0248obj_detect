import torch
import torch.nn as nn
import torch.nn.functional as F

class DigitCNN(nn.Module):
    """
    用于 MNIST 手写数字识别的卷积神经网络 (CNN)。
    
    继承自 nn.Module，这是 PyTorch 中构建神经网络的基类。
    """
    def __init__(self):
        """
        初始化神经网络层。
        
        架构设计：
        1. 卷积层 1 (Conv1): 提取基础特征。
        2. 池化层 (Pool): 下采样，减少参数量。
        3. 卷积层 2 (Conv2): 提取高级特征。
        4. 全连接层 (Fully Connected): 进行分类。
        """
        super(DigitCNN, self).__init__()
        
        # -----------------------------------------------------------
        # 第一层卷积块
        # -----------------------------------------------------------
        # nn.Conv2d 参数说明:
        # in_channels=1: 输入图片是灰度图，单通道。
        # out_channels=32: 使用 32 个卷积核，输出 32 个特征图 (Feature Maps)。
        # kernel_size=3: 卷积核大小为 3x3。
        # padding=1: 填充 1 圈 0，保证卷积后图像尺寸不变 (28x28 -> 28x28)。
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        
        # -----------------------------------------------------------
        # 第二层卷积块
        # -----------------------------------------------------------
        # in_channels=32: 承接上一层的 32 个输出通道。
        # out_channels=64: 增加特征深度到 64。
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        
        # -----------------------------------------------------------
        # 全连接层 (分类器)
        # -----------------------------------------------------------
        # 维度计算说明:
        # 1. 输入: [Batch, 1, 28, 28]
        # 2. Conv1 -> [Batch, 32, 28, 28]
        # 3. MaxPool1 (2x2) -> [Batch, 32, 14, 14]
        # 4. Conv2 -> [Batch, 64, 14, 14]
        # 5. MaxPool2 (2x2) -> [Batch, 64, 7, 7]
        # 6. Flatten 展平 -> 64 * 7 * 7 = 3136 个特征点
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10) # 10个数字分类 (0-9)
        
        # Dropout: 训练时随机将 50% 的神经元置零，防止过拟合
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        """
        定义前向传播逻辑 (数据流向)。
        
        参数:
            x (Tensor): 输入图像数据，形状为 [batch_size, 1, 28, 28]
        """
        # Block 1
        x = self.conv1(x)
        x = F.relu(x)          # 激活函数: 引入非线性
        x = F.max_pool2d(x, 2) # 最大池化: 28x28 -> 14x14
        
        # Block 2
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2) # 最大池化: 14x14 -> 7x7
        #基本上图像分类问题最后都要缩小成7*7的特征图，然后进行全连接层分类
         
        # Flatten: 将多维特征图拉平成一维向量
        # start_dim=1 表示保持 batch 维度不变，从 channel 维度开始展平
        x = torch.flatten(x, 1) 
        
        # Fully Connected Block
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)    # 仅在训练时生效
        x = self.fc2(x)
        
        # 输出层使用 Log Softmax，配合 NLLLoss 计算损失
        output = F.log_softmax(x, dim=1)
        return output
