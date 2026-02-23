import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import DigitCNN
import os

def train():
    """
    训练主函数。负责数据加载、模型初始化、训练循环和保存模型。
    """
    # -----------------------------------------------------------
    # 1. 环境配置
    # -----------------------------------------------------------
    # 检查是否有 GPU (CUDA)，如果没有则使用 CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"当前运行设备: {device}")

    # 超参数配置 (Hyperparameters)
    BATCH_SIZE = 64         # 每一批次训练的样本数
    LEARNING_RATE = 0.001   # 学习率，控制参数更新的步长
    EPOCHS = 5              # 训练轮数 (遍历整个数据集的次数)
    
    # -----------------------------------------------------------
    # 2. 数据准备
    # -----------------------------------------------------------
    # 定义数据预处理流程
    # ToTensor: 将图像 (PIL Image 或 numpy) 转为 Tensor，并归一化到 [0, 1]
    # Normalize: 标准化。mean=0.1307, std=0.3081 是 MNIST 数据集的统计值
    # 公式: image = (image - mean) / std
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    print("正在加载 MNIST 数据集 (如果不存在将自动下载)...")
    # 训练集
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # 测试集 (用于评估)
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    # -----------------------------------------------------------
    # 3. 初始化模型与优化器
    # -----------------------------------------------------------
    model = DigitCNN().to(device) # 将模型移动到指定设备 (GPU/CPU)
    
    # 优化器: Adam 是一种自适应学习率的优化算法，通常比 SGD 收敛更快
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 损失函数: 负对数似然损失 (Negative Log Likelihood Loss)
    # 适用于分类问题，且模型输出层使用了 LogSoftmax
    criterion = nn.NLLLoss()

    # -----------------------------------------------------------
    # 4. 训练循环
    # -----------------------------------------------------------
    for epoch in range(1, EPOCHS + 1):
        model.train() # 切换模型到训练模式 (启用 Dropout 等)
        
        for batch_idx, (data, target) in enumerate(train_loader):
            # 将数据移动到设备
            data, target = data.to(device), target.to(device)
            
            # [关键步骤 1] 梯度清零
            # PyTorch 会累加梯度，因此每次反向传播前必须手动清零
            optimizer.zero_grad()
            
            # [关键步骤 2] 前向传播 (Forward Pass)
            # 计算模型对输入的预测输出
            output = model(data)
            
            # [关键步骤 3] 计算损失 (Loss)
            # 衡量预测值 output 与真实标签 target 之间的差距
            loss = criterion(output, target)
            
            # [关键步骤 4] 反向传播 (Backward Pass)
            # 计算损失相对于模型参数的梯度
            loss.backward()
            
            # [关键步骤 5] 参数更新
            # 根据梯度更新模型权重
            optimizer.step()
            
            # 打印训练进度
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] '
                      f'Loss: {loss.item():.6f}')
        
        # 每个 Epoch 结束后进行测试
        evaluate(model, device, test_loader, criterion)

    # -----------------------------------------------------------
    # 5. 保存模型
    # -----------------------------------------------------------
    if not os.path.exists('weights'):
        os.makedirs('weights')
    save_path = "weights/mnist_cnn.pth"
    torch.save(model.state_dict(), save_path)
    print(f"\n训练完成！模型权重已保存至: {save_path}")

def evaluate(model, device, test_loader, criterion):
    """
    在测试集上评估模型性能。
    """
    model.eval() # 切换模型到评估模式 (关闭 Dropout)
    test_loss = 0
    correct = 0
    
    # torch.no_grad(): 评估时不需要计算梯度，可以节省显存并加速
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            # 累加 batch 损失
            test_loss += criterion(output, target).item()
            
            # 获取预测结果
            # output shape: [batch, 10] -> argmax 获取概率最大的类别索引
            pred = output.argmax(dim=1, keepdim=True) 
            
            # 统计预测正确的数量
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    
    print(f'\n[验证结果] 平均损失: {test_loss:.4f}, 准确率: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n')

if __name__ == '__main__':
    train()
