import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from model import DigitCNN
import random
import os

def predict():
    """
    加载训练好的模型并进行预测演示。
    """
    # -----------------------------------------------------------
    # 1. 环境与模型加载
    # -----------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 实例化模型结构
    model = DigitCNN().to(device)
    
    model_path = "weights/mnist_cnn.pth"
    if not os.path.exists(model_path):
        print(f"错误：找不到模型文件 {model_path}。请先运行 train.py。")
        return

    # 加载权重
    # map_location 确保即使在 GPU 训练但在 CPU 预测也能正常工作
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval() # 评估模式
    print("模型加载成功！")

    # -----------------------------------------------------------
    # 2. 准备测试数据
    # -----------------------------------------------------------
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # 下载测试集 (如果已存在则直接读取)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    
    # -----------------------------------------------------------
    # 3. 随机抽取一张图片进行预测
    # -----------------------------------------------------------
    # 随机选择一个索引
    idx = random.randint(0, len(test_dataset) - 1)
    image_tensor, label = test_dataset[idx]
    
    # 增加 batch 维度
    # PyTorch 模型期望输入形状为 [batch, channel, height, width]
    # 原始形状 [1, 28, 28] -> 扩展为 [1, 1, 28, 28]
    input_batch = image_tensor.unsqueeze(0).to(device)
    
    # -----------------------------------------------------------
    # 4. 执行推理 (Inference)
    # -----------------------------------------------------------
    with torch.no_grad():
        output = model(input_batch)
        # 获取概率最大的索引作为预测结果
        prediction = output.argmax(dim=1).item()

    print(f"当前图片索引: {idx}")
    print(f"真实标签 (Ground Truth): {label}")
    print(f"模型预测 (Prediction): {prediction}")

    # -----------------------------------------------------------
    # 5. 可视化结果
    # -----------------------------------------------------------
    # 将 Tensor 转回 numpy 数组以便绘图
    # squeeze() 去除维度为 1 的轴: [1, 28, 28] -> [28, 28]
    image_display = image_tensor.squeeze().numpy()
    
    plt.figure(figsize=(4, 4))
    plt.imshow(image_display, cmap='gray')
    plt.title(f"Truth: {label} | Pred: {prediction}")
    plt.axis('off') # 不显示坐标轴
    plt.show()

if __name__ == "__main__":
    predict()
