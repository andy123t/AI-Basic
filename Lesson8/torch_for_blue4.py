# -*- coding: utf-8 -*-
# 2025-4-8 18:36:05
# Author: Andy

# Fix error in Windows
# import os
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# import numpy as np
import dataset
import plot_utils
import torch
import torch.nn as nn
import torch.optim as optim

# 从数据中获取随机豆豆
m = 100
X, Y = dataset.get_beans4(m)
print(X)
print(Y)
plot_utils.show_scatter(X, Y)

# 转换为 Tensor 类型
X = torch.from_numpy(X).float()
Y = torch.from_numpy(Y).float()

# 修改：将 X 转换为二维张量
# X = X.unsqueeze(1)

# 定义模型
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(2, 2)  # 输入层到隐藏层，隐藏层有 10 个神经元
        self.sigmoid = nn.Sigmoid()  # 隐藏层的激活函数
        self.fc2 = nn.Linear(2, 1)  # 隐藏层到输出层
        self.sigmoid = nn.Sigmoid()  # 输出层的激活函数

    def forward(self, x):
        out = self.fc1(x)
        out = self.sigmoid(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out

model = SimpleNet()

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.05)

# 训练模型
epochs = 100000
batch_size = 10
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs.squeeze(), Y)
    loss.backward()
    optimizer.step()

    with torch.no_grad():
        predicted_labels = (outputs.squeeze() >= 0.5).float()
        accuracy = (predicted_labels == Y).float().mean()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch: {epoch + 1}/{epochs}, Loss: {loss.item():.4f}, Accuracy: {accuracy.item():.2f}')

# 进行预测
# with torch.no_grad():
#     predictions = model(X)
#     predicted_labels = (predictions.squeeze() >= 0.5).float()
#     accuracy = (predicted_labels == Y).float().mean()
#     print(f'Final Accuracy: {accuracy.item():.4f}')

plot_utils.show_scatter_surface(X, Y, model)

