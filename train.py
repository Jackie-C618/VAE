import torch
import torch.optim as optim
from models import VAE
from utils import get_mnist_dataloaders, loss_function
import matplotlib.pyplot as plt
import os

# ================== 超参数 ==================
LATENT_DIM = 20          # 隐空间维度
BATCH_SIZE = 128         # 图片数量
EPOCHS = 20              # 训练轮次
LEARNING_RATE = 1e-3     # 学习率，参数更新步长
DATA_DIR = './data'      # MNIST 文件所在目录
SAVE_MODEL_PATH = './vae_model.pth'  # 模型保存路径

# ================== 设备配置 ==================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')   # print(f)可以直接在其中嵌入变量名

# ================== 加载数据 ==================
"""
torchvision实例化数据时默认路径为root/datasets/MNIST/raw
因此在调用torchvision实例化数据时，应采用root/DATA_DIR/数据名称/raw的形式
download=False
"""
train_loader, test_loader = get_mnist_dataloaders(DATA_DIR, BATCH_SIZE)

# ================== 初始化模型与优化器 ==================
model = VAE(latent_dim=LATENT_DIM).to(device)   # 实例化模型VAE() 并将参数传输到device上
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE) # 优化算法

# ================== 训练循环 ==================
train_losses = []   # 记录每个 epoch 的训练平均损失，在已知图片上的损失，做训练不断拟合
test_losses = []    # 记录每个 epoch 的测试平均损失，在未知图片上的损失，网络参数不变

for epoch in range(1, EPOCHS + 1):
    model.train() # 调用nn.Module的train()方法，启用 BatchNormalization 和 Dropout
    # Dropout 随机关闭一部分神经元以防止过拟合
    # BatchNormalization 在训练时会计算每一层的均值和方差，并在测试时使用这些统计量进行归一化
    train_loss_epoch = 0 # 所有batch的训练损失总和，后续除以batch数量得到平均损失
    # enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标
    # data: batch的输入数据，_ : batch的标签（这里不需要使用标签，所以用 _ 占位）
    # 无监督学习无需标签（根据所给样本特征学习即可），因此在训练循环中只使用输入数据 data，而忽略标签 _。
    # 每当train_loader输出一个batch的数据时，enumerate()会返回该batch的索引和数据
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()

        recon_batch, mu, logvar = model(data) #前向传播
        loss, bce, kld = loss_function(recon_batch, data, mu, logvar)

        loss.backward() # 反向传播计算梯度
        optimizer.step() # 更新模型参数

        train_loss_epoch += loss.item() # loss.item()仅返回一个标量值，loss为tensor包括其他信息

    # 计算当前 epoch 平均训练损失
    avg_train_loss = train_loss_epoch / len(train_loader)
    train_losses.append(avg_train_loss)

    # ================== 测试阶段 ==================
    # eval关闭Dropout和BatchNormalization
    model.eval()
    test_loss_epoch = 0
    # 测试时无需参数更新，禁用计算梯度
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            recon, mu, logvar = model(data)
            loss, _, _ = loss_function(recon, data, mu, logvar)
            test_loss_epoch += loss.item()
    avg_test_loss = test_loss_epoch / len(test_loader)
    test_losses.append(avg_test_loss)

    print(f'Epoch {epoch:2d}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Test Loss: {avg_test_loss:.4f}')

print('训练完成！')

# ================== 绘制 ELBO 曲线 ==================
plt.figure(figsize=(8, 5))
plt.plot(range(1, EPOCHS + 1), train_losses, 'o-', label='Train Negative ELBO')
plt.plot(range(1, EPOCHS + 1), test_losses, 'o-', label='Test Negative ELBO')
plt.xlabel('Epoch')
plt.ylabel('Negative ELBO (lower is better)')
plt.title('VAE Training Curve (ELBO)')
plt.legend()
plt.grid(True)
plt.savefig('elbo_curve.png', dpi=150)
plt.show()

# ================== 保存模型 ==================
torch.save(model.state_dict(), SAVE_MODEL_PATH) # state.dict中包含所有已学习参数的权重和偏置，不包含网格结构信息
print(f'模型已保存至 {SAVE_MODEL_PATH}')