# VAE — 变分自编码器（MNIST）

基于 **PyTorch** 实现的变分自编码器（Variational Autoencoder, VAE），在 **MNIST** 手写数字数据集上完成训练，支持图像重建、从先验分布随机生成新样本，以及对隐空间进行 t-SNE 可视化。

## 项目简介

本项目采用全连接结构的 VAE，将 28×28 的灰度图像（MNIST）编码为低维隐变量分布，再通过重参数化技巧采样并解码重建图像。训练目标是最大化证据下界（ELBO），即最小化「重构损失 + KL 散度」。

- 输入：28×28 单通道灰度图（MNIST）
- 隐空间维度（latent_dim）：20
- 编码器 / 解码器：全连接网络（MLP）
- 训练数据集：MNIST（需提前下载至 `./data` 目录）

## 环境依赖

- Python 3
- torch
- torchvision
- matplotlib
- numpy
- scikit-learn（`visual.py` 中的 t-SNE 降维需要）

可使用 pip 安装：

```bash
pip install torch torchvision matplotlib numpy scikit-learn
```

## 项目结构

```
.
├── models.py          # VAE 模型定义（编码器、重参数化、解码器）
├── utils.py           # 数据加载（MNIST）与 VAE 损失函数
├── train.py           # 训练脚本（训练、绘制 ELBO 曲线、保存模型）
├── visual.py          # 可视化脚本（重建对比、随机生成、隐空间 t-SNE）
├── data/              # MNIST 数据集（需自行放置，代码 download=False）
├── vae_model.pth      # 训练完成后保存的模型权重（state_dict）
├── elbo_curve.png     # 训练 / 测试损失曲线
├── reconstruction.png # 原始 vs 重建图像对比
├── generated_samples.png # 从 N(0, I) 随机生成的样本
└── latent_tsne.png    # 隐空间 μ 的 t-SNE 二维可视化
```

## 模型结构

### 编码器（Encoder）

将输入图像展平后，通过两层全连接 + ReLU 映射到隐变量分布的参数 μ 和 log(σ²)：

```
输入 (B, 1, 28, 28)
  → Flatten → (B, 784)
  → Linear(784 → 400) → ReLU
  → Linear(400 → 200) → ReLU
  → Linear(200 → 20) [fc_mu]  → 均值 μ
  → Linear(200 → 20) [fc_logvar] → 对数方差 log(σ²)
```

### 重参数化（Reparameterization）

从分布 N(μ, σ²) 采样 z，等价于 `z = μ + σ · ε`，其中 `ε ~ N(0, 1)`，以保证梯度可反向传播：

```python
std = torch.exp(0.5 * logvar)
eps = torch.randn_like(std)
z = mu + eps * std
```

### 解码器（Decoder）

将隐变量 z 重建为图像，输出经 Sigmoid 限制在 (0, 1)：

```
z (B, 20)
  → Linear(20 → 200) → ReLU
  → Linear(200 → 400) → ReLU
  → Linear(400 → 784) → Sigmoid
  → reshape 为 (B, 1, 28, 28)
```

## 损失函数

VAE 的损失为**负 ELBO**，由两部分组成（见 `utils.py` 的 `loss_function`）：

- **重构损失（BCE）**：逐像素二元交叉熵
  ```python
  BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum') / batch_size
  ```
- **KL 散度（KLD）**：隐变量分布 N(μ, σ²) 与标准正态 N(0, I) 的 KL 散度闭式解
  ```python
  KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size
  ```

总损失：`loss = BCE + KLD`

## 使用方法

### 1. 准备数据

将 MNIST 数据集放置于 `./data` 目录下（代码 `download=False`，不会自动下载）。预期路径形如：

```
data/MNIST/raw/...      # torchvision 默认读取位置
```

### 2. 训练模型

运行 `train.py` 开始训练，训练完成后会：

- 在控制台逐 epoch 打印 Train / Test 损失；
- 绘制并保存 ELBO 曲线 `elbo_curve.png`；
- 将模型权重保存为 `vae_model.pth`。

```bash
python train.py
```

### 3. 可视化分析

运行 `visual.py`（需已存在 `vae_model.pth`），生成三类可视化结果：

```bash
python visual.py
```

- `reconstruction.png`：原始图像与重建图像对比（前 8 张）；
- `generated_samples.png`：从标准高斯 N(0, I) 随机采样 64 个隐向量解码生成的 8×8 样本网格；
- `latent_tsne.png`：将 20 维隐空间均值 μ 经 t-SNE 降到 2 维并按标签着色的散点图。

## 超参数

`train.py` 中可配置的主要超参数：

| 参数 | 值 | 说明 |
| --- | --- | --- |
| `LATENT_DIM` | 20 | 隐空间维度 |
| `BATCH_SIZE` | 128 | 批大小 |
| `EPOCHS` | 20 | 训练轮次 |
| `LEARNING_RATE` | 1e-3 | 学习率 |
| `DATA_DIR` | `./data` | MNIST 数据目录 |
| `SAVE_MODEL_PATH` | `./vae_model.pth` | 模型保存路径 |

优化器：`torch.optim.Adam`，设备自动选择 CUDA / CPU。

## 数据与预处理

`utils.py` 中 `get_mnist_dataloaders` 使用如下预处理：

- `transforms.ToTensor()`：转换为张量；
- `transforms.Lambda(...)`：加入微小随机噪声并 clamp 到 [0, 1]，以增强数值稳定性。

训练集随机打乱，测试集不打乱。MNIST 为无标签（无监督）任务，训练时仅使用图像输入，忽略标签。
