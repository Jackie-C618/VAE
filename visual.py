import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid
from sklearn.manifold import TSNE
from models import VAE
from utils import get_mnist_dataloaders
import os

# ================== 设置 ==================
LATENT_DIM = 20
DATA_DIR = './data'
MODEL_PATH = './vae_model.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ================== 加载模型 ==================
model = VAE(latent_dim=LATENT_DIM).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
print('模型加载完毕，开始可视化分析...')

# 获取一些测试数据用于后续分析
_, test_loader = get_mnist_dataloaders(DATA_DIR, batch_size=256)
test_data, test_labels = next(iter(test_loader))
test_data = test_data.to(device)
test_labels = test_labels.numpy()

# ========================================================
# 1. 重建图像对比 (原始 vs 重建)
# ========================================================
with torch.no_grad():
    recon, _, _ = model(test_data[:8])          # 取前 8 张做对比
    recon = recon.view(-1, 1, 28, 28).cpu()
    original = test_data[:8].cpu()

# 绘制对比图
fig, axes = plt.subplots(2, 8, figsize=(12, 3))
for i in range(8):
    # 第一行: 原始图像
    axes[0, i].imshow(original[i].squeeze(), cmap='gray')
    axes[0, i].set_title(f'Label: {test_labels[i]}')
    axes[0, i].axis('off')
    # 第二行: 重建图像
    axes[1, i].imshow(recon[i].squeeze(), cmap='gray')
    axes[1, i].axis('off')
axes[0, 0].set_ylabel('Original')
axes[1, 0].set_ylabel('Reconstructed')
plt.suptitle('Original vs Reconstructed Images', y=1.02)
plt.tight_layout()
plt.savefig('reconstruction.png', dpi=150)
plt.show()

# ========================================================
# 2. 生成全新样本 (从标准高斯随机采样)
# ========================================================
with torch.no_grad():
    # 从 N(0, I) 采样 64 个隐向量
    z = torch.randn(64, LATENT_DIM).to(device)
    generated = model.decode(z).view(-1, 1, 28, 28).cpu()

# 排列成 8x8 网格
grid_img = make_grid(generated, nrow=8, normalize=True)
plt.figure(figsize=(8, 8))
plt.imshow(grid_img.permute(1, 2, 0), cmap='gray')
plt.title('Generated Samples from N(0, I)')
plt.axis('off')
plt.tight_layout()
plt.savefig('generated_samples.png', dpi=150)
plt.show()

# ========================================================
# 3. 隐空间可视化 (t-SNE 降维)
# ========================================================
print('正在进行 t-SNE 降维，可能需要几十秒...')
# 获取所有测试数据的隐变量均值 μ
mu_list = []
label_list = []
with torch.no_grad():
    for data, labels in test_loader:
        data = data.to(device)
        mu, _ = model.encode(data)
        mu_list.append(mu.cpu().numpy())
        label_list.append(labels.numpy())

mu_all = np.concatenate(mu_list, axis=0)
labels_all = np.concatenate(label_list, axis=0)

# 用 t-SNE 将 20 维隐空间降到 2 维
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
mu_2d = tsne.fit_transform(mu_all)

# 绘制散点图
plt.figure(figsize=(10, 8))
scatter = plt.scatter(mu_2d[:, 0], mu_2d[:, 1], c=labels_all, cmap='tab10', s=5, alpha=0.7)
plt.colorbar(scatter, ticks=range(10))
plt.title('t-SNE Visualization of Latent Space (μ)')
plt.xlabel('t-SNE Dim 1')
plt.ylabel('t-SNE Dim 2')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('latent_tsne.png', dpi=150)
plt.show()

print('所有可视化结果已保存为图片文件。')