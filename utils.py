import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os

def get_mnist_dataloaders(data_dir='./data', batch_size=128):
    """
    加载 MNIST 数据集并返回训练/测试 DataLoader
    参数:
        data_dir: 存放 MNIST 原始文件的目录 (默认 ./data)
        batch_size: 批大小
    返回:
        train_loader, test_loader
    """
    # 数据预处理流水线
    # 使用clamp截断，if x < 0.0, set to 0.0; if x > 1.0, set to 1.0; else keep x
    transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: (x + 0.001 * torch.rand_like(x)).clamp(0.0, 1.0))
])

    # 训练集 (注意: train=True 会读取 train文件,否则读取 t10k文件)
    train_dataset = datasets.MNIST(
        root=data_dir, train=True, transform=transform, download=False
    )
    # 测试集
    test_dataset = datasets.MNIST(
        root=data_dir, train=False, transform=transform, download=False
    )

    # 构建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, test_loader


def loss_function(recon_x, x, mu, logvar):
    """
    VAE 损失函数: 负 ELBO = 重构损失 + KL 散度
    参数:
        recon_x: 解码器输出的重建图像
        x: 原始输入图像
        mu, logvar: 编码器输出的隐变量分布参数
    返回:
        总损失, 重构损失, KL 散度 (用于日志记录)
    """
    # 1. 重构损失: 逐像素二元交叉熵 (BCE)
    #    recon_x 形状 (B, 784), x 形状 (B, 1, 28, 28) -> 需要flatten
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum') / x.size(0)

    # 2. KL 散度: D_KL( N(μ,σ^2) || N(0,I) )
    #    闭式解: -0.5 * sum( 1 + log(σ^2) - μ^2 - σ^2 )
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)

    return BCE + KLD, BCE, KLD