import torch
import torch.nn as nn

class VAE(nn.Module):
    """
    变分自编码器 (VAE) 模型
    输入: 28x28 的灰度图 (MNIST)
    结构: 全连接编码器 + 重参数化 + 全连接解码器
    """
    def __init__(self, latent_dim=20):
        """
        参数:latent_dim: 隐变量 z 的维度，对于MNIST维度20已足够
        super(VAE, self).__init__() 调用父类 nn.Module 执行初始化
        """
        # super(VAE,self)在当前类 VAE 的上下文中，找到它的父类（即 nn.Module），并绑定当前实例 self
        # 之后调用父类的 __init__() 方法，完成父类的初始化工作
        super(VAE, self).__init__()
        self.latent_dim = latent_dim

        # ================== 编码器 ==================
        # 功能: 将输入图像 x 映射为隐变量分布参数 μ 和 log(σ^2)
        """
        采用全连接层MLP架构:
        输入: (B, 1, 28, 28) -> 展平为 (B, 784)
        B:Batch size,图片数量
        1:通道数,灰度图为1
        隐层: 784 -> 400 -> 200
        """
        """
        input->dense->relu->dense->relu->dense(mu/logvar)
        784->400->200->latent_dim
        """
        self.encoder = nn.Sequential(
            nn.Flatten(),                    # (B, 1, 28, 28) -> (B, 784)
            nn.Linear(784, 400),             # 全连接层 784 -> 400
            nn.ReLU(inplace=True),           # 非线性激活
            nn.Linear(400, 200),             # 400 -> 200
            nn.ReLU(inplace=True),           # 非线性激活
        )
        # 分布参数头部 (输出均值和对数方差)
        self.fc_mu = nn.Linear(200, latent_dim)       # 均值 μ
        self.fc_logvar = nn.Linear(200, latent_dim)   # 对数方差 log(σ^2)

        # ================== 解码器 ==================
        # 功能: 将隐变量 z 重建为图像 x_hat
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 200),      # z -> 200
            nn.ReLU(inplace=True),
            nn.Linear(200, 400),             # 200 -> 400
            nn.ReLU(inplace=True),
            nn.Linear(400, 784),             # 400 -> 784
            nn.Sigmoid(),                    # 输出像素概率，限制在 (0,1)
        )

    def encode(self, x):
        """获取输入 x 的隐变量分布参数 μ 和 logvar"""
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        """
        重参数化: 从 N(μ, σ^2) 采样 z
        等价于: z = μ + σ * ε, 其中 ε ~ N(0, 1)
        """
        std = torch.exp(0.5 * logvar)            # 标准差 σ
        eps = torch.randn_like(std)              # randn_like生成与std相同维度的标准正态分布
        return mu + eps * std

    def decode(self, z):
        """将隐变量 z 重建为图像"""
        return self.decoder(z)

    def forward(self, x):
        """
        前向传播: 编码 -> 采样 -> 解码
        返回:
            recon_x: 重建图像
            mu, logvar: 编码器输出的分布参数
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar