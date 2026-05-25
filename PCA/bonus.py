import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch

# --- 核心科学计算与绘图 ---
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# --- 附加任务 1: 非线性降维 ---
from sklearn.manifold import TSNE
import umap

# --- 附加任务 2: 深度学习 (VAE) ---
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# ==============================================================================
# 0. 统一数据准备与预处理
# ==============================================================================
print("正在加载并预处理数据...")
digits = load_digits()

# 筛选数字 '3' 的数据用于可视化
mask_3 = (digits.target == 3)
data_3_raw = digits.data[mask_3]
images_3_raw = digits.images[mask_3]

# 数据标准化：这对大多数 ML 方法非常重要，尤其是 UMAP 和深度学习
scaler = StandardScaler()
data_3_std = scaler.fit_transform(data_3_raw)

# 用于 VAE 训练的完整数据集（所有数字）
data_full_std = scaler.fit_transform(digits.data)

# 将 VAE 训练数据转换为 PyTorch 张量
X_train_tensor = torch.FloatTensor(data_full_std)
train_dataset = TensorDataset(X_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)


# ==============================================================================
# 1. 附加任务 1: 实现 t-SNE 或 UMAP 的可视化
# ==============================================================================
print("正在执行 t-SNE 和 UMAP...")

# 方法 1.1: t-SNE (t-Distributed Stochastic Neighbor Embedding)
tsne = TSNE(n_components=2, random_state=42)
projections_tsne = tsne.fit_transform(data_3_std)

# 方法 1.2: UMAP (Uniform Manifold Approximation and Projection)
reducer_umap = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
projections_umap = reducer_umap.fit_transform(data_3_std)


# ==============================================================================
# 2. 附加任务 2: 实现深度学习基础的方法 (VAE)
# ==============================================================================
print("正在训练 VAE (这可能需要几分钟)...")

# 定义 VAE 架构
class VAE(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=32, latent_dim=2):
        super(VAE, self).__init__()
        
        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        # 获取均值 μ 和方差倒数 log(σ^2)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_log_var = nn.Linear(hidden_dim, latent_dim)
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid() # 像素值范围在0-1之间
        )

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.encoder(x)
        mu, log_var = self.fc_mu(h), self.fc_log_var(h)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decoder(z)
        return x_recon, mu, log_var

# 定义损失函数 (重构误差 + KL 散度)
def loss_function(x_recon, x, mu, log_var):
    BCE = nn.functional.binary_cross_entropy(x_recon, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCE + KLD

# 初始化和训练
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vae_model = VAE().to(device)
optimizer = optim.Adam(vae_model.parameters(), lr=1e-3)

epochs = 30
vae_model.train()
for epoch in range(epochs):
    train_loss = 0
    for batch_idx, (data,) in enumerate(train_loader):
        data = data.to(device)
        # 像素值归一化，这对 Sigmoid 激活是必须的
        data_norm = (data - data.min()) / (data.max() - data.min())
        
        optimizer.zero_grad()
        recon_batch, mu, log_var = vae_model(data_norm)
        loss = loss_function(recon_batch, data_norm, mu, log_var)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    # print(f"Epoch {epoch+1}/{epochs}, Loss: {train_loss / len(train_loader.dataset):.4f}")

# 提取 '3' 数据在 VAE 潜在空间中的映射
vae_model.eval()
with torch.no_grad():
    data_3_tensor = torch.FloatTensor(data_3_std).to(device)
    _, mu_3, _ = vae_model(data_3_tensor)
    latent_spaces_vae = mu_3.cpu().numpy()


# ==============================================================================
# 3. 统一可视化与对比
# ==============================================================================
print("正在生成可视化对比图...")

# 创建 2x2 子图布局
fig, axs = plt.subplots(2, 2, figsize=(14, 12), gridspec_kw={'wspace': 0.15, 'hspace': 0.3})
plt.suptitle("B. Advance task: Multi-method Dimensionality Reduction for Digit '3'", fontsize=16)

# 辅助函数：绘制散点图和图像网格，模拟原图风格
def plot_method_results(ax, points, images, title, x_label='Dim 1', y_label='Dim 2'):
    ax.scatter(points[:, 0], points[:, 1], s=5, c='lime', alpha=0.6, label='_nolegend_')
    ax.axhline(0, color='black', linewidth=1)
    ax.axvline(0, color='black', linewidth=1)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    # 在每个散点图内模拟一个 5x5 的原始图像格
    n_rows, n_cols = 5, 5
    grid_img = np.zeros((n_rows * 8, n_cols * 8))
    for i in range(n_rows):
        for j in range(n_cols):
            idx = i * n_cols + j
            if idx < len(images):
                grid_img[i*8:(i+1)*8, j*8:(j+1)*8] = images[idx]
    
    # 将图像放在一个小的 inset_axes 中
    ax_ins = ax.inset_axes([0.65, 0.65, 0.33, 0.33])
    ax_ins.imshow(grid_img, cmap='binary', origin='upper')
    for i in range(n_rows + 1):
        ax_ins.axhline(i * 8 - 0.5, color='red', linewidth=1)
        ax_ins.axvline(i * 8 - 0.5, color='red', linewidth=1)
    ax_ins.set_axis_off()
    ax_ins.set_title("Sample Images")

# 方法 A (作为基准): PCA
# 实际上 PCA 已经被封装在 StandardScaler 之后的 fit_transform 过程中
pca_ref = PCA(n_components=2)
projections_pca = pca_ref.fit_transform(data_3_std)
plot_method_results(axs[0, 0], projections_pca, images_3_raw, "Reference (PCA)", "First PC", "Second PC")

# 附加任务 1: t-SNE
plot_method_results(axs[0, 1], projections_tsne, images_3_raw, "Method: t-SNE", "Dimension 1", "Dimension 2")

# 附加任务 1: UMAP
plot_method_results(axs[1, 0], projections_umap, images_3_raw, "Method: UMAP", "Dimension 1", "Dimension 2")

# 附加任务 2: VAE 潜在空间
# 模拟原图风格，在 VAE 潜在空间上画出均值位置
plot_method_results(axs[1, 1], latent_spaces_vae, images_3_raw, "Method: VAE Latent Space (Learned from all digits)", "VAE Z1 (Mean μ)", "VAE Z2 (Mean μ)")
# 在 VAE 图中，我们稍微强调一下原点
axs[1, 1].axhline(0, color='red', linewidth=1, linestyle='--')
axs[1, 1].axvline(0, color='red', linewidth=1, linestyle='--')

plt.tight_layout()
plt.show()

print("任务完成！")