import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap

# 1. 加载全量数据 (0-9)
digits = load_digits()
X = digits.data
y = digits.target
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

# 2. 执行三种算法
print("正在处理全数据集降维...")
pca_2d = PCA(n_components=2).fit_transform(X_std)
tsne_2d = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto').fit_transform(X_std)
umap_2d = umap.UMAP(n_components=2, random_state=42).fit_transform(X_std)

# 3. 可视化对比
fig, axs = plt.subplots(1, 3, figsize=(20, 6))
methods = [('PCA', pca_2d), ('t-SNE', tsne_2d), ('UMAP', umap_2d)]

for i, (name, coords) in enumerate(methods):
    ax = axs[i]
    # 使用 scatter 的 c 参数传入标签 y，cmap 调用色彩光谱
    scatter = ax.scatter(coords[:, 0], coords[:, 1], c=y, cmap='tab10', s=5, alpha=0.7)
    ax.set_title(f'Method: {name}')
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # 在第一个图旁边添加图例
    if i == 2:
        legend = ax.legend(*scatter.legend_elements(), loc="center left", bbox_to_anchor=(1, 0.5), title="Digits")
        ax.add_artist(legend)

plt.tight_layout()
plt.show()