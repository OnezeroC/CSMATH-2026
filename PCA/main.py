import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA

# 1. 加载 UCI 手写数字数据集
digits = load_digits()

# 2. 仅筛选数字 '3' 的数据
# digits.data 包含了 8x8 的像素展开后的 64 维向量
mask = (digits.target == 3)
data_3 = digits.data[mask]
images_3 = digits.images[mask]

# 3. 执行 PCA，降维至 2 个分量
pca = PCA(n_components=2)
projected = pca.fit_transform(data_3)

# 4. 绘图准备
plt.figure(figsize=(12, 6))

# 左侧：PCA 散点图
plt.subplot(1, 2, 1)
plt.scatter(projected[:, 0], projected[:, 1], s=5, c='lime', alpha=0.6)

# 设置坐标轴和网格线 (模拟图中样式)
plt.axhline(0, color='black', linewidth=1)
plt.axvline(0, color='black', linewidth=1)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('PCA of Digit 3')

# 5. 右侧：展示部分数字 '3' 的原始图像 (模拟 5x5 网格)
plt.subplot(1, 2, 2)
n_rows, n_cols = 5, 5
grid_img = np.zeros((n_rows * 8, n_cols * 8))

for i in range(n_rows):
    for j in range(n_cols):
        idx = i * n_cols + j
        if idx < len(images_3):
            grid_img[i*8:(i+1)*8, j*8:(j+1)*8] = images_3[idx]

plt.imshow(grid_img, cmap='binary')
# 给每个小格子画红框 (模拟图中视觉效果)
for i in range(n_rows + 1):
    plt.axhline(i * 8 - 0.5, color='red', linewidth=1)
    plt.axvline(i * 8 - 0.5, color='red', linewidth=1)

plt.axis('off')
plt.title('Sample Images of 3')

plt.tight_layout()
plt.show()