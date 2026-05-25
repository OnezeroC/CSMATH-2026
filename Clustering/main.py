import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from sklearn.cluster import KMeans

# ==========================================
# 0. 公共数据生成 (固定随机种子以确保一致性)
# ==========================================
def generate_shared_data(n_samples=400, seed=42):
    np.random.seed(seed) # 固定种子
    
    # 定义两个高斯分布的参数 (一个紧凑，一个松散且倾斜)
    mu1, sigma1 = [2, 2], [[0.5, 0], [0, 0.5]]
    mu2, sigma2 = [6, 6], [[1.8, 0.6], [0.6, 1.8]]
    
    # 按照概率 p1=0.4, p2=0.6 生成 MOG 数据
    n1 = int(n_samples * 0.4)
    n2 = n_samples - n1
    
    data1 = np.random.multivariate_normal(mu1, sigma1, n1)
    data2 = np.random.multivariate_normal(mu2, sigma2, n2)
    
    X = np.vstack((data1, data2))
    return X, (mu1, mu2) # 返回数据和真实均值用于参考

# ==========================================
# 1. Mean-Shift 算法实现
# ==========================================
class MeanShift:
    def __init__(self, bandwidth=2.0, tol=1e-3, max_iter=100):
        self.bandwidth = bandwidth
        self.tol = tol
        self.max_iter = max_iter

    def fit(self, data):
        centroids = np.copy(data)
        n_samples = len(centroids)
        
        for i in range(n_samples):
            for _ in range(self.max_iter):
                current_p = centroids[i]
                distances = np.linalg.norm(data - current_p, axis=1)
                within_window = data[distances < self.bandwidth]
                
                if len(within_window) == 0: break
                
                new_p = np.mean(within_window, axis=0)
                if np.linalg.norm(new_p - current_p) < self.tol: break
                centroids[i] = new_p
        
        self.centroids = self._group_centroids(centroids)
        self.labels = self._assign_labels(data, self.centroids)

    def _group_centroids(self, centroids):
        unique_centroids = []
        if len(centroids) > 0:
            unique_centroids.append(centroids[0])
            for c in centroids[1:]:
                # 合并距离小于 bandwidth/2 的中心
                if np.all(np.linalg.norm(np.array(unique_centroids) - c, axis=1) > self.bandwidth / 2):
                    unique_centroids.append(c)
        return np.array(unique_centroids)

    def _assign_labels(self, data, centroids):
        labels = []
        for d in data:
            dist = np.linalg.norm(centroids - d, axis=1)
            labels.append(np.argmin(dist))
        return np.array(labels)

# ==========================================
# 2. EM 算法实现 MoG
# ==========================================
def em_mog(X, k=2, max_iter=100, tol=1e-4):
    n_samples, n_features = X.shape
    
    # 初始化: 使用 K-means 的结果作为初始均值 (比随机初始化更稳定)
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
    means = kmeans.cluster_centers_
    weights = np.ones(k) / k
    covs = [np.eye(n_features) for _ in range(k)]
    
    resp = np.zeros((n_samples, k))
    
    for i in range(max_iter):
        prev_means = means.copy()
        
        # E-step
        for j in range(k):
            resp[:, j] = weights[j] * multivariate_normal.pdf(X, means[j], covs[j], allow_singular=True)
        
        # 处理可能的数值不稳定 (避免除以零)
        row_sums = resp.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1e-12 
        resp /= row_sums
        
        # M-step
        Nk = resp.sum(axis=0)
        weights = Nk / n_samples
        
        for j in range(k):
            means[j] = np.sum(resp[:, j][:, np.newaxis] * X, axis=0) / (Nk[j] + 1e-9)
            diff = X - means[j]
            # 增加正则化项以防协方差矩阵奇异
            reg_term = 1e-6 * np.eye(n_features)
            covs[j] = (resp[:, j][:, np.newaxis, np.newaxis] * np.einsum('ni,nj->nij', diff, diff)).sum(axis=0) / (Nk[j] + 1e-9) + reg_term

        if np.linalg.norm(means - prev_means) < tol: break
            
    return means, np.argmax(resp, axis=1)

# ==========================================
# 3. 谱聚类 (Spectral Clustering) 实现
# ==========================================
def spectral_clustering(X, k=2, sigma=1.0):
    n_samples = X.shape[0]
    
    # 1. 构建相似度矩阵 W (RBF Kernel)
    # 计算全距离矩阵
    dists = np.sum(X**2, axis=1).reshape(-1, 1) + np.sum(X**2, axis=1) - 2 * np.dot(X, X.T)
    W = np.exp(-dists / (2 * sigma**2))
    np.fill_diagonal(W, 0) # 自身相似度设为 0
    
    # 2. 构建归一化拉普拉斯矩阵 L_sym = I - D^-1/2 * W * D^-1/2
    D = np.diag(np.sum(W, axis=1))
    D_diag = np.diagonal(D).copy()
    D_diag[D_diag == 0] = 1e-9 # 避免除以零
    D_inv_sqrt = np.diag(1.0 / np.sqrt(D_diag))
    L_sym = np.eye(n_samples) - D_inv_sqrt @ W @ D_inv_sqrt
    
    # 3. 特征分解，取前 k 个最小特征值对应的特征向量
    eigvals, eigvecs = np.linalg.eigh(L_sym)
    features = eigvecs[:, :k]
    
    # 4. 对特征空间进行 K-means 聚类
    # 归一化特征向量行 (常用步骤)
    row_norms = np.linalg.norm(features, axis=1, keepdims=True)
    row_norms[row_norms == 0] = 1e-9
    features = features / row_norms
    
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = kmeans.fit_predict(features)
    
    return labels

# ==========================================
# 4. 执行所有算法并绘图对比
# ==========================================
# 运行数据
X, true_means = generate_shared_data(400)

# 设置超参数 (这些对结果影响很大)
ms_bandwidth = 2.2 # Mean-Shift 带宽
em_k = 2           # MoG 成分数
sc_k = 2           # 谱聚类簇数
sc_sigma = 1.2     # 谱聚类 RBF 核参数

# A. 运行 Mean-Shift
ms = MeanShift(bandwidth=ms_bandwidth)
ms.fit(X)

# B. 运行 EM-MoG
em_means, em_labels = em_mog(X, k=em_k)

# C. 运行 Spectral Clustering
sc_labels = spectral_clustering(X, k=sc_k, sigma=sc_sigma)

# 绘图对比
fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharex=True, sharey=True)
plt.subplots_adjust(wspace=0.1) # 减小子图间距

# 统一散点图参数
scatter_kwargs = {'marker': 'o', 'alpha': 0.5, 'edgecolors': 'w', 'linewidths': 0.5, 's': 30}
cmap = 'viridis'

# 图 1: Mean-Shift
axes[0].scatter(X[:, 0], X[:, 1], c=ms.labels, cmap=cmap, **scatter_kwargs)
axes[0].scatter(ms.centroids[:, 0], ms.centroids[:, 1], c='red', marker='x', s=150, linewidths=4, label='Estimated Modes')
axes[0].set_title(f"1. Mean-Shift (BW={ms_bandwidth})\nFound {len(ms.centroids)} clusters")
axes[0].legend()
axes[0].grid(True, linestyle='--', alpha=0.3)

# 图 2: EM-MoG
# 将 em_labels 映射到 viridis 颜色空间 (为了看起来颜色和别的图一致)
axes[1].scatter(X[:, 0], X[:, 1], c=em_labels, cmap=cmap, **scatter_kwargs)
axes[1].scatter(em_means[:, 0], em_means[:, 1], c='red', marker='P', s=150, linewidths=2, label='Estimated Means ($\mu$)')
# 绘制真实均值作为对比
true_means_arr = np.array(true_means)
axes[1].scatter(true_means_arr[:, 0], true_means_arr[:, 1], c='cyan', marker='o', s=150, facecolors='none', linewidths=3, label='Ground Truth $\mu$')
axes[1].set_title(f"2. EM-MoG (K={em_k})\nParameter Estimation")
axes[1].legend()
axes[1].grid(True, linestyle='--', alpha=0.3)

# 图 3: Spectral Clustering
axes[2].scatter(X[:, 0], X[:, 1], c=sc_labels, cmap=cmap, **scatter_kwargs)
axes[2].set_title(f"3. Spectral Clustering (K={sc_k}, $\sigma$={sc_sigma})\nGraph-based connectivity")
axes[2].grid(True, linestyle='--', alpha=0.3)

plt.suptitle("Comparison of Clustering Algorithms on the Same MoG Dataset", fontsize=16, fontweight='bold', y=1.02)
plt.show()