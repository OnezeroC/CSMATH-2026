import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb

def bernstein_poly(i, n, t):
    """计算第 i 个 n 阶伯恩斯坦基函数"""
    return comb(n, i) * (t**(i)) * (1 - t)**(n - i)

def plot_bezier_fits(params_list):
    n_plots = len(params_list)
    n_cols = 2 if n_plots > 1 else 1
    n_rows = (n_plots + 1) // 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8 * n_cols, 6.5 * n_rows))
    axes = np.array(axes).flatten()

    for idx, task in enumerate(params_list):
        # 参数解析
        N = task[0]
        M = task[1] # 阶数 (Degree)
        ln_lam = task[2] if len(task) == 3 else None

        # 1. 数据生成
        np.random.seed(45)
        x_train = np.linspace(0, 1, N)
        y_train = np.sin(2 * np.pi * x_train) + np.random.normal(0, 0.2, N)
        
        # 转换为张量以便处理
        t_train = torch.tensor(x_train, dtype=torch.float64)
        target = torch.tensor(y_train, dtype=torch.float64)

        # 2. 构造伯恩斯坦特征矩阵 (Bernstein Matrix)
        # 每一列是一个伯恩斯坦基函数在所有采样点的值
        X = torch.zeros((N, M + 1), dtype=torch.float64)
        for i in range(M + 1):
            X[:, i] = torch.tensor([bernstein_poly(i, M, t.item()) for t in t_train])

        # 3. 计算控制点 (系数)
        if ln_lam is None:
            # 最小二乘/伪逆解
            weights = torch.linalg.pinv(X) @ target
            title = f"Bezier N={N}, M={M}\n(No Reg)"
        else:
            # 带正则化的岭回归解
            lam = np.exp(ln_lam)
            XTX = X.T @ X
            weights = torch.linalg.inv(XTX + lam * torch.eye(M + 1)) @ X.T @ target
            title = f"Bezier N={N}, M={M}\n(ln $\lambda$={ln_lam})"

        # 4. 生成平滑曲线
        t_test = torch.linspace(0, 1, 1000, dtype=torch.float64)
        X_test = torch.zeros((1000, M + 1), dtype=torch.float64)
        for i in range(M + 1):
            X_test[:, i] = torch.tensor([bernstein_poly(i, M, t.item()) for t in t_test])
        y_pred = X_test @ weights

        # 5. 绘图
        ax = axes[idx]
        ax.plot(t_test.numpy(), np.sin(2 * np.pi * t_test.numpy()), 'g', label='True')
        ax.plot(t_test.numpy(), y_pred.numpy(), 'r', label='Bezier Fit')
        ax.scatter(x_train, y_train, facecolors='none', edgecolors='b', label='Data')
        
        # 特色：绘制控制点 (Bezier Control Points)
        # 贝塞尔曲线的一个特性是它被包裹在控制点的凸包内
        ax.plot(np.linspace(0, 1, M+1), weights.numpy(), '--ko', alpha=0.3, label='Control Points')
        
        ax.set_title(title)
        ax.set_ylim(-1.5, 1.5)
        ax.legend(fontsize='small')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 测试不同组合
    tasks = [
        (10, 3),      # 低阶贝塞尔
        (10, 9),      # 高阶贝塞尔
    ]
    plot_bezier_fits(tasks)