import numpy as np
import matplotlib.pyplot as plt

def plot_polynomial_custom(params_list):
    """
    params_list: 列表，每个元素可以是 (N, M) 或 (N, M, ln_lambda)
    """
    n_plots = len(params_list)
    n_cols = 2 if n_plots > 1 else 1
    n_rows = (n_plots + 1) // 2
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8 * n_cols, 6.5 * n_rows))
    axes = np.array(axes).flatten()

    for i, task in enumerate(params_list):
        # 1. 动态解析参数
        N = task[0]
        M = task[1]
        ln_lam = task[2] if len(task) == 3 else None
        
        # 2. 生成数据
        np.random.seed(45)
        x_train = np.linspace(0, 1, N)
        y_train = np.sin(2 * np.pi * x_train) + np.random.normal(0, 0.2, N)
        
        # 3. 构造范德蒙德矩阵
        X_train = np.vander(x_train, M + 1, increasing=True)
        
        # --- 4. 算法逻辑切换 ---
        if ln_lam is None:
            # 两个参数：不带正则化，使用伪逆强行拟合噪点
            weights = np.linalg.pinv(X_train) @ y_train
            title_label = f"N={N}, M={M}(No Regularization)"
            color = 'r' # 红色表示可能存在震荡
        else:
            # 三个参数：带正则化，使用岭回归公式
            lam = np.exp(ln_lam)
            XTX = X_train.T @ X_train
            # 岭回归解析解公式
            weights = np.linalg.inv(XTX + lam * np.eye(M + 1)) @ X_train.T @ y_train
            title_label = f"N={N}, M={M}(Regularized $\ln \lambda = {ln_lam}$)"
            color = 'b' # 蓝色表示受控平滑

        # 5. 绘图
        x_test = np.linspace(0, 1, 1000)
        X_test = np.vander(x_test, M + 1, increasing=True)
        y_pred = X_test @ weights
        y_true = np.sin(2 * np.pi * x_test)

        ax = axes[i]
        ax.plot(x_test, y_true, 'g', label=r'$\sin(2\pi x)$')
        ax.plot(x_test, y_pred, color, label='Model Prediction')
        ax.scatter(x_train, y_train, facecolors='none', edgecolors='b', s=80, label='Data')
        
        ax.set_title(title_label)
        ax.set_xlabel('x')
        ax.set_ylabel('t')
        
        # 为了对比明显，统一 M>=9 时的 Y 轴显示范围
        if M >= 9 and ln_lam is None:
            ax.set_ylim(-3.5, 3.5)
        else:
            ax.set_ylim(-1.5, 1.5)
            
        ax.legend(prop={'size': 9})
        ax.grid(True, linestyle='--', alpha=0.5)

    # 隐藏空余子图
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    tasks = [
        (10, 3),    
        (10, 9),
        (15, 9),
        (100, 9),
        (10, 9, -18)
    ]
    
    plot_polynomial_custom(tasks)