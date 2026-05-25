import torch
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. Levenberg-Marquardt 算法核心实现
# ==========================================
def lm_algorithm(func_res, x0, max_iter=100, tol=1e-6, lam=0.01):
    """
    func_res: 目标残差函数 (返回 torch.tensor)
    x0: 初始点 [x, y]
    """
    p = torch.tensor(x0, dtype=torch.float32)
    history = [p.numpy().copy()]
    
    # 获取计算雅可比矩阵的函数
    from torch.func import jacrev
    get_jacobian = jacrev(func_res)

    for i in range(max_iter):
        # 计算当前残差和损失
        r = func_res(p)
        current_loss = 0.5 * torch.sum(r**2)
        
        # 计算雅可比矩阵 J (m x n)
        J = get_jacobian(p)
        
        # 构造正规方程: (J^T J + λI) δ = -J^T r
        jtj = J.T @ J
        rhs = -J.T @ r
        
        # 求解步长 delta
        # 使用 λI 阻尼项，确保矩阵正定且可逆
        I = torch.eye(len(p))
        delta = torch.linalg.solve(jtj + lam * I, rhs)
        
        # 尝试更新
        new_p = p + delta
        new_r = func_res(new_p)
        new_loss = 0.5 * torch.sum(new_r**2)
        
        # 策略：如果误差减少，减小 λ (向高斯-牛顿靠拢)；否则增加 λ (向梯度下降靠拢)
        if new_loss < current_loss:
            lam /= 10
            p = new_p
            history.append(p.numpy().copy())
            # 收敛判断
            if torch.norm(delta) < tol:
                break
        else:
            lam *= 10
            
    return np.array(history)

# ==========================================
# 2. 定义 5 个测试函数 (返回残差向量)
# ==========================================

def rosenbrock(p):
    # 极小值在 (1, 1), 经典的狭长谷底
    return torch.stack([10 * (p[1] - p[0]**2), 1 - p[0]])

def booth(p):
    # 极小值在 (1, 3)
    return torch.stack([p[0] + 2*p[1] - 7, 2*p[0] + p[1] - 5])

def beale(p):
    # 极小值在 (3, 0.5)
    x, y = p[0], p[1]
    return torch.stack([
        1.5 - x * (1 - y),
        2.25 - x * (1 - y**2),
        2.625 - x * (1 - y**3)
    ])

def himmelblau(p):
    # 四个局部极小值，例如 (3, 2), (-2.8, 3.13) 等
    return torch.stack([
        p[0]**2 + p[1] - 11,
        p[0] + p[1]**2 - 7
    ])

def circle_fit(p):
    # 模拟圆拟合残差：寻找圆心 (x,y)，让它距离点 (1,0) 和 (0,1) 的距离接近半径 1
    # 这是一个简化的演示
    x, y = p[0], p[1]
    return torch.stack([
        torch.sqrt(x**2 + y**2) - 1, # 距离原点距离为1
        x + y - 1.414                # 位于某直线上
    ])

# ==========================================
# 3. 可视化函数
# ==========================================

def plot_test_case(func, history, name, ax):
    # 确定绘图范围
    all_x = history[:, 0]
    all_y = history[:, 1]
    margin = 1.5
    x_min, x_max = all_x.min() - margin, all_x.max() + margin
    y_min, y_max = all_y.min() - margin, all_y.max() + margin
    
    x = np.linspace(x_min, x_max, 100)
    y = np.linspace(y_min, y_max, 100)
    X, Y = np.meshgrid(x, y)
    
    # 计算 Loss 表面
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            res = func(torch.tensor([X[i,j], Y[i,j]], dtype=torch.float32))
            Z[i,j] = 0.5 * torch.sum(res**2).item()
            
    # 绘图
    ax.contourf(X, Y, np.log1p(Z), levels=20, cmap='viridis', alpha=0.8)
    ax.plot(history[:, 0], history[:, 1], 'r.-', label='LM Steps')
    ax.plot(history[0, 0], history[0, 1], 'go', label='Start')
    ax.plot(history[-1, 0], history[-1, 1], 'bo', label='End')
    ax.set_title(f"{name}\nSteps: {len(history)-1}")
    ax.legend(fontsize='small')

# ==========================================
# 4. 执行与展示
# ==========================================

test_cases = [
    (rosenbrock, [-1.5, 2.0], "Rosenbrock"),
    (booth, [0.0, 0.0], "Booth"),
    (beale, [1.0, 1.0], "Beale"),
    (himmelblau, [0.0, 0.0], "Himmelblau"),
    (circle_fit, [2.0, 2.0], "Circle Optimization")
]

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for i, (func, start_pos, name) in enumerate(test_cases):
    history = lm_algorithm(func, start_pos)
    plot_test_case(func, history, name, axes[i])

# 移除最后一个多余的子图
fig.delaxes(axes[-1])
plt.tight_layout()
plt.show()