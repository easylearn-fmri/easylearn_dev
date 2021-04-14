'''
    线性回归：
            输入        输出
            0.5      5.0
            0.6      5.5
            0.8      6.0
            1.1      6.8
            1.4      7.0
            ...
            y = f(x)

        预测函数：y = w0+w1x
                x: 输入
                y: 输出
                w0和w1: 模型参数

        所谓模型训练，就是根据已知的x和y，找到最佳的模型参数w0 和 w1，尽可能精确地描述出输入和输出的关系。
            如：5.0 = w0 + w1 × 0.5       5.5 = w0 + w1 × 0.6

        单样本误差：根据预测函数求出输入为x时的预测值：y' = w0 + w1x，单样本误差为1/2(y' - y)2。

        总样本误差：把所有单样本误差相加即是总样本误差：1/2 Σ(y' - y)2

        损失函数：loss = 1/2 Σ(w0 + w1x - y)2
            损失函数就是总样本误差关于模型参数w0 w1的函数，该函数属于三维数学模型，即需要找到一组w0 w1使得loss取极小值。

        示例：画图模拟梯度下降的过程
            1>整理训练集数据，自定义梯度下降算法规则，求出w0 ， w1 ，绘制回归线。
            2>绘制随着每次梯度下降，w0，w1，loss的变化曲线。
            3>基于三维曲面绘制梯度下降过程中的每一个点。
            4>基于等高线的方式绘制梯度下降的过程。
'''

import numpy as np
import matplotlib.pyplot as mp
from mpl_toolkits.mplot3d import axes3d
import warnings

warnings.filterwarnings('ignore')

train_x = np.array([0.5, 0.6, 0.8, 1.1, 1.4])
train_y = np.array([5.0, 5.5, 6.0, 6.8, 7.0])

# 实现梯度下降的过程
times = 1000  # 迭代次数
lrate = 0.01  # 学习率，取值不应太大
w0, w1 = [1], [1]  # 初始化模型参数，记录每次梯度下降的参数
losses = []  # 保存每次迭代过程中损失函数值
epoches = []  # 保存每次迭代过程的索引
for i in range(1, times + 1):
    # 输出每次下降时：w0,w1,loss值的变化
    epoches.append(i)
    loss = ((w0[-1] + w1[-1] * train_x - train_y) ** 2).sum() / 2
    losses.append(loss)
    print('{:4}> w0={:.6f},w1={:.6f},loss={:.6f}'.format(epoches[-1], w0[-1], w1[-1], losses[-1]))

    # 每次梯度下降过程，需要求出w0和w1的修正值，求修正值需要推导loss函数在w0及w1方向的偏导数
    d0 = (w0[-1] + w1[-1] * train_x - train_y).sum()
    d1 = ((w0[-1] + w1[-1] * train_x - train_y) * train_x).sum()
    # w0和w1的值不断修正
    w0.append(w0[-1] - lrate * d0)
    w1.append(w1[-1] - lrate * d1)
print(w0[-1], w1[-1])

pred_y = w0[-1] + w1[-1] * train_x

# 绘制样本点
mp.figure('Linear Regression', facecolor='lightgray')
mp.title('Linear Regression')
mp.grid(linestyle=':')
mp.scatter(train_x, train_y, s=60, c='orangered', label='Samples', marker='o')
# 绘制回归线
mp.plot(train_x, pred_y, color='dodgerblue', label='Regression Line')
mp.legend()

# 绘制随着每次梯度下降，w0，w1，loss的变化曲线。
mp.figure('BGD Params', facecolor='lightgray')
mp.title('BGD Params')
mp.tick_params(labelsize=10)
mp.subplot(311)
mp.title('w0')
mp.plot(epoches, w0[:-1], color='dodgerblue', label='w0')
mp.grid(linestyle=':')
mp.legend()

mp.subplot(312)
mp.title('w1')
mp.plot(epoches, w1[:-1], color='orangered', label='w1')
mp.grid(linestyle=':')
mp.legend()

mp.subplot(313)
mp.title('loss')
mp.plot(epoches, losses, color='yellowgreen', label='loss')
mp.grid(linestyle=':')
mp.legend()

# 基于三维曲面绘制梯度下降过程中的每一个点。
# 整理网格点坐标矩阵，计算每个点的loss绘制曲面

grid_w0, grid_w1 = np.meshgrid(np.linspace(0, 9, 500), np.linspace(0, 3.5, 500))
grid_loss = np.zeros_like(grid_w0)
for x, y in zip(train_x, train_y):
    grid_loss += ((grid_w0 + grid_w1 * x - y) ** 2) / 2
# 绘制3D损失函数图
mp.figure('Loss Function', facecolor='lightgray')
ax3d = mp.gca(projection='3d')
ax3d.set_xlabel('w0')
ax3d.set_ylabel('w1')
ax3d.set_zlabel('loss')
ax3d.plot_surface(grid_w0, grid_w1, grid_loss, cmap='jet')
# 绘制3D梯度下降曲线图
ax3d.plot(w0[:-1], w1[:-1], losses, 'o-', color='orangered', label='BGD', zorder=3)
mp.tight_layout()

# 基于等高线的方式绘制梯度下降的过程。
mp.figure('BGD Contour', facecolor='lightgray')
mp.title('BGD Contour')
mp.xlabel('w0')
mp.ylabel('w1')
mp.grid(linestyle=':')
cntr = mp.contour(grid_w0, grid_w1, grid_loss, c='black', linewidths=0.5)
mp.clabel(cntr, fmt='%.2f', inline_spacing=0.2, fontsize=8)
mp.contourf(grid_w0, grid_w1, grid_loss, cmap='jet')
mp.plot(w0[:-1], w1[:-1], c='orangered', label='BGD')
mp.legend()

# mp.show()输出结果：4.065692318299849 2.2634176028710415