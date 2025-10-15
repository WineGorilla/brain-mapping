import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def plot_model_space(coords, model_names, title="Brain–Model Mapping Space", color_map='viridis'):
    """
    在3D空间中绘制模型分布。
    参数：
        coords: np.ndarray (M, 3) - 每个模型的三维坐标
        model_names: list[str] - 模型名称
        title: str - 图标题
        color_map: str - matplotlib 颜色映射方案
    """
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    M = len(model_names)
    colors = plt.get_cmap(color_map)(np.linspace(0, 1, M))

    for i, (name, color) in enumerate(zip(model_names, colors)):
        x, y, z = coords[i]
        ax.scatter(x, y, z, color=color, s=80, label=name, edgecolors='k')
        ax.text(x, y, z, name, fontsize=10, weight='bold')

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel("Component 1", fontsize=12)
    ax.set_ylabel("Component 2", fontsize=12)
    ax.set_zlabel("Component 3", fontsize=12)
    ax.legend(loc="best", fontsize=9)
    plt.tight_layout()
    plt.show()


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_model_space_origin_axes(coords, model_names, title="Brain–Model Mapping Space", color_map='viridis'):
    """
    绘制模型分布的3D空间图，坐标轴固定穿过原点 (0, 0, 0)
    """
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # === 绘制模型点 ===
    M = len(model_names)
    colors = plt.get_cmap(color_map)(np.linspace(0, 1, M))
    for i, (name, color) in enumerate(zip(model_names, colors)):
        x, y, z = coords[i]
        ax.scatter(x, y, z, color=color, s=80, label=name, edgecolors='k')
        ax.text(x, y, z, name, fontsize=10, weight='bold')

    # === 计算边界范围 ===
    max_range = np.array([
        coords[:, 0].max() - coords[:, 0].min(),
        coords[:, 1].max() - coords[:, 1].min(),
        coords[:, 2].max() - coords[:, 2].min()
    ]).max() / 2.0

    mid_x = (coords[:, 0].max() + coords[:, 0].min()) * 0.5
    mid_y = (coords[:, 1].max() + coords[:, 1].min()) * 0.5
    mid_z = (coords[:, 2].max() + coords[:, 2].min()) * 0.5

    # 统一坐标范围，使原点居中
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    # === 绘制穿过原点的三条黑色坐标轴 ===
    axis_length = max_range * 1.2
    ax.plot([-axis_length, axis_length], [0, 0], [0, 0], color='k', lw=2)  # X轴
    ax.plot([0, 0], [-axis_length, axis_length], [0, 0], color='k', lw=2)  # Y轴
    ax.plot([0, 0], [0, 0], [-axis_length, axis_length], color='k', lw=2)  # Z轴

    # === 标注原点坐标轴 ===
    ax.text(axis_length, 0, 0, "X", color='k', fontsize=12, weight='bold')
    ax.text(0, axis_length, 0, "Y", color='k', fontsize=12, weight='bold')
    ax.text(0, 0, axis_length, "Z", color='k', fontsize=12, weight='bold')

    # === 其他外观美化 ===
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel("Component 1", fontsize=12)
    ax.set_ylabel("Component 2", fontsize=12)
    ax.set_zlabel("Component 3", fontsize=12)
    ax.legend(loc="upper left", fontsize=9, frameon=False)
    ax.set_box_aspect([1, 1, 1])
    ax.view_init(elev=25, azim=35)

    # 移除默认灰色背景
    for spine in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        spine.fill = False

    plt.tight_layout()
    plt.show()
