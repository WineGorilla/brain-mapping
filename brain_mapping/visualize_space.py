import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler


# ====================================================
# ✅ 3D PCA 降维
# ====================================================

def run_pca_2d(B):
    """
    对模型特征矩阵进行标准化后，再进行 PCA 降维至 2 维。
    参数:
        B: np.ndarray (N_models, N_features)
    返回:
        coords_2d: np.ndarray (N_models, 2)
        explained: 方差解释率 (list)
    """
    # 确保每行代表一个模型
    if B.shape[0] < B.shape[1]:
        X = B  # 形状: (N_models, N_features)
    else:
        X = B.T

    # === 标准化 ===
    X_std = (X - X.mean(axis=1, keepdims=True)) / X.std(axis=1, keepdims=True)
    X_std[np.isnan(X_std)] = 0  # 防止除零异常

    # === PCA 降维 ===
    pca = PCA(n_components=3)
    coords_2d = pca.fit_transform(X_std)

    explained = pca.explained_variance_ratio_
    print(f"PCA 解释方差比例: {explained}, 总方差: {np.sum(explained):.3f}")

    return coords_2d, explained


def plot_model_space_origin_axes(coords, model_names, title="Brain–Model Mapping Space", color_map='viridis'):
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection='3d')

    # === 绘制模型点 ===
    M = len(model_names)
    colors = plt.get_cmap(color_map)(np.linspace(0, 1, M))
    for i, (name, color) in enumerate(zip(model_names, colors)):
        x, y, z = coords[i]
        ax.scatter(x, y, z, color=color, s=80, label=name, edgecolors='k')
        ax.text(x, y, z, name, fontsize=9, weight='bold')

    # === 坐标边界与居中 ===
    max_range = np.ptp(coords, axis=0).max() / 2.0
    mid = coords.mean(axis=0)
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

    # === 绘制原点坐标轴 ===
    axis_length = max_range * 1.2
    ax.plot([-axis_length, axis_length], [0, 0], [0, 0], color='k', lw=2)  # X
    ax.plot([0, 0], [-axis_length, axis_length], [0, 0], color='k', lw=2)  # Y
    ax.plot([0, 0], [0, 0], [-axis_length, axis_length], color='k', lw=2)  # Z
    ax.text(axis_length, 0, 0, "X", color='k', fontsize=12, weight='bold')
    ax.text(0, axis_length, 0, "Y", color='k', fontsize=12, weight='bold')
    ax.text(0, 0, axis_length, "Z", color='k', fontsize=12, weight='bold')

    # === 美化 ===
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel("Component 1", fontsize=11)
    ax.set_ylabel("Component 2", fontsize=11)
    ax.set_zlabel("Component 3", fontsize=11)
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    ax.set_box_aspect([1, 1, 1])
    ax.view_init(elev=25, azim=35)

    for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        pane.fill = False

    plt.tight_layout()
    plt.show()






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
        #ax.text(x, y, z, name, fontsize=10, weight='bold')

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
def plot_model_space_origin_axes(coords, model_names, labels=None, title="Brain–Model Mapping Space"):
    """
    绘制三模态模型在3D脑空间中的分布（不同形状 + 专业简约风格）
    参数：
        coords: np.ndarray (M, 3) - 模型三维坐标
        model_names: list[str] - 模型名称
        labels: list[str] - 模态标签（'lang', 'audio', 'vision'）
        title: str - 图标题
    """
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    # === 每种模态的颜色与形状 ===
    style_map = {
        "lang":   {"color": "#4C72B0", "marker": "o"},   # 圆
        "audio":  {"color": "#55A868", "marker": "s"},   # 方
        "vision": {"color": "#C44E52", "marker": "^"}    # 三角
    }

    # === 绘制散点 ===
    for i, (x, y, z) in enumerate(coords):
        label = labels[i] if labels is not None else "unknown"
        style = style_map.get(label, {"color": "gray", "marker": "o"})
        ax.scatter(x, y, z,
                   color=style["color"],
                   marker=style["marker"],
                   s=70, alpha=0.9,
                   edgecolors='k', linewidths=0.8)

    # === 原点坐标轴箭头 ===
    max_range = np.ptp(coords, axis=0).max() / 2
    origin = coords.mean(axis=0)
    for vec, col in zip(np.eye(3), ["#666", "#666", "#666"]):
        ax.quiver(origin[0], origin[1], origin[2],
                  vec[0]*max_range*0.7, vec[1]*max_range*0.7, vec[2]*max_range*0.7,
                  color=col, arrow_length_ratio=0.1, linewidth=1.2)

    # === 坐标轴样式 ===
    ax.set_xlabel("Component 1", fontsize=11, labelpad=8)
    ax.set_ylabel("Component 2", fontsize=11, labelpad=8)
    ax.set_zlabel("Component 3", fontsize=11, labelpad=8)
    ax.set_title(title, fontsize=14, weight="bold", pad=12)
    ax.grid(False)
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.pane.fill = False
    ax.set_box_aspect([1, 1, 1])

    # === 图例 ===
    if labels is not None:
        unique_labels = sorted(set(labels))
        handles = [
            plt.Line2D([0], [0],
                       marker=style_map[l]["marker"],
                       color="w",
                       markerfacecolor=style_map[l]["color"],
                       markeredgecolor="none",
                       markersize=8, label=l)
            for l in unique_labels
        ]
        ax.legend(handles=handles, loc="upper right",
                  title="Modality", frameon=False, fontsize=9, title_fontsize=10)

    plt.tight_layout()
    plt.show()