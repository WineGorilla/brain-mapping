from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import numpy as np
import matplotlib.pyplot as plt

# def run_pls_2d(B, labels):
#     """
#     使用 PLS 进行 2D 降维分析（模态有监督）
#     参数:
#         B: np.ndarray (N_ROI × N_models)
#         labels: list[str], 模态标签（如 ['lang', 'audio', 'vision', ...]）
#     返回:
#         coords_2d: np.ndarray (N_models × 2)
#         pls: PLSRegression 对象
#     """
#     if B.shape[0] < B.shape[1]:
#         X = B
#     else:
#         X = B.T

#     # === 标准化特征 ===
#     X_std = StandardScaler().fit_transform(X)

#     # === One-hot 编码模态标签 ===
#     enc = OneHotEncoder(sparse_output=False)
#     Y = enc.fit_transform(np.array(labels).reshape(-1, 1))

#     # === PLS 降维 ===
#     pls = PLSRegression(n_components=2)
#     coords_2d = pls.fit_transform(X_std, Y)[0]

#     # === 打印解释度 ===
#     X_var = np.var(coords_2d, axis=0)
#     explained_ratio = np.round(X_var / X_var.sum(), 3)

#     return coords_2d, pls



def run_pls_2d(B, labels):
    if B.shape[0] < B.shape[1]:
        X = B
    else:
        X = B.T

    X_std = (X - X.mean(axis=1, keepdims=True)) / X.std(axis=1, keepdims=True)
    X_std[np.isnan(X_std)] = 0  # 防止除零异常

    enc = OneHotEncoder(sparse_output=False)
    Y = enc.fit_transform(np.array(labels).reshape(-1, 1))

    pls = PLSRegression(n_components=2)
    coords_2d = pls.fit_transform(X_std, Y)[0]

    X_var = np.var(coords_2d, axis=0)
    explained_ratio = np.round(X_var / X_var.sum(), 3)
    
    return coords_2d, pls


def plot_pls_space_2d(coords, model_names, labels, title="PLS Modality Space (2D)", show_labels=False):
    """
    绘制优化版 PLS 模态空间二维散点图
    - 自动调整坐标范围、添加轻微抖动避免重叠
    - 色彩区分模态，支持显示模型标签
    """

    fig, ax = plt.subplots(figsize=(4.5, 3.5))

    # === 模态颜色 ===
    color_map = {"lang": "#2b6cb0", "audio": "#ed8936", "vision": "#38a169"}
    colors = [color_map[l] for l in labels]

    # === 对坐标加轻微抖动，避免重叠太整齐 ===
    jitter = np.random.normal(scale=0.2, size=coords.shape)
    coords_j = coords + jitter

    # === 绘制散点 ===
    for i, (name, label) in enumerate(zip(model_names, labels)):
        x, y = coords_j[i]
        ax.scatter(x, y,
                   color=color_map[label],
                   s=120, alpha=0.85,
                   edgecolors='k', linewidth=0.6)

        if show_labels:
            ax.text(x + 0.2, y + 0.2, name,
                    fontsize=8.5, weight='bold',
                    ha='left', va='center')

    x_min, x_max = np.percentile(coords[:, 0], [5, 95])
    y_min, y_max = np.percentile(coords[:, 1], [5, 95])
    ax.set_xlim(x_min - 2, x_max + 2)
    ax.set_ylim(y_min - 2, y_max + 2)
    #ax.set_aspect('equal', adjustable='datalim')

    ax.set_title(title, fontsize=15, fontweight='bold', pad=12)
    ax.set_xlabel("PLS1", fontsize=11)
    ax.set_ylabel("PLS2", fontsize=11)

    legend_handles = [
        plt.Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=color_map[k],
                   markeredgecolor='k',
                   label=k, markersize=10)
        for k in color_map
    ]
    ax.legend(handles=legend_handles, title="Modality", fontsize=9.5)

    ax.grid(alpha=0.25, linestyle='--')
    plt.tight_layout()
    plt.show()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

def plotp_pls_space_2d(coords, model_names, labels, 
                       title="PLS Modality Space", 
                       show_labels=True,
                       figsize=(5, 4),
                       jitter_scale=0.15,
                       font="Arial"):

    plt.rcParams.update({
        "font.family": font,
        "axes.labelsize": 8,
        "axes.titlesize": 14,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8
    })

    fig, ax = plt.subplots(figsize=figsize)

    style_map = {
        "lang":   {"color": "#4C72B0", "marker": "o"},   
        "audio":  {"color": "#55A868", "marker": "s"},   
        "vision": {"color": "#C44E52", "marker": "^"}   
    }

    jitter = np.random.normal(scale=jitter_scale, size=coords.shape)
    coords_j = coords + jitter

    for i, (name, label) in enumerate(zip(model_names, labels)):
        style = style_map[label]
        x, y = coords_j[i]
        ax.scatter(x, y,
                   color=style["color"],
                   marker=style["marker"],
                   s=30, alpha=0.9,
                   edgecolors='k', linewidth=0.6,
                   zorder=3)
        
        if show_labels:
            ax.text(x + 0.15, y + 0.15, name,
                    fontsize=8.5, fontweight='bold',
                    ha='left', va='center',
                    color='black', alpha=0.9, zorder=4)

    x_pad = (coords[:, 0].max() - coords[:, 0].min()) * 0.25
    y_pad = (coords[:, 1].max() - coords[:, 1].min()) * 0.25
    ax.set_xlim(coords[:, 0].min() - x_pad, coords[:, 0].max() + x_pad)
    ax.set_ylim(coords[:, 1].min() - y_pad, coords[:, 1].max() + y_pad)
    ax.set_aspect('equal', adjustable='box')

    ax.set_xlabel("PLS1", labelpad=5)
    ax.set_ylabel("PLS2", labelpad=5)
    ax.set_title(title, pad=10)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    legend_elements = [
        Line2D([0], [0],
               marker=style_map[m]["marker"],
               color='w',
               label=m.capitalize(),
               markerfacecolor=style_map[m]["color"],
               markeredgecolor='k',
               markersize=5)
        for m in style_map
    ]
    leg = ax.legend(handles=legend_elements, title="Modality",
                    loc="upper right", frameon=False)
    leg.get_title().set_fontweight('bold')

    plt.tight_layout()
    plt.show()
