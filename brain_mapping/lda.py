import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler

def run_lda_3d(B, labels):
    """
    对模型特征矩阵进行标准化后，使用 LDA 降维至 3 维。
    参数:
        B: np.ndarray (N_models, N_features)
        labels: list[str]，每个模型对应的模态标签（如 'lang', 'audio', 'vision'）
    返回:
        coords_3d: np.ndarray (N_models, 3)
        lda: 训练后的 LDA 对象
    """
    # === 确保每行代表一个模型 ===
    if B.shape[0] < B.shape[1]:
        X = B
    else:
        X = B.T

    # === 标准化 ===
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)

    # === LDA 降维 ===
    n_components = min(3, len(np.unique(labels)) - 1)  # LDA维度 = 类别数-1
    lda = LinearDiscriminantAnalysis(n_components=n_components)
    coords_3d = lda.fit_transform(X_std, labels)

    print(f"LDA 完成，输出维度: {n_components}")
    explained = np.var(coords_3d, axis=0) / np.sum(np.var(coords_3d, axis=0))
    print(f"投影后方差比例（近似解释度）: {np.round(explained, 3)}")

    return coords_3d, lda



def plot_lda_space_origin_axes_2d(coords, model_names, labels, title="LDA Modality Space (2D)"):
    """
    绘制二维 LDA 投影空间，带模态颜色与原点坐标轴。
    """
    fig, ax = plt.subplots(figsize=(8, 7))

    color_map = {"lang": "#2b6cb0", "audio": "#ed8936", "vision": "#38a169"}
    colors = [color_map[l] for l in labels]

    # === 绘制点 ===
    for i, (name, color) in enumerate(zip(model_names, colors)):
        x, y = coords[i]
        ax.scatter(x, y, color=color, s=90, edgecolors='k', alpha=0.9)
        ax.text(x, y, name, fontsize=3, ha='center', va='bottom', weight='bold')

    # === 坐标范围与原点轴 ===
    max_range = np.ptp(coords, axis=0).max() / 2.0
    mid = coords.mean(axis=0)
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range*2)
    ax.set_ylim(mid[1], mid[1])

    axis_length = max_range * 1.2
    ax.plot([-axis_length, axis_length], [0, 0], color='k', lw=1.8)
    ax.plot([0, 0], [-axis_length, axis_length], color='k', lw=1.8)
    ax.text(axis_length * 1.05, 0, "LD1", color='k', fontsize=12, weight='bold')
    ax.text(0, axis_length * 1.05, "LD2", color='k', fontsize=12, weight='bold')

    # === 美化 ===
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel("LD1", fontsize=11)
    ax.set_ylabel("LD2", fontsize=11)
    ax.legend(
        handles=[plt.Line2D([0], [0], marker='o', color='w',
                            markerfacecolor=color_map[k], label=k, markersize=9, markeredgecolor='k')
                 for k in color_map],
        title="Modality", fontsize=9
    )
    ax.set_aspect('equal', adjustable='box')
    ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.show()
