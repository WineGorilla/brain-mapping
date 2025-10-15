from .encoding_model import ridge
import numpy as np
import torch

'''
ROI数据格式，前多少个feature属于哪一个脑区
roi_dict = {
    "V1": np.arange(0, 300),
    "STG": np.arange(300, 800),
    "IFG": np.arange(800, 1200)
}

'''


def compute_roi_fit(X_best, y_roi, alpha=1.0):
    """
    计算单个 ROI 的拟合程度（平均 voxel 相关）
    参数：
        X_best: np.ndarray, shape (T, D) 模型最佳层 embedding
        y_roi: np.ndarray, shape (T, V_roi) 脑区对应的 fMRI 信号
        alpha: float, 岭回归正则化参数
    返回：
        roi_score: float，该 ROI 的平均 Pearson 相关系数
    """
    X = np.asarray(X_best)
    Y = np.asarray(y_roi)

    # Ridge 拟合
    W = ridge(X, Y, alpha=alpha)
    Y_pred = X @ W

    # 去均值后计算 Pearson r
    Y_c = Y - Y.mean(0)
    Yp_c = Y_pred - Y_pred.mean(0)
    corr = np.sum(Y_c * Yp_c, axis=0) / (
        np.sqrt(np.sum(Y_c ** 2, axis=0)) * np.sqrt(np.sum(Yp_c ** 2, axis=0))
    )
    corr[np.isnan(corr)] = 0.0

    # 返回该 ROI 所有 voxel 的平均相关系数
    return float(np.mean(corr))

def compute_model_fit_all_rois(X_best, y, roi_dict, alpha=1.0):
    """
    计算模型的最佳层在所有 ROI 上的解释力。
    参数：
        X_best: np.ndarray, shape (T, D)
        y: np.ndarray, shape (T, V)
        roi_dict: dict[str, np.ndarray]，每个 ROI 的 voxel 索引
        alpha: float，Ridge 正则参数
    返回：
        roi_vector: np.ndarray, shape (N_ROI,)
            每个 ROI 的平均 Pearson 相关，表示该模型在各 ROI 的解释力
        roi_scores: dict[str, float]
            ROI 名称 → 对应 brain score
    """
    if isinstance(X_best, torch.Tensor):
        X_best = X_best.detach().cpu().numpy()
    if isinstance(y, torch.Tensor):
        y = y.detach().cpu().numpy()
    roi_scores = {}
    for roi, voxels in roi_dict.items():
        y_roi = y[:, voxels]
        score = compute_roi_fit(X_best, y_roi, alpha=alpha)
        roi_scores[roi] = score
        print(f"ROI {roi}: {score:.4f}")

    roi_vector = np.array(list(roi_scores.values()))
    return roi_vector, roi_scores
