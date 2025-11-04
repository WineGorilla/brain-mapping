from .encoding_model import ridge
import numpy as np
import torch
from numpy.linalg import svd


def compute_roi_fit(X_best, y_roi, alpha=1.0):

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

def ridge_fit(stim, resp, alpha):
    U, S, Vh = svd(stim, full_matrices=False)
    UR = U.T @ resp
    return Vh.T @ np.diag(S / (S**2 + alpha**2)) @ UR

def layer_roi_pearson_matrix(X_layers, Y, alpha=1.0, device='cpu'):

    X_layers = np.asarray(X_layers)
    Y = np.asarray(Y)

    n_layers, T, D = X_layers.shape
    V = Y.shape[1]

    pearson = np.zeros((n_layers, V))

    for li in range(n_layers):
        X = X_layers[li]

        # Ridge fit
        W = ridge_fit(X, Y, alpha)

        # Predict
        Y_pred = X @ W

        # Pearson per voxel
        Yc = Y - Y.mean(0)
        Ypc = Y_pred - Y_pred.mean(0)

        corr = np.sum(Yc * Ypc, 0) / (
            np.sqrt(np.sum(Yc**2, 0)) * np.sqrt(np.sum(Ypc**2, 0))
        )

        corr[np.isnan(corr)] = 0.0
        pearson[li] = corr
    return pearson




def get_best_layer(pearson_mat):

    pearson_mat = np.asarray(pearson_mat)

    # 每层平均 ROI 分数
    layer_mean_scores = pearson_mat.mean(axis=1)

    # 找全脑表现最强的层
    best_layer = int(np.argmax(layer_mean_scores))
    best_layer_scores = pearson_mat[best_layer]  

    # # 每个 ROI 分别找最强层
    # roi_best_layers = np.argmax(pearson_mat, axis=0)

    # print("每层平均score:", np.round(layer_mean_scores, 4))
    # print(f"最佳层: {best_layer} | score={layer_mean_scores[best_layer]:.4f}")

    return best_layer, best_layer_scores #, layer_mean_scores, roi_best_layers
