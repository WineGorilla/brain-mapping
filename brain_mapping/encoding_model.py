import numpy as np
from sklearn.model_selection import KFold
import torch

def ridge(stim, resp, alpha, singcutoff=1e-10, normalpha=False):
    stim = np.asarray(stim)
    resp = np.asarray(resp)

    U, S, Vh = np.linalg.svd(stim, full_matrices=False)
    UR = np.dot(U.T, np.nan_to_num(resp))

    if isinstance(alpha, (float, int)):
        alpha = np.ones(resp.shape[1]) * alpha

    norm = S[0]
    nalphas = alpha * norm if normalpha else alpha

    wt = np.zeros((stim.shape[1], resp.shape[1]))
    for ua in np.unique(nalphas):
        selvox = np.nonzero(nalphas == ua)[0]
        awt = Vh.T @ np.diag(S / (S**2 + ua**2)) @ UR[:, selvox]
        wt[:, selvox] = awt
    return wt  # shape: (N_features, N_voxels)

def evaluate_layers_cv(X_layers, y, alpha=1.0, n_splits=5, device='cpu'):
    """
    Cross-validated brain score using ridge regression.
    参数：
      X_layers: list of (T×D) numpy arrays
      y: (T×V) numpy array
      alpha: ridge 参数
      device: torch 计算设备 ('cpu', 'mps', 'cuda')
    返回：
      每层的平均 voxel 相关性
    """
    scores = []
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    y = torch.tensor(y, dtype=torch.float32, device=device)

    for layer, X_np in enumerate(X_layers):
        fold_corrs = []
        if isinstance(X_np, torch.Tensor):
            X_np = X_np.detach().cpu().numpy()

        for train_idx, test_idx in kf.split(X_np):
            # 切分数据集
            X_train_np, X_test_np = X_np[train_idx], X_np[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # 调用ridge进行拟合
            W_np = ridge(X_train_np, y_train.cpu().numpy(), alpha=alpha)
            W = torch.tensor(W_np, dtype=torch.float32, device=device)  # 转 torch

            # 预测
            X_test = torch.tensor(X_test_np, dtype=torch.float32, device=device)
            y_pred = X_test @ W

            # 计算 Pearson correlation（所有 voxel）
            y_test_c = y_test - y_test.mean(0)
            y_pred_c = y_pred - y_pred.mean(0)
            corr = torch.sum(y_test_c * y_pred_c, dim=0) / (
                torch.sqrt(torch.sum(y_test_c ** 2, dim=0)) *
                torch.sqrt(torch.sum(y_pred_c ** 2, dim=0))
            )
            corr[torch.isnan(corr)] = 0.0
            mean_corr = corr.mean().item()
            fold_corrs.append(mean_corr)

        scores.append(np.mean(fold_corrs))

    return np.array(scores)

def get_best_brain_score(scores):
    # 计算每一个voxel与不同layer之间的相关系数，然后对每一个layer中所有的voxel相关系数取均值
    scores = np.array(scores)
    best_layer = int(np.argmax(scores))
    best_score = float(np.max(scores))
    return best_layer, best_score
