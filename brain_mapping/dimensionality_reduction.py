from sklearn.manifold import MDS
import umap

def reduce_model_space(S, method="MDS", n_components=3, random_state=42):
    """
    将模型相似度矩阵降维到3D空间。
    参数：
        S: np.ndarray (M, M) 相似度矩阵
        method: str, 'MDS' 或 'UMAP'
    返回：
        coords: np.ndarray (M, 3)
    """
    # 转换为距离矩阵
    D = 1 - S

    if method == "MDS":
        mds = MDS(n_components=n_components, dissimilarity="precomputed", random_state=random_state)
        coords = mds.fit_transform(D)
    elif method == "UMAP":
        reducer = umap.UMAP(n_components=n_components, metric="precomputed", random_state=random_state)
        coords = reducer.fit_transform(D)
    else:
        raise ValueError("method must be 'MDS' or 'UMAP'")

    return coords
