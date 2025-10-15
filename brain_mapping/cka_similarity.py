import numpy as np

def linear_CKA(X,Y):
    '''
    计算两个表示之间的CKA相似度
    '''
    X = X - X.mean()
    Y = Y - Y.mean()
    numerator = np.linalg.norm(np.dot(X.T, Y)) ** 2
    denominator = np.linalg.norm(np.dot(X.T, X)) * np.linalg.norm(np.dot(Y.T, Y))
    return numerator / denominator if denominator > 0 else 0.0

def compute_model_similarity(B):
    '''
    计算模型之间的相似度矩阵
    '''
    M = B.shape[1]
    S = np.zeros((M,M))
    for i in range(M):
        for j in range(M):
            S[i, j] = linear_CKA(B[:, i], B[:, j])

    return S

