import numpy as np

def pca(X):
    # データを中心化
    X_centered = X - np.mean(X, axis=0)
    # SVDを使用して主成分を計算
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    P = Vt.T
    T = X_centered @ P
    return P, T

def pca_svd(X):
    # データを中心化
    X_centered = X - np.mean(X, axis=0)
    # 共分散行列のSVD
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    P = Vt.T
    T = X_centered @ P
    return P, T

# テスト用のデータ
if __name__ == "__main__":
    X = np.array([[2, 2], [1, -1], [-1, 1], [-2, -2]])
    P, T = pca(X)
    print("PCA principal components:")
    print(P)

    P, T = pca_svd(X)
    print("\nSVD-based PCA principal components:")
    print(P)