import numpy as np
import pytest
from test_pca import pca, pca_svd

def test_pca_basic():
    # 基本的な2次元データでのテスト
    X = np.array([[2, 2], [1, -1], [-1, 1], [-2, -2]])
    P, T = pca(X)
    
    # 主成分ベクトルが正規直交であることを確認
    assert np.allclose(P.T @ P, np.eye(2))
    
    # 変換後のデータの分散が主成分方向で最大になることを確認
    T_cov = np.cov(T.T)
    assert np.allclose(np.diag(T_cov), np.sort(np.diag(T_cov))[::-1])

def test_pca_svd_basic():
    # SVDベースのPCAの基本的なテスト
    X = np.array([[2, 2], [1, -1], [-1, 1], [-2, -2]])
    P, T = pca_svd(X)
    
    # 主成分ベクトルが正規直交であることを確認
    assert np.allclose(P.T @ P, np.eye(2))
    
    # 変換後のデータの分散が主成分方向で最大になることを確認
    T_cov = np.cov(T.T)
    assert np.allclose(np.diag(T_cov), np.sort(np.diag(T_cov))[::-1])

def test_pca_consistency():
    # 通常のPCAとSVDベースのPCAが同じ結果を返すことを確認
    X = np.array([[2, 2], [1, -1], [-1, 1], [-2, -2]])
    P1, T1 = pca(X)
    P2, T2 = pca_svd(X)
    
    # 主成分ベクトルが同じ方向を向いていることを確認（符号は異なる可能性あり）
    assert np.allclose(np.abs(P1), np.abs(P2))
    
    # 変換後のデータが同じ分布を持つことを確認
    assert np.allclose(np.abs(T1), np.abs(T2))

def test_pca_3d():
    # 3次元データでのテスト
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
    P, T = pca(X)
    
    # 主成分ベクトルが正規直交であることを確認
    assert np.allclose(P.T @ P, np.eye(3))
    
    # 変換後のデータの分散が主成分方向で最大になることを確認
    T_cov = np.cov(T.T)
    assert np.allclose(np.diag(T_cov), np.sort(np.diag(T_cov))[::-1]) 