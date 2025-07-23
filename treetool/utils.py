"""
MIT License - Copyright (c) 2021 porteratzo
このファイルは、pclpyへの依存をなくし、Open3DとNumPyのみを
使用するように修正されました。
"""
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
import open3d as o3d

def rotation_matrix_from_vectors(vector1, vector2):
    """
    vector1をvector2に揃えるための回転行列を見つける。
    """
    a, b = (vector1 / np.linalg.norm(vector1)).reshape(3), (vector2 / np.linalg.norm(vector2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    if s < 1e-8: # ベクトルがほぼ平行
        return np.eye(3) if c > 0 else -np.eye(3)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix

def angle_between_vectors(vector1,vector2):
    """
    2つのベクトル間の角度を見つける。
    """
    value = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
    return np.arccos(np.clip(value, -1.0, 1.0))

def makecylinder(model=[0,0,0,0,0,1,0.1],height = 1,density=10):
    """
    7パラメータの円筒モデルから点群を生成する。
    """
    radius = model[6]
    X,Y,Z = model[:3]
    h_steps = np.linspace(0, height, density)
    theta = np.linspace(0, 2*np.pi, density)
    
    theta_grid, z_grid = np.meshgrid(theta, h_steps)
    x_grid = radius * np.cos(theta_grid)
    y_grid = radius * np.sin(theta_grid)
    
    cyl = np.vstack([x_grid.flatten(), y_grid.flatten(), z_grid.flatten()]).T
    
    rotation = rotation_matrix_from_vectors([0,0,1], model[3:6])
    rotated_cylinder = cyl @ rotation.T + np.array([X,Y,Z])
    return rotated_cylinder

def getPrincipalVectors(A):
    """
    (0,0,0)中心の行列の主成分ベクトルと主成分値を取得する。
    """
    if A.shape[0] < A.shape[1]:
        return np.eye(A.shape[1]), np.zeros(A.shape[1])
        
    covariance_matrix = np.cov(A, rowvar=False)
    values, vectors = np.linalg.eig(covariance_matrix)
    
    sort_indices = np.argsort(values)[::-1]
    return vectors[:, sort_indices].T, values[sort_indices]

def similarize(test, target):
    """
    ベクトル間の角度がpi/2より大きい場合、その方向を反転させる。
    """
    if angle_between_vectors(test, target) > np.pi/2:
        return -test
    return test